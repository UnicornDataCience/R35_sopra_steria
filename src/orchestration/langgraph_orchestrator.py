from typing import Dict, Any, List
from langgraph.graph import Graph, StateGraph, END, START
from langgraph.prebuilt import ToolExecutor
from langchain.schema import BaseMessage
from pydantic import BaseModel
import asyncio

class AgentState(BaseModel):
    """Estado compartido entre agentes"""
    messages: List[BaseMessage] = []
    current_agent: str = "coordinator"
    context: Dict[str, Any] = {}
    dataset_uploaded: bool = False
    analysis_complete: bool = False
    generation_complete: bool = False
    validation_complete: bool = False
    simulation_complete: bool = False
    evaluation_complete: bool = False
    synthetic_data: Any = None
    user_input: str = ""
    next_action: str = ""

class MedicalAgentsOrchestrator:
    """Orquestador de agentes médicos usando LangGraph"""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        if "coordinator" not in self.agents:
            from ..agents.coordinator_agent import CoordinatorAgent
            self.agents["coordinator"] = CoordinatorAgent()
    
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """Crea el workflow de agentes con LangGraph"""
        
        # Definir el grafo de estados
        workflow = StateGraph(AgentState)
        
        # Añadir nodos (agentes)
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("simulator", self._simulator_node)
        workflow.add_node("evaluator", self._evaluator_node)
        
        # Definir flujo condicional
        workflow.add_conditional_edges(
            "coordinator",
            self._route_from_coordinator,
            {
                "analyzer": "analyzer",
                "generator": "generator",
                "validator": "validator",
                "simulator": "simulator",
                "evaluator": "evaluator",
                "end": END
            }
        )
        
        # Flujo secuencial principal
        workflow.add_edge("analyzer", "generator")
        workflow.add_edge("generator", "validator")  
        workflow.add_edge("validator", "simulator")
        workflow.add_edge("simulator", "evaluator")
        workflow.add_edge("evaluator", END)
        
        # Punto de entrada
        workflow.set_entry_point("coordinator")
        
        return workflow.compile()
    
    def _route_from_coordinator(self, state: AgentState) -> str:
        """Enruta desde el coordinador según el estado"""
        
        user_input = state.user_input.lower()
        
        if "archivo" in user_input or "dataset" in user_input:
            return "analyzer"
        elif state.analysis_complete and not state.generation_complete:
            return "generator"
        elif state.generation_complete and not state.validation_complete:
            return "validator"
        elif state.validation_complete and not state.simulation_complete:
            return "simulator"
        elif state.simulation_complete and not state.evaluation_complete:
            return "evaluator"
        else:
            return "end"
    
    async def _coordinator_node(self, state: AgentState) -> AgentState:
        """Nodo coordinador principal"""
        
        coordinator_agent = self.agents.get("coordinator")
        
        if coordinator_agent:
            response = await coordinator_agent.process(
                state.user_input,
                state.context
            )
            
            # Actualizar estado basado en respuesta
            if "archivo" in response.get("message", "").lower():
                state.next_action = "request_upload"
            
        return state
    
    async def _analyzer_node(self, state: AgentState) -> AgentState:
        """Nodo del agente analista"""
        
        analyzer_agent = self.agents.get("analyzer")
        
        if analyzer_agent and state.dataset_uploaded:
            # Obtener dataset del contexto
            dataframe = state.context.get("dataframe")
            
            if dataframe is not None:
                response = await analyzer_agent.analyze_dataset(
                    dataframe, 
                    state.context
                )
                
                # Actualizar estado
                state.analysis_complete = True
                state.context["analysis_result"] = response
        
        return state
    
    async def _generator_node(self, state: AgentState) -> AgentState:
        """Nodo del agente generador"""
        
        generator_agent = self.agents.get("generator")
        
        if generator_agent and state.analysis_complete:
            response = await generator_agent.process(
                f"Generar datos sintéticos basados en: {state.context.get('analysis_result', {})}",
                state.context
            )
            
            state.generation_complete = True
            state.context["generation_result"] = response
        
        return state
    
    async def _validator_node(self, state: AgentState) -> AgentState:
        """Nodo del agente validador"""
        
        validator_agent = self.agents.get("validator")
        
        if validator_agent and state.generation_complete:
            response = await validator_agent.process(
                "Validar coherencia médica de los datos sintéticos generados",
                state.context
            )
            
            state.validation_complete = True
            state.context["validation_result"] = response
        
        return state
    
    async def _simulator_node(self, state: AgentState) -> AgentState:
        """Nodo del agente simulador"""
        
        simulator_agent = self.agents.get("simulator")
        
        if simulator_agent and state.validation_complete:
            response = await simulator_agent.process(
                "Simular evolución temporal de pacientes sintéticos",
                state.context
            )
            
            state.simulation_complete = True
            state.context["simulation_result"] = response
        
        return state
    
    async def _evaluator_node(self, state: AgentState) -> AgentState:
        """Nodo del agente evaluador"""
        
        evaluator_agent = self.agents.get("evaluator")
        
        if evaluator_agent and state.simulation_complete:
            response = await evaluator_agent.process(
                "Evaluar utilidad y calidad de los datos sintéticos finales",
                state.context
            )
            
            state.evaluation_complete = True
            state.context["evaluation_result"] = response
        
        return state
    
    async def process_user_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Procesa input del usuario a través del workflow"""
        
        # Crear estado inicial
        initial_state = AgentState(
            user_input=user_input,
            context=context or {}
        )
        
        # Ejecutar workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        return {
            "state": final_state,
            "response": final_state.context.get("current_response", "Proceso completado"),
            "next_action": final_state.next_action
        }