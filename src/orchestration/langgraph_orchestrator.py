"""
Orquestador LangGraph para coordinar múltiples agentes médicos especializados.
"""

from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, START, END
import pandas as pd
import logging

from ..agents.coordinator_agent import CoordinatorAgent
from ..agents.analyzer_agent import ClinicalAnalyzerAgent
from ..agents.generator_agent import SyntheticGeneratorAgent
from ..agents.validator_agent import MedicalValidatorAgent
from ..agents.simulator_agent import PatientSimulatorAgent
from ..agents.evaluator_agent import UtilityEvaluatorAgent
from ..adapters.universal_dataset_detector import UniversalDatasetDetector

# Configurar logger
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    user_input: str
    context: Dict[str, Any]
    coordinator_response: Dict[str, Any]
    universal_analysis: Dict[str, Any]
    next_agent: str
    error: str
    messages: List[Dict[str, Any]]

class MedicalAgentsOrchestrator:
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.universal_detector = UniversalDatasetDetector()
        self.workflow = self._create_workflow()
        print("🏗️ LangGraph Orchestrator V2 inicializado con componentes universales")

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("universal_analyzer", self._universal_analyzer_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("validator", self._validator_node)
        
        workflow.add_edge(START, "coordinator")
        workflow.add_conditional_edges(
            "coordinator",
            self._route_from_coordinator,
            {
                "universal_analyzer": "universal_analyzer",
                "analyzer": "analyzer",
                "generator": "generator",
                "validator": "validator",
                "__end__": END
            }
        )
        workflow.add_edge("universal_analyzer", "analyzer")
        workflow.add_edge("analyzer", END)
        workflow.add_edge("generator", END)
        workflow.add_edge("validator", END)
        return workflow.compile()

    async def _coordinator_node(self, state: AgentState) -> AgentState:
        response = await self.agents["coordinator"].process(state["user_input"], state["context"])
        state["coordinator_response"] = response
        return state

    async def _universal_analyzer_node(self, state: AgentState) -> AgentState:
        df = state["context"].get("dataframe")
        if df is None or not isinstance(df, pd.DataFrame):
            state["error"] = "Dataset no encontrado para análisis."
            return state
        analysis = self.universal_detector.analyze_dataset(df)
        state["universal_analysis"] = analysis
        state["context"]["universal_analysis"] = analysis # Asegurar que el contexto se actualiza
        return state

    async def _analyzer_node(self, state: AgentState) -> AgentState:
        response = await self.agents["analyzer"].analyze_dataset(None, state["context"])
        state["messages"] = state.get("messages", []) + [response]
        return state

    async def _generator_node(self, state: AgentState) -> AgentState:
        params = state["coordinator_response"].get("parameters", {})
        
        # Obtener DataFrame original
        df = state["context"].get("dataframe")
        if df is None:
            state["error"] = "Dataset no encontrado para generación."
            return state
        
        # Verificar si hay columnas seleccionadas por el usuario
        selected_columns = state["context"].get("selected_columns")
        
        if selected_columns:
            # Usar las columnas seleccionadas por el usuario
            df_for_generation = df[selected_columns].copy()
            logger.info(f"Usando {len(selected_columns)} columnas seleccionadas por el usuario para generación")
        else:
            # Aplicar lógica automática basada en el tipo de dataset
            universal_analysis = state["context"].get("universal_analysis", {})
            dataset_type = universal_analysis.get("medical_domain", "unknown")
            
            if dataset_type == "covid19":
                # Para COVID-19, usar solo 10 columnas específicas
                covid_columns = [
                    'age', 'sex', 'patient_type', 'pneumonia', 'diabetes', 
                    'copd', 'asthma', 'inmsupr', 'hypertension', 'cardiovascular'
                ]
                available_covid_cols = [col for col in covid_columns if col in df.columns]
                
                if len(available_covid_cols) >= 5:  # Mínimo 5 columnas
                    df_for_generation = df[available_covid_cols].copy()
                    logger.info(f"Usando {len(available_covid_cols)} columnas COVID-19 específicas")
                else:
                    # Fallback: usar todas las columnas
                    df_for_generation = df.copy()
                    logger.warning("No suficientes columnas COVID-19, usando todas las columnas")
            else:
                # Para otros datasets, usar todas las columnas (máximo 15 para rendimiento)
                if len(df.columns) > 15:
                    # Priorizar columnas numéricas y categóricas importantes
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    
                    selected_auto = (numeric_cols[:10] + categorical_cols[:5])[:15]
                    df_for_generation = df[selected_auto].copy()
                    logger.info(f"Dataset grande: usando {len(selected_auto)} columnas automáticamente seleccionadas")
                else:
                    df_for_generation = df.copy()
                    logger.info(f"Usando todas las {len(df.columns)} columnas del dataset")
        
        # Actualizar el contexto con el DataFrame procesado
        updated_context = {**state["context"], **params}
        updated_context["dataframe"] = df_for_generation
        updated_context["original_dataframe"] = df  # Mantener referencia al original
        
        # Llamar al agente generador
        response = await self.agents["generator"].process(state["user_input"], updated_context)
        state["messages"] = state.get("messages", []) + [response]
        return state

    async def _validator_node(self, state: AgentState) -> AgentState:
        """Nodo del validador médico que prioriza datos sintéticos sobre originales"""
        try:
            print("🔍 Iniciando _validator_node")
            context = state["context"]
            print(f"🔍 Context keys: {list(context.keys())}")
            
            # Priorizar datos sintéticos si están disponibles
            synthetic_data = context.get("synthetic_data")
            original_data = context.get("dataframe")
            if original_data is None:
                original_data = context.get("original_dataframe")
            
            print(f"🔍 synthetic_data type: {type(synthetic_data)}")
            print(f"🔍 original_data type: {type(original_data)}")
            
            if synthetic_data is not None and not synthetic_data.empty:
                # Usar datos sintéticos
                validation_context = {
                    **context,
                    "synthetic_data": synthetic_data,
                    "dataframe": original_data,  # Para comparación/referencia
                    "validation_target": "synthetic"
                }
                print("✅ Validando datos sintéticos")
            elif original_data is not None and not original_data.empty:
                # Usar datos originales como fallback
                validation_context = {
                    **context,
                    "synthetic_data": original_data,  # Usar como datos a validar
                    "dataframe": original_data,
                    "validation_target": "original"
                }
                print("✅ Validando datos originales (no hay datos sintéticos disponibles)")
            else:
                # Error: no hay datos para validar
                state["error"] = "No hay datos disponibles para validación."
                return state
            
            print("🔍 Llamando al agente validator...")
            # Procesar validación
            response = await self.agents["validator"].process(state["user_input"], validation_context)
            state["messages"] = state.get("messages", []) + [response]
            print("✅ Validación completada")
            return state
        except Exception as e:
            print(f"❌ Error en _validator_node: {e}")
            import traceback
            traceback.print_exc()
            state["error"] = f"Error en validación: {str(e)}"
            return state

    def _route_from_coordinator(self, state: AgentState) -> str:
        try:
            coordinator_response = state["coordinator_response"]
            intended_agent = coordinator_response.get("agent")
            intention = coordinator_response.get("intention")
            
            print(f"🔀 Routing: intention={intention}, agent={intended_agent}")
            
            # Si es una conversación, terminar directamente con la respuesta del coordinador
            if intention == "conversacion" or intended_agent == "coordinator":
                state["messages"] = [coordinator_response]
                return "__end__"

            # Si es un comando específico, dirigir al agente correspondiente
            if intended_agent == "analyzer":
                if not state["context"].get("universal_analysis"):
                    return "universal_analyzer"
                return "analyzer"
            
            if intended_agent == "generator":
                return "generator"
                
            if intended_agent == "validator":
                return "validator"
            
            # Para cualquier otro agente o caso, terminar con la respuesta del coordinador
            state["messages"] = [coordinator_response]
            return "__end__"
        except Exception as e:
            print(f"❌ Error en routing: {e}")
            state["error"] = f"Error en routing: {str(e)}"
            return "__end__"

    async def process_user_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        initial_state = {
            "user_input": user_input, 
            "context": context or {}, 
            "messages": [],
            "coordinator_response": {},
            "universal_analysis": {},
            "next_agent": "",
            "error": ""
        }
        
        try:
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Si hay mensajes, devolver el último
            if final_state.get("messages"):
                return final_state["messages"][-1]
            
            # Si hay una respuesta del coordinador pero no mensajes, usar esa respuesta
            if final_state.get("coordinator_response"):
                return final_state["coordinator_response"]
            
            # Si hay un error específico, devolverlo
            if final_state.get("error"):
                return {
                    "message": f"❌ Error: {final_state['error']}", 
                    "agent": "system",
                    "error": True
                }
            
            # Fallback: respuesta por defecto
            return {
                "message": "Lo siento, no pude procesar tu solicitud. ¿Podrías intentar reformularla?",
                "agent": "coordinator",
                "error": False
            }
        
        except Exception as e:
            logger.error(f"Error en process_user_input: {e}")
            return {
                "message": f"❌ Error interno: {str(e)}",
                "agent": "system", 
                "error": True
            }
