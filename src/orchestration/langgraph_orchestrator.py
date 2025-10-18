"""
Orquestador LangGraph para coordinar múltiples agentes médicos especializados.
"""

from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, START, END
import pandas as pd
import logging
import time
import datetime

from ..agents.coordinator_agent import CoordinatorAgent
from ..agents.analyzer_agent import ClinicalAnalyzerAgent
from ..agents.generator_agent import SyntheticGeneratorAgent
from ..agents.validator_agent import MedicalValidatorAgent
from ..agents.simulator_agent import PatientSimulatorAgent
from ..agents.evaluator_agent import UtilityEvaluatorAgent
from ..adapters.universal_dataset_detector import UniversalDatasetDetector
from ..utils.streamlit_async_wrapper import run_async_safe
from src.utils.logging_config import get_logger

# Configurar logger centralizado
logger = get_logger(__name__)

class AgentState(TypedDict, total=False):
    """Estado del agente usando TypedDict para compatibilidad con LangGraph"""
    user_input: str
    context: Dict[str, Any]
    coordinator_response: Dict[str, Any]
    universal_analysis: Dict[str, Any]
    next_agent: str
    error: str
    messages: List[Dict[str, Any]]

class MedicalAgentsOrchestrator:
    def __init__(self, agents: Dict[str, Any]):
        start_time = time.time()
        logger.info("Iniciando LangGraph Orchestrator...")
        
        self.agents = agents
        self.universal_detector = UniversalDatasetDetector()
        
        logger.info("Creando workflow...")
        self.workflow = self._create_workflow()
        
        end_time = time.time()
        logger.info("LangGraph Orchestrator inicializado en %.2fs", end_time - start_time)

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("coordinator", self._coordinator_node)
        workflow.add_node("universal_analyzer", self._universal_analyzer_node)
        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("evaluator", self._evaluator_node)
        workflow.add_node("simulator", self._simulator_node)
        
        workflow.add_edge(START, "coordinator")
        workflow.add_conditional_edges(
            "coordinator",
            self._route_from_coordinator,
            {
                "universal_analyzer": "universal_analyzer",
                "analyzer": "analyzer",
                "generator": "generator",
                "validator": "validator",
                "evaluator": "evaluator",
                "simulator": "simulator",
                "__end__": END
            }
        )
        workflow.add_edge("universal_analyzer", "analyzer")
        workflow.add_edge("analyzer", END)
        workflow.add_edge("generator", END)
        workflow.add_edge("validator", END)
        workflow.add_edge("evaluator", END)
        workflow.add_edge("simulator", END)
        return workflow.compile()

    async def _coordinator_node(self, state: AgentState) -> AgentState:
        try:
            start_time = time.time()
            logger.info("Iniciando coordinator_node...")
            
            user_input = state.get("user_input", "")
            context = state.get("context", {})
            if not user_input:
                state["error"] = "No se proporcionó entrada del usuario"
                return state
            
            logger.debug("Input: %s", user_input[:80])
            
            # Crear un nuevo event loop si es necesario para evitar "Event loop is closed"
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            logger.info("Llamando al agente coordinador...")
            response = await self.agents["coordinator"].process(user_input, context)
            state["coordinator_response"] = response
            
            end_time = time.time()
            logger.info("Coordinator completado en %.2fs", end_time - start_time)
            return state
        except Exception as e:
            logger.error("Error en _coordinator_node: %s", e)
            state["error"] = f"Error en coordinador: {str(e)}"
            return state

    async def _universal_analyzer_node(self, state: AgentState) -> AgentState:
        try:
            start_time = time.time()
            logger.info("Iniciando universal_analyzer_node...")
            
            context = state.get("context", {})
            df = context.get("dataframe")
            if df is None or not isinstance(df, pd.DataFrame):
                state["error"] = "Dataset no encontrado para análisis."
                return state
            
            logger.info("Analizando dataset de %sx%s...", df.shape[0], df.shape[1])
            
            # 1. Análisis de clasificación/detección (rápido)
            detection_analysis = self.universal_detector.analyze_dataset(df)
            
            # 2. 🚀 FASE 2: Análisis EDA completo con estadísticas
            from src.analysis.complete_eda import CompleteEDAAnalyzer
            eda_analyzer = CompleteEDAAnalyzer()
            
            # Usar la misma muestra si el detector la usó
            sample_info = detection_analysis.get('sampling_info', {})
            if sample_info.get('sampling_applied', False):
                sample_size = sample_info.get('analyzed_rows', 2000)
                df_for_eda = df.sample(n=min(sample_size, len(df)), random_state=42)
                logger.info("📊 Usando muestra de %d filas para análisis EDA", len(df_for_eda))
            else:
                df_for_eda = df
            
            eda_analysis = eda_analyzer.analyze(df_for_eda, sample_info)
            
            # 3. Combinar ambos análisis
            combined_analysis = {
                **detection_analysis,  # Clasificación, dominio, mapeos
                **eda_analysis,  # Estadísticas, correlaciones, valores nulos
            }
            
            state["universal_analysis"] = combined_analysis
            state["context"]["universal_analysis"] = combined_analysis
            
            end_time = time.time()
            logger.info("Universal analyzer + EDA completado en %.2fs", end_time - start_time)
            return state
        except Exception as e:
            logger.error("Error en _universal_analyzer_node: %s", e)
            state["error"] = f"Error en analizador universal: {str(e)}"
            return state

    async def _analyzer_node(self, state: AgentState) -> AgentState:
        try:
            start_time = time.time()
            logger.info("Iniciando analyzer_node...")
            
            context = state.get("context", {})
            # IMPORTANTE: Asegurarse de que universal_analysis esté en el contexto
            if "universal_analysis" in state and state["universal_analysis"]:
                context["universal_analysis"] = state["universal_analysis"]
                logger.info("✅ Universal analysis agregado al contexto del analyzer")
            else:
                logger.warning("⚠️ No hay universal_analysis en el state")
            
            logger.info("Llamando al agente analyzer...")
            response = await self.agents["analyzer"].analyze_dataset(None, context)
            state["messages"] = state.get("messages", []) + [response]
            
            end_time = time.time()
            logger.info("Analyzer completado en %.2fs", end_time - start_time)
            return state
        except Exception as e:
            logger.error("Error en _analyzer_node: %s", e)
            state["error"] = f"Error en analizador: {str(e)}"
            return state

    async def _generator_node(self, state: AgentState) -> AgentState:
        try:
            start_time = time.time()
            logger.info("🔥 Iniciando generator_node...")
            
            # Obtener parámetros desde coordinator_response O directamente desde context
            params = state["coordinator_response"].get("parameters", {})
            
            # Si no hay parámetros en coordinator_response, buscar en context
            if not params or not any(k in params for k in ['model_type', 'num_samples']):
                logger.info("Parámetros no encontrados en coordinator_response, buscando en context...")
                context_params = {
                    "model_type": state["context"].get("model_type"),
                    "num_samples": state["context"].get("num_samples"),
                }
                # Filtrar valores None
                params = {k: v for k, v in context_params.items() if v is not None}
                logger.info("Parámetros extraídos del context: %s", params)
            
            df = state["context"].get("dataframe")
            if df is None:
                error_msg = "Dataset no encontrado para generación."
                logger.error("❌ %s", error_msg)
                logger.error("Context keys disponibles: %s", list(state["context"].keys()))
                state["error"] = error_msg
                return state
            
            logger.info("📊 Dataset original: %sx%s", df.shape[0], df.shape[1])
        except Exception as e:
            logger.error("❌ Error al iniciar generator_node: %s", e, exc_info=True)
            state["error"] = f"Error al iniciar generación: {str(e)}"
            return state
        
        selected_columns = state["context"].get("selected_columns")
        
        if selected_columns:
            df_for_generation = df[selected_columns].copy()
            logger.info("Usando %s columnas seleccionadas por el usuario para generación", len(selected_columns))
        else:
            universal_analysis = state["context"].get("universal_analysis", {})
            dataset_type = universal_analysis.get("medical_domain", "unknown")
            
            if dataset_type == "covid19":
                covid_columns = [
                    'age', 'sex', 'patient_type', 'pneumonia', 'diabetes', 
                    'copd', 'asthma', 'inmsupr', 'hypertension', 'cardiovascular'
                ]
                available_covid_cols = [col for col in covid_columns if col in df.columns]
                
                if len(available_covid_cols) >= 5:
                    df_for_generation = df[available_covid_cols].copy()
                    logger.info("Usando %s columnas COVID-19 específicas", len(available_covid_cols))
                else:
                    df_for_generation = df.copy()
                    logger.warning("No suficientes columnas COVID-19, usando todas las columnas")
            else:
                if len(df.columns) > 15:
                    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    selected_auto = (numeric_cols[:10] + categorical_cols[:5])[:15]
                    df_for_generation = df[selected_auto].copy()
                    logger.info("Dataset grande: usando %s columnas automáticamente seleccionadas", len(selected_auto))
                else:
                    df_for_generation = df.copy()
                    logger.info("Usando todas las %s columnas del dataset", len(df.columns))
        
        logger.info("Dataset para generación: %sx%s", df_for_generation.shape[0], df_for_generation.shape[1])
        
        # 🔥 IMPORTANTE: Limpiar datos antes de generar
        import numpy as np
        df_cleaned = df_for_generation.copy()
        
        # Convertir columnas numéricas con comas a puntos decimales y limpiar
        cleaned_count = 0
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                try:
                    # Reemplazar comas por puntos y convertir a numérico
                    temp_series = df_cleaned[col].astype(str).str.replace(',', '.', regex=False)
                    # Intentar convertir a numérico
                    numeric_series = pd.to_numeric(temp_series, errors='coerce')
                    # Si se convirtió exitosamente (más del 50% no son NaN), usar la versión numérica
                    if numeric_series.notna().sum() > len(numeric_series) * 0.5:
                        df_cleaned[col] = numeric_series
                        cleaned_count += 1
                        logger.debug("Columna '%s' convertida de object a numeric", col)
                except Exception as e:
                    logger.debug("No se pudo convertir columna '%s': %s", col, e)
        
        # Rellenar NaN en columnas numéricas con la mediana
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_cleaned[col].isna().any():
                median_val = df_cleaned[col].median()
                if not pd.isna(median_val):
                    df_cleaned[col].fillna(median_val, inplace=True)
                else:
                    df_cleaned[col].fillna(0, inplace=True)
        
        logger.info("✅ Datos limpiados para generación (%d columnas convertidas a numéricas)", cleaned_count)
        
        try:
            updated_context = {**state["context"], **params}
            updated_context["dataframe"] = df_cleaned
            updated_context["original_dataframe"] = df
            
            logger.info("🤖 Llamando al agente generator con contexto actualizado...")
            logger.info("   - model_type: %s", updated_context.get("model_type", "auto"))
            logger.info("   - num_samples: %s", updated_context.get("num_samples", 100))
            logger.info("   - dataframe shape: %s", df_cleaned.shape)
            
            response = await self.agents["generator"].process(state["user_input"], updated_context)
            
            logger.info("📨 Generator response keys: %s", list(response.keys()))
            logger.info("📨 Generator response tiene synthetic_data: %s", response.get("synthetic_data") is not None)
            
            if response.get("error"):
                logger.error("❌ Generator devolvió un error: %s", response.get("message"))
                state["error"] = response.get("message")
                state["messages"] = state.get("messages", []) + [response]
                return state
            
            state["messages"] = state.get("messages", []) + [response]
            
            # 🔥 IMPORTANTE: Guardar datos sintéticos en el state y context
            if response.get("synthetic_data") is not None:
                state["context"]["synthetic_data"] = response["synthetic_data"]
                state["synthetic_data"] = response["synthetic_data"]
                logger.info("✅ Datos sintéticos guardados en state: %s registros", len(response["synthetic_data"]))
            else:
                logger.warning("⚠️ Response del generator NO contiene synthetic_data")
            
            end_time = time.time()
            logger.info("🎉 Generator completado exitosamente en %.2fs", end_time - start_time)
            return state
            
        except Exception as e:
            logger.error("❌ Error durante la ejecución del generator: %s", e, exc_info=True)
            state["error"] = f"Error durante la generación: {str(e)}"
            return state

    async def _validator_node(self, state: AgentState) -> AgentState:
        """Nodo del validador médico que prioriza datos sintéticos sobre originales"""
        try:
            start_time = time.time()
            logger.info("Iniciando validator_node")
            
            context = state["context"]
            logger.debug("Context keys: %s", list(context.keys()))
            
            synthetic_data = context.get("synthetic_data")
            
            # 🔥 FIX: Obtener original_data sin usar 'or' con DataFrames
            original_data = context.get("dataframe")
            if original_data is None:
                original_data = context.get("original_dataframe")
            
            # 🔥 FIX: Check if synthetic_data is a DataFrame and not empty
            has_synthetic = (synthetic_data is not None and 
                           isinstance(synthetic_data, pd.DataFrame) and 
                           not synthetic_data.empty)
            
            has_original = (original_data is not None and 
                          isinstance(original_data, pd.DataFrame) and 
                          not original_data.empty)
            
            if has_synthetic:
                validation_context = {
                    **context,
                    "synthetic_data": synthetic_data,
                    "dataframe": original_data,
                    "validation_target": "synthetic"
                }
                logger.info("Validando datos sintéticos (%sx%s)", synthetic_data.shape[0], synthetic_data.shape[1])
            elif has_original:
                validation_context = {
                    **context,
                    "synthetic_data": original_data,
                    "dataframe": original_data,
                    "validation_target": "original"
                }
                logger.info("Validando datos originales (%sx%s)", original_data.shape[0], original_data.shape[1])
            else:
                state["error"] = "No hay datos disponibles para validación."
                return state
            
            logger.info("Llamando al agente validator...")
            response = await self.agents["validator"].process(state["user_input"], validation_context)
            state["messages"] = state.get("messages", []) + [response]
            
            end_time = time.time()
            logger.info("Validator completado en %.2fs", end_time - start_time)
            return state
        except Exception as e:
            logger.error("Error en _validator_node: %s", e)
            state["error"] = f"Error en validación: {str(e)}"
            return state

    async def _evaluator_node(self, state: AgentState) -> AgentState:
        """Nodo del evaluador de utilidad para medir calidad de datos sintéticos"""
        try:
            start_time = time.time()
            logger.info("Iniciando evaluator_node")
            
            context = state["context"]
            logger.debug("Context keys: %s", list(context.keys()))
            
            synthetic_data = context.get("synthetic_data")
            
            # 🔥 FIX: Obtener original_data sin usar 'or' con DataFrames
            original_data = context.get("dataframe")
            if original_data is None:
                original_data = context.get("original_dataframe")
            
            # 🔥 FIX: Check if synthetic_data is a DataFrame and not empty
            has_synthetic = (synthetic_data is not None and 
                           isinstance(synthetic_data, pd.DataFrame) and 
                           not synthetic_data.empty)
            
            has_original = (original_data is not None and 
                          isinstance(original_data, pd.DataFrame) and 
                          not original_data.empty)
            
            if has_synthetic:
                evaluation_context = {
                    **context,
                    "synthetic_data": synthetic_data,
                    "dataframe": original_data,
                    "evaluation_target": "synthetic"
                }
                logger.info("Evaluando datos sintéticos (%sx%s)", synthetic_data.shape[0], synthetic_data.shape[1])
            elif has_original:
                evaluation_context = {
                    **context,
                    "dataframe": original_data,
                    "evaluation_target": "original"
                }
                logger.info("Evaluando datos originales (%sx%s)", original_data.shape[0], original_data.shape[1])
            else:
                state["error"] = "No hay datos disponibles para evaluación."
                return state
            
            logger.info("Llamando al agente evaluator...")
            response = await self.agents["evaluator"].process(state["user_input"], evaluation_context)
            state["messages"] = state.get("messages", []) + [response]
            
            end_time = time.time()
            logger.info("Evaluator completado en %.2fs", end_time - start_time)
            return state
        except Exception as e:
            import traceback
            logger.error("Error en _evaluator_node: %s", e)
            logger.error("Traceback: %s", traceback.format_exc())
            state["error"] = f"Error en evaluación: {str(e)}"
            return state

    async def _simulator_node(self, state: AgentState) -> AgentState:
        """Nodo del simulador de pacientes para evolución clínica"""
        try:
            start_time = time.time()
            logger.info("Iniciando simulator_node")
            
            context = state["context"]
            logger.debug("Context keys: %s", list(context.keys()))
            
            synthetic_data = context.get("synthetic_data")
            
            # 🔥 FIX: Obtener original_data sin usar 'or' con DataFrames
            original_data = context.get("dataframe")
            if original_data is None:
                original_data = context.get("original_dataframe")
            
            # 🔥 FIX: Check if synthetic_data is a DataFrame and not empty
            has_synthetic = (synthetic_data is not None and 
                           isinstance(synthetic_data, pd.DataFrame) and 
                           not synthetic_data.empty)
            
            has_original = (original_data is not None and 
                          isinstance(original_data, pd.DataFrame) and 
                          not original_data.empty)
            
            if has_synthetic:
                simulation_context = {
                    **context,
                    "synthetic_data": synthetic_data,
                    "dataframe": original_data,
                    "simulation_target": "synthetic"
                }
                logger.info("Simulando evolución con datos sintéticos (%sx%s)", synthetic_data.shape[0], synthetic_data.shape[1])
            elif has_original:
                simulation_context = {
                    **context,
                    "dataframe": original_data,
                    "simulation_target": "original"
                }
                logger.info("Simulando evolución con datos originales (%sx%s)", original_data.shape[0], original_data.shape[1])
            else:
                state["error"] = "No hay datos disponibles para simulación."
                return state
            
            logger.info("Llamando al agente simulator...")
            response = await self.agents["simulator"].process(state["user_input"], simulation_context)
            state["messages"] = state.get("messages", []) + [response]
            
            end_time = time.time()
            logger.info("Simulator completado en %.2fs", end_time - start_time)
            return state
        except Exception as e:
            import traceback
            logger.error("Error en _simulator_node: %s", e)
            logger.error("Traceback: %s", traceback.format_exc())
            state["error"] = f"Error en simulación: {str(e)}"
            return state

    def _route_from_coordinator(self, state: AgentState) -> str:
        coordinator_response = state["coordinator_response"]
        intended_agent = coordinator_response.get("agent")
        intention = coordinator_response.get("intention")
        
        logger.info("Routing desde coordinator: agent=%s, intention=%s", intended_agent, intention)
        
        if intention == "conversacion" or intended_agent == "coordinator":
            state["messages"] = [coordinator_response]
            logger.info("Routing a END (conversación)")
            return "__end__"

        if intended_agent == "analyzer":
            if not state["context"].get("universal_analysis"):
                logger.info("Routing a universal_analyzer (análisis inicial)")
                return "universal_analyzer"
            logger.info("Routing a analyzer (análisis detallado)")
            return "analyzer"
        
        if intended_agent == "generator":
            logger.info("Routing a generator")
            return "generator"
        
        if intended_agent == "validator":
            logger.info("Routing a validator")
            return "validator"
        
        if intended_agent == "evaluator":
            logger.info("Routing a evaluator")
            return "evaluator"
        
        if intended_agent == "simulator":
            logger.info("Routing a simulator")
            return "simulator"
        
        logger.warning("Routing no encontrado, yendo a END por defecto")
        state["messages"] = [coordinator_response]
        return "__end__"

    async def process_user_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        workflow_start_time = time.time()
        logger.info("Iniciando process_user_input para: %s", user_input[:50])
        
        initial_state: AgentState = {
            "user_input": user_input, 
            "context": context or {}, 
            "messages": [],
            "coordinator_response": {},
            "universal_analysis": {},
            "next_agent": "",
            "error": ""
        }
        
        try:
            logger.info("Invocando workflow LangGraph...")
            final_state = await self.workflow.ainvoke(initial_state)
            
            workflow_end_time = time.time()
            logger.info("Workflow completado en %.2fs", workflow_end_time - workflow_start_time)
            
            if final_state.get("messages"):
                logger.info("Devolviendo último mensaje de %s mensajes", len(final_state['messages']))
                result = final_state["messages"][-1]
                # 🔥 IMPORTANTE: Incluir datos sintéticos si existen
                if final_state.get("synthetic_data") is not None:
                    result["synthetic_data"] = final_state["synthetic_data"]
                    logger.info("✅ Datos sintéticos incluidos en resultado: %s registros", len(final_state["synthetic_data"]))
                return result
            
            if final_state.get("coordinator_response"):
                logger.info("Devolviendo respuesta del coordinador")
                return final_state["coordinator_response"]
            
            if final_state.get("error"):
                logger.error("Error en workflow: %s", final_state['error'])
                return {
                    "message": f"❌ Error: {final_state['error']}", 
                    "agent": "system",
                    "error": True
                }
            
            logger.warning("No hay mensajes ni respuestas válidas")
            return {
                "message": "Lo siento, no pude procesar tu solicitud. ¿Podrías intentar reformularla?",
                "agent": "coordinator",
                "error": False
            }
        
        except Exception as e:
            workflow_end_time = time.time()
            logger.error("Error en workflow después de %.2fs: %s", workflow_end_time - workflow_start_time, e)
            return {
                "message": f"❌ Error interno: {str(e)}",
                "agent": "system", 
                "error": True
            }

    def process_user_input_sync(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Versión síncrona robusta usando wrapper para evitar problemas de event loop en Streamlit"""
        return run_async_safe(self.process_user_input, user_input, context)
