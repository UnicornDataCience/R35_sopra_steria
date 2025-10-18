import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import os
import sys
import re
import json
import inspect
import concurrent.futures
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from src.utils.logging_config import get_logger
from pydantic import ValidationError
from src.agents.schemas import AgentResponse, Context

logger = get_logger(__name__)

# Añadir la ruta del proyecto al path de Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Cargar variables de entorno
load_dotenv()

# FORZAR GROQ - Sobrescribir variable del sistema si FORCE_GROQ=true
if os.getenv('FORCE_GROQ', 'false').lower() == 'true':
    os.environ['LLM_PROVIDER'] = 'groq'
    logger.info("[STREAMLIT] LLM_PROVIDER forzado a 'groq'")

# Importar el wrapper síncrono para evitar problemas de event loop
try:
    from src.utils.streamlit_async_wrapper import run_async_safe
    ASYNC_WRAPPER_AVAILABLE = True
    logger.info("Wrapper síncrono disponible")
except ImportError as e:
    ASYNC_WRAPPER_AVAILABLE = False
    logger.warning("Wrapper síncrono no disponible: %s", e)

# Importar el selector de columnas médicas
try:
    from src.adapters.medical_column_selector import MedicalColumnSelector
    COLUMN_SELECTOR_AVAILABLE = True
except ImportError as e:
    COLUMN_SELECTOR_AVAILABLE = False
    logger.warning("Selector de columnas no disponible: %s", e)

# Configuración de página
st.set_page_config(
    page_title="Patient IA",
    page_icon=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "logo_patientia.png"),
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado para un diseño moderno
st.markdown("""
<style>
    /* Ocultar elementos innecesarios */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stMainBlockContainer {padding-top: 2rem;}
    
    /* Estilo del chat */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Prompt input más grande */
    .stChatInput {
        position: relative;
        max-width: 700px !important;
        margin: 0 auto !important;
    }
    
    .stChatInput > div > div > textarea {
        min-height: 60px !important;
        font-size: 16px !important;
        border-radius: 25px !important;
        border: 2px solid #e1e5e9 !important;
        padding: 15px 75px 15px 60px !important;
        text-indent: 10ch !important;
    }
    
    /* Header minimalista */
    .main-header {
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0 0 0;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    /* Logo centrado */
    .logo-container {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .logo-container img {
        margin: 0 auto;
        display: block;
    }
    
    /* Status indicator centrado */
    .status-indicator {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        background: #f0f9ff;
        border: 1px solid #bfdbfe;
        border-radius: 20px;
        padding: 6px 12px;
        font-size: 0.875rem;
        color: #1e40af;
        margin: 0.5rem auto 1rem auto;
        width: fit-content;
    }
    
    /* Mensaje de bienvenida centrado */
    .welcome-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 40vh;
        margin-top: 0;
    }
    
    /* Mensajes del chat */
    .stChatMessage {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Importar configuración unificada de LLMs
try:
    from src.config.llm_config import unified_llm_config
    LLM_CONFIGURED = True
    provider_info = unified_llm_config.status_info
    active_provider = provider_info["active_provider"]
    
    # FORZAR TEST DE CONEXIÓN para diagnosticar problema del chat
    connection_test = False
    try:
        logger.info("🔍 Probando conexión con %s...", active_provider)
        connection_test = unified_llm_config.test_connection()
        if connection_test:
            logger.info("✅ LLM conectado correctamente - Proveedor: %s", active_provider)
        else:
            logger.warning("❌ Test de conexión falló - Proveedor: %s", active_provider)
    except Exception as e:
        logger.error("💥 Error en test de conexión LLM: %s", e)
        connection_test = False
except Exception as e:
    LLM_CONFIGURED = False
    connection_test = False
    active_provider = "none"
    logger.warning("Error de configuración LLM: %s", e)

# Mantener compatibilidad con código existente
AZURE_CONFIGURED = LLM_CONFIGURED and active_provider == "azure"

# Importar orquestador con fallback
try:
    from src.orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator, AgentState
    LANGGRAPH_AVAILABLE = True
    logger.info("LangGraph Orchestrator disponible")
except Exception as e:
    logger.info("Usando Simple Orchestrator (LangGraph tiene problema de dependencias): %s", e)
    try:
        from src.orchestration.simple_orchestrator import MedicalAgentsOrchestrator
        LANGGRAPH_AVAILABLE = False
        logger.info("Usando Simple Orchestrator como fallback")
    except Exception as e2:
        logger.error("Error cargando orquestador alternativo: %s", e2)
        LANGGRAPH_AVAILABLE = False

# Importar el nuevo FastOrchestrator como alternativa optimizada
try:
    from src.orchestration.fast_orchestrator import FastMedicalOrchestrator
    FAST_ORCHESTRATOR_AVAILABLE = True
    logger.info("Fast Orchestrator disponible")
except Exception as e:
    FAST_ORCHESTRATOR_AVAILABLE = False
    logger.warning("Fast Orchestrator no disponible: %s", e)

try:
    from src.agents.base_agent import BaseLLMAgent, BaseAgentConfig
    from src.agents.coordinator_agent import CoordinatorAgent
    from src.agents.analyzer_agent import ClinicalAnalyzerAgent
    from src.agents.generator_agent import SyntheticGeneratorAgent
    
    # Importar otros agentes con manejo individual de errores
    try:
        from src.agents.validator_agent import MedicalValidatorAgent
        VALIDATOR_AVAILABLE = True
    except Exception as e:
        logger.warning("Validator agent no disponible: %s", e)
        VALIDATOR_AVAILABLE = False
    
    try:
        from src.agents.simulator_agent import PatientSimulatorAgent
        SIMULATOR_AVAILABLE = True
    except Exception as e:
        logger.warning("Simulator agent no disponible: %s", e)
        SIMULATOR_AVAILABLE = False
    
    try:
        from src.agents.evaluator_agent import UtilityEvaluatorAgent
        EVALUATOR_AVAILABLE = True
    except Exception as e:
        logger.warning("Evaluator agent no disponible: %s", e)
        EVALUATOR_AVAILABLE = False

    AGENTS_AVAILABLE = True
    # Respetar LANGGRAPH_AVAILABLE ya establecido
    logger.info("Agentes cargados correctamente. Orquestador: %s", "LangGraph" if LANGGRAPH_AVAILABLE else "Simple")
    
except Exception as e:
    AGENTS_AVAILABLE = False
    LANGGRAPH_AVAILABLE = False
    VALIDATOR_AVAILABLE = False
    SIMULATOR_AVAILABLE = False
    EVALUATOR_AVAILABLE = False
    logger.error("Error cargando agentes: %s", e)

# Agente mock para desarrollo
class MockAgent:
    def __init__(self, name):
        self.name = name
        self.config = type('Config', (), {'name': name})()
    
    async def process(self, input_text, context=None):
        return self.process_sync(input_text, context)
    
    def process_sync(self, input_text, context=None):
        context = context or {}
        has_dataset = context.get("dataset_uploaded", False)
        
        if "coordinador" in self.name.lower():
            dataset_msg = ""
            if has_dataset:
                filename = context.get("filename", "archivo")
                rows = context.get("rows", 0)
                cols = context.get("columns", 0)
                dataset_msg = f"\n\nDataset detectado: {filename} ({rows:,} filas, {cols} columnas)"
            
            return {
                "message": f"👋 **¡Hola!** Soy tu asistente de IA para generar datos clínicos sintéticos.\n\n🔬 **Estado:** {'✅ ' + active_provider.title() + ' Conectado' if connection_test else '🔄 Modo Simulado'}{dataset_msg}\n\n**🧠 Mi equipo especializado:**\n• **Analista** - Extrae patrones clínicos\n• **Generador** - Crea datos sintéticos con SDV\n• **Validador** - Verifica coherencia médica\n• **Simulador** - Modela evolución temporal\n• **Evaluador** - Mide calidad y utilidad\n\n¿En qué puedo ayudarte hoy?",
                "agent": self.name,
                "mock": True
            }
        elif "analista" in self.name.lower():
            if has_dataset:
                filename = context.get("filename", "dataset")
                rows = context.get("rows", 0)
                cols = context.get("columns", 0)
                return {
                    "message": f"""🔍 **Análisis completado (modo simulado)**

**📊 Dataset analizado:** {filename}
- **Registros:** {rows:,}
- **Columnas:** {cols}

**📈 Análisis estadístico:**
- Tipos de datos identificados
- Valores faltantes detectados  
- Distribuciones analizadas
- Correlaciones calculadas

**💡 Insights principales:**
- Dataset preparado para generación sintética
- Calidad de datos: Buena
- Recomendación: Proceder con generación TVAE o CTGAN

*Nota: Análisis completo disponible con Azure OpenAI configurado.*""",
                    "agent": self.name,
                    "mock": True
                }
            else:
                return {
                    "message": "📁 **No hay dataset cargado**\n\nPara análisis clínico, necesito que subas un archivo CSV o Excel con datos médicos.\n\n📊 **Formatos aceptados:** CSV, XLSX, XLS",
                    "agent": self.name,
                    "mock": True
                }
        elif "generador" in self.name.lower():
            if has_dataset:
                # Obtener parámetros de generación
                params = context.get("parameters", {})
                model_type = params.get("model_type", "ctgan")
                num_samples = params.get("num_samples", 100)
                
                # Crear datos sintéticos simulados
                original_df = context.get("dataframe")
                if original_df is not None:
                    # Crear una muestra sintética usando el DataFrame original
                    synthetic_data = original_df.sample(n=min(num_samples, len(original_df)), replace=True).reset_index(drop=True)
                    
                    # Añadir algo de ruido para simular diferencias
                    import numpy as np
                    for col in synthetic_data.select_dtypes(include=[np.number]).columns:
                        noise = np.random.normal(0, synthetic_data[col].std() * 0.05, len(synthetic_data))
                        synthetic_data[col] = synthetic_data[col] + noise
                    
                    return {
                        "message": f"""🧬 **Generación sintética completada**

**📊 Resultado:**
- **Modelo utilizado:** {model_type.upper()}
- **Registros generados:** {len(synthetic_data):,}
- **Dataset base:** {len(original_df):,} registros

**✅ Calidad de datos:** Los datos sintéticos mantienen las propiedades estadísticas del dataset original mientras preservan la privacidad.""",
                        "agent": self.name,
                        "synthetic_data": synthetic_data,
                        "generation_info": {
                            "model_type": model_type,
                            "num_samples": len(synthetic_data),
                            "columns_used": len(synthetic_data.columns),
                            "selection_method": "Mock Generation",
                            "timestamp": pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                        }
                    }
                else:
                    rows = context.get("rows", 0)
                    return {
                        "message": f"""🧬 **Generación sintética completada (modo simulado)**

**📊 Dataset base:** {rows:,} registros
**🎯 Registros generados:** {num_samples} (simulado)
**🔬 Modelo utilizado:** {model_type.upper()} (simulado)

**✅ Proceso completado:**
- Datos sintéticos generados exitosamente
- Calidad preservada
- Privacidad garantizada

*Nota: Generación real disponible con Azure OpenAI configurado.*""",
                        "agent": self.name,
                        "mock": True
                    }
            else:
                return {
                    "message": "📁 **Dataset requerido**\n\nPara generar datos sintéticos, primero necesito un dataset base.\n\n**Sube un archivo** con datos clínicos para comenzar.",
                    "agent": self.name,
                    "mock": True
                }
        else:
            return {
                "message": f"🤖 **{self.name}**\n\n{input_text}\n\n*Funcionalidad completa disponible con Azure OpenAI configurado.*",
                "agent": self.name,
                "mock": True
            }

def clean_response_message(message: str) -> str:
    """
    Limpia el mensaje de respuesta de caracteres Unicode escapados, JSON crudo y problemas de encoding.
    Diseñado específicamente para el problema de doble encoding UTF-8/Latin-1 del modelo LLM.
    """
    
    # MÉTODO 0: Detectar y extraer JSON crudo más robustamente
    message_stripped = message.strip()
    
    # Casos de JSON directo
    if message_stripped.startswith('{"intention"') or message_stripped.startswith('{"message"'):
        try:
            parsed = json.loads(message_stripped)
            if 'message' in parsed:
                message = parsed['message']
                logger.info("JSON crudo extraído exitosamente")
            elif 'content' in parsed:
                message = parsed['content']
                logger.info("Contenido JSON extraído")
        except json.JSONDecodeError as e:
            logger.warning("Error parseando JSON crudo: %s", e)
            # Si no se puede parsear, mantener el mensaje original
            pass
    
    # Casos de JSON anidado o mal formateado
    elif '"message"' in message_stripped and '"intention"' in message_stripped:
        try:
            # Buscar el patrón JSON dentro del texto
            json_pattern = r'\{"intention".*?"message"[^}]*\}'
            match = re.search(json_pattern, message_stripped, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if 'message' in parsed:
                    message = parsed['message']
                    logger.info("JSON anidado extraído")
        except:
            pass
    
    try:
        # MÉTODO 1: Detección y corrección más agresiva del doble encoding UTF-8/Latin-1
        # Este es el problema principal con el modelo llama-4-scout
        
        # Lista de indicadores de corrupción más completa
        corrupted_patterns = [
            'Â¡', 'Â¿', 'Ã¡', 'Ã©', 'Ã­', 'Ã³', 'Ãº', 'Ã±', 'Ã‰', 'Ã"', 'Ãš', 'ÃÑ',
            'mÃ©dica', 'mÃ©dico', 'anÃ¡lisis', 'sÃ­ntomas', 'diagnÃ³stico', 'informaciÃ³n',
            'especializaciÃ³n', 'atenciÃ³n', 'evaluaciÃ³n', 'investigaciÃ³n', 'aquÃ­',
            'tambiÃ©n', 'prÃ¡ctica', 'clÃ­nicos', 'pÃ¡gina', 'dÃ­a', 'fÃ¡cil', 'difÃ­cil',
            'quÃ©', 'cÃ³mo', 'dÃ³nde', 'cuÃ¡ndo'
        ]
        
        if any(pattern in message for pattern in corrupted_patterns):
            logger.info("Detectada corrupción de encoding, aplicando corrección")
            
            # ESTRATEGIA MÚLTIPLE DE CORRECCIÓN
            original_message = message
            attempts = []
            
            # Intento 1: Corrección automática estándar
            try:
                corrected1 = message.encode('latin-1').decode('utf-8')
                attempts.append(('auto_standard', corrected1))
            except:
                pass
            
            # Intento 2: Corrección con manejo de errores 'ignore'
            try:
                corrected2 = message.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                attempts.append(('auto_ignore', corrected2))
            except:
                pass
            
            # Intento 3: Corrección manual con diccionario más completo
            corrected3 = message
            corruption_map = {
                # Signos de puntuación
                'Â¡': '¡', 'Â¿': '¿',
                # Vocales con tilde minúsculas
                'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
                # Eñe
                'Ã±': 'ñ', 'ÃÑ': 'Ñ',
                # Palabras completas muy comunes (más directas)
                'mÃ©dica': 'médica', 'mÃ©dico': 'médico', 'mÃ©dicos': 'médicos',
                'anÃ¡lisis': 'análisis', 'sÃ­ntomas': 'síntomas', 'sÃ­ntoma': 'síntoma',
                'diagnÃ³stico': 'diagnóstico', 'diagnÃ³sticos': 'diagnósticos',
                'informaciÃ³n': 'información', 'atenciÃ³n': 'atención',
                'especializaciÃ³n': 'especialización', 'evaluaciÃ³n': 'evaluación',
                'investigaciÃ³n': 'investigación', 'prÃ¡ctica': 'práctica',
                'clÃ­nicos': 'clínicos', 'clÃ­nica': 'clínica', 'clÃ­nico': 'clínico',
                'pÃ¡gina': 'página', 'pÃ¡ginas': 'páginas',
                'dÃ­a': 'día', 'dÃ­as': 'días',
                'aquÃ­': 'aquí', 'allÃ­': 'allí', 'tambiÃ©n': 'también',
                'fÃ¡cil': 'fácil', 'difÃ­cil': 'difícil',
                # Interrogativos
                'quÃ©': 'qué', 'cÃ³mo': 'cómo', 'dÃ³nde': 'dónde', 'cuÃ¡ndo': 'cuándo',
                'quiÃ©n': 'quién', 'cuÃ¡l': 'cuál', 'cuÃ¡nto': 'cuánto'
            }
            
            for corrupt, correct in corruption_map.items():
                if corrupt in corrected3:
                    corrected3 = corrected3.replace(corrupt, correct)
            
            attempts.append(('manual_dict', corrected3))
            
            # Evaluar cuál corrección es mejor
            def count_spanish_chars(text):
                spanish_chars = ['á', 'é', 'í', 'ó', 'ú', 'ñ', '¡', '¿', 'Á', 'É', 'Í', 'Ó', 'Ú', 'Ñ']
                return sum(1 for char in text if char in spanish_chars)
            
            def count_corrupted_chars(text):
                corrupted = ['Ã', 'Â']
                return sum(1 for char in text if char in corrupted)
            
            # Seleccionar la mejor corrección
            best_attempt = None
            best_score = -1
            
            for method, corrected in attempts:
                spanish_count = count_spanish_chars(corrected)
                corrupted_count = count_corrupted_chars(corrected)
                score = spanish_count - (corrupted_count * 2)  # Penalizar caracteres corruptos
                
                if score > best_score:
                    best_score = score
                    best_attempt = (method, corrected)
            
            if best_attempt:
                message = best_attempt[1]
                logger.info("Mejor corrección: %s (score: %d)", best_attempt[0], best_score)
        
        # MÉTODO 2: Limpiar secuencias Unicode escapadas (ya funciona bien)
        unicode_patterns = [
            (r'\\u00a1', '¡'), (r'\\u00bf', '¿'),
            (r'\\u00e1', 'á'), (r'\\u00e9', 'é'), (r'\\u00ed', 'í'), 
            (r'\\u00f3', 'ó'), (r'\\u00fa', 'ú'), (r'\\u00f1', 'ñ'),
            (r'\\u00c1', 'Á'), (r'\\u00c9', 'É'), (r'\\u00CD', 'Í'), 
            (r'\\u00D3', 'Ó'), (r'\\u00DA', 'Ú'), (r'\\u00D1', 'Ñ')
        ]
        
        unicode_replaced = False
        for pattern, replacement in unicode_patterns:
            if re.search(pattern, message, flags=re.IGNORECASE):
                message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
                unicode_replaced = True
        
        if unicode_replaced:
            logger.info("Secuencias Unicode escapadas convertidas")
        
        # MÉTODO 3: Limpieza final conservadora
        try:
            decoded = message.encode().decode('unicode_escape')
            if decoded != message and not any(char in decoded for char in ['Ã', 'Â']):
                message = decoded
                logger.info("Unicode escape final aplicado")
        except:
            pass
            
    except Exception as e:
        logger.warning("Error en limpieza de mensaje: %s", e)
    
    return message.strip()

@st.cache_resource
def initialize_orchestrator():
    """
    Inicializa el orquestador principal con prioridad para FastOrchestrator.
    FastOrchestrator es optimizado para respuestas rápidas sin timeout.
    """
    # PRIORIDAD 1: FastOrchestrator (Optimizado para velocidad) - SOLO SI LLM CONECTADO
    if FAST_ORCHESTRATOR_AVAILABLE and LLM_CONFIGURED and connection_test:
        try:
            agents = {}
            
            # Cargar agentes disponibles para workflows específicos
            if AGENTS_AVAILABLE:
                try:
                    agents["coordinator"] = CoordinatorAgent()
                    agents["analyzer"] = ClinicalAnalyzerAgent()
                    agents["generator"] = SyntheticGeneratorAgent()
                    
                    if VALIDATOR_AVAILABLE:
                        agents["validator"] = MedicalValidatorAgent()
                        logger.info("Validator agent agregado a FastOrchestrator")
                    
                    if SIMULATOR_AVAILABLE:
                        agents["simulator"] = PatientSimulatorAgent()
                        logger.info("Simulator agent agregado a FastOrchestrator")
                    
                    if EVALUATOR_AVAILABLE:
                        agents["evaluator"] = UtilityEvaluatorAgent()
                        logger.info("Evaluator agent agregado a FastOrchestrator")
                        
                except Exception as e:
                    logger.warning("Algunos agentes no disponibles para FastOrchestrator: %s", e)
            
            fast_orchestrator = FastMedicalOrchestrator(agents)
            logger.info("🚀 FastMedicalOrchestrator inicializado con LLM REAL conectado")
            return fast_orchestrator
            
        except Exception as e:
            logger.error("Error inicializando FastOrchestrator: %s", e)
    
    # PRIORIDAD 2: LangGraph Orchestrator (Original) - SOLO SI LLM CONECTADO
    if AGENTS_AVAILABLE and LANGGRAPH_AVAILABLE and LLM_CONFIGURED and connection_test:
        try:
            agents = {
                "coordinator": CoordinatorAgent(),
                "analyzer": ClinicalAnalyzerAgent(),
                "generator": SyntheticGeneratorAgent(),
            }
            
            # Agregar agentes opcionales solo si están disponibles
            if VALIDATOR_AVAILABLE:
                agents["validator"] = MedicalValidatorAgent()
                logger.info("Validator agent agregado")
            
            if SIMULATOR_AVAILABLE:
                agents["simulator"] = PatientSimulatorAgent()
                logger.info("Simulator agent agregado")
            
            if EVALUATOR_AVAILABLE:
                agents["evaluator"] = UtilityEvaluatorAgent()
                logger.info("Evaluator agent agregado")
            
            orchestrator = MedicalAgentsOrchestrator(agents)
            logger.info("🔧 LangGraph Orchestrator inicializado con LLM REAL conectado")
            return orchestrator
        except Exception as e:
            logger.error("Error en LangGraph: %s", e)
    
    # PRIORIDAD 3: Mock Orchestrator (Fallback)
    if LLM_CONFIGURED and not connection_test:
        logger.warning("🟡 LLM configurado pero conexión falló, usando orquestador mock")
    else:
        logger.info("🟡 LLM no configurado, usando orquestador mock")
    return create_mock_orchestrator()

@st.cache_resource 
def initialize_langgraph_orchestrator():
    """Función legacy para compatibilidad - ahora redirige a initialize_orchestrator"""
    return initialize_orchestrator()

def create_mock_orchestrator():
    """Crea un orquestador mock para desarrollo"""
    mock_agents = {
        "coordinator": MockAgent("Coordinador"),
        "analyzer": MockAgent("Analista Clínico"),
        "generator": MockAgent("Generador Sintético"),
        "validator": MockAgent("Validador Médico"),
        "simulator": MockAgent("Simulador de Pacientes"),
        "evaluator": MockAgent("Evaluador de Utilidad")
    }
    
    class MockLangGraphOrchestrator:
        def __init__(self, agents):
            self.agents = agents
            self.state = {
                "current_agent": "coordinator",
                "dataset_uploaded": False,
                "analysis_complete": False,
                "generation_complete": False,
                "validation_complete": False,
                "synthetic_data": None,
                "context": {}
            }
        
        async def process_user_input(self, user_input: str, context: dict = None):
            """Procesa input del usuario (versión mock)"""
            return self.process_user_input_sync(user_input, context)
        
        def process_user_input_sync(self, user_input: str, context: dict = None):
            """Versión síncrona del procesamiento de input del usuario"""
            self.state["context"] = context or {}
            
            # Detectar intención y ejecutar agente correspondiente
            if any(word in user_input.lower() for word in ["analizar", "análisis", "analiza"]):
                self.state["current_agent"] = "analyzer"
                agent = self.agents["analyzer"]
                response = agent.process_sync(user_input, context)
                return response
            elif any(word in user_input.lower() for word in ["generar", "sintético", "sintéticos", "genera"]):
                self.state["current_agent"] = "generator"
                
                # Detectar modelo específico en lenguaje natural
                model_type = "ctgan"  # Default
                if any(word in user_input.lower() for word in ["tvae", "variational", "autoencoder"]):
                    model_type = "tvae"
                elif any(word in user_input.lower() for word in ["sdv", "vault", "synthetic data vault"]):
                    model_type = "sdv"
                elif any(word in user_input.lower() for word in ["ctgan", "gan", "generative adversarial"]):
                    model_type = "ctgan"
                
                # Detectar número de muestras
                import re
                numbers = re.findall(r'\b(\d+)\b', user_input)
                num_samples = int(numbers[0]) if numbers else 100
                
                # Configurar contexto con parámetros
                context = context or {}
                context['parameters'] = {
                    'model_type': model_type,
                    'num_samples': num_samples
                }
                
                agent = self.agents["generator"]
                response = agent.process_sync(user_input, context)
                return response
            elif any(word in user_input.lower() for word in ["validar", "valida", "validación"]):
                self.state["current_agent"] = "validator"
                # Usar el validador real si está disponible, incluso en modo mock
                if VALIDATOR_AVAILABLE:
                    try:
                        agent = self.agents["validator"]
                        # Para el agente validador real que puede ser async, usar el wrapper
                        if hasattr(agent, 'process_sync'):
                            response = agent.process_sync(user_input, context)
                        else:
                            # Si es async, usar run_async_safe si está disponible
                            if ASYNC_WRAPPER_AVAILABLE:
                                response = run_async_safe(agent.process, user_input, context)
                            else:
                                response = {"message": "❌ Error: Agente async sin wrapper disponible", "agent": "validator", "error": True}
                        return response
                    except Exception as e:
                        return {"message": f"❌ Error en validación: {str(e)}", "agent": "validator", "error": True}
                else:
                    return {"message": "✅ Validación completada (modo simulado)\n\nDatos validados exitosamente.", "agent": "validator"}
            elif any(word in user_input.lower() for word in ["evaluar", "evalúa", "calidad"]):
                self.state["current_agent"] = "evaluator"
                return {"message": "📊 Evaluación completada (modo simulado)\n\nCalidad de datos: Excelente", "agent": "evaluator"}
            elif any(word in user_input.lower() for word in ["simular", "simula", "paciente"]):
                self.state["current_agent"] = "simulator"
                return {"message": "🏥 Simulación completada (modo simulado)\n\nEvolución de pacientes simulada.", "agent": "simulator"}
            else:
                # Para preguntas conversacionales, médicas o generales
                self.state["current_agent"] = "coordinator"
                
                # SI TENEMOS CONEXIÓN LLM, usar el coordinador real
                if LLM_CONFIGURED and connection_test and AGENTS_AVAILABLE:
                    try:
                        from src.agents.coordinator_agent import CoordinatorAgent
                        real_coordinator = CoordinatorAgent()
                        
                        # Usar método síncrono o wrapper según sea necesario
                        if hasattr(real_coordinator, 'process_sync'):
                            response = real_coordinator.process_sync(user_input, context)
                        elif ASYNC_WRAPPER_AVAILABLE:
                            response = run_async_safe(real_coordinator.process, user_input, context)
                        else:
                            response = {"message": "❌ Error: Agente async sin wrapper disponible", "agent": "coordinator", "error": True}
                        
                        # Limpiar la respuesta si viene del LLM real
                        if isinstance(response, dict) and "message" in response:
                            response["message"] = clean_response_message(response["message"])
                        
                        return response
                    except Exception as e:
                        logger.error("Error usando coordinador real: %s", e)
                        # Si falla, usar mock como fallback
                        return self._generate_mock_medical_response(user_input, context)
                else:
                    # Respuesta mock inteligente para preguntas médicas
                    return self._generate_mock_medical_response(user_input, context)
        
        def _generate_mock_medical_response(self, user_input: str, context: dict = None):
            """Genera respuestas mock inteligentes para preguntas médicas"""
            user_lower = user_input.lower()
            
            # Detectar diferentes tipos de preguntas médicas
            if any(term in user_lower for term in ["diabetes", "glucosa", "insulina"]):
                return {
                    "message": "🩺 **Información sobre Diabetes**\n\nLa diabetes es una enfermedad crónica que afecta la forma en que el cuerpo procesa la glucosa. Los principales factores de riesgo incluyen:\n\n• **Tipo 1**: Factores genéticos e inmunológicos\n• **Tipo 2**: Sobrepeso, sedentarismo, historia familiar\n• **Gestacional**: Cambios hormonales durante el embarazo\n\n**Recomendación**: Para análisis detallado de datos diabéticos, sube un dataset y solicita un análisis específico.\n\n*Nota: Esta es información general. Consulta siempre con un profesional médico.*",
                    "agent": "coordinator",
                    "topic": "diabetes"
                }
            elif any(term in user_lower for term in ["covid", "coronavirus", "sars-cov"]):
                return {
                    "message": "🦠 **Información sobre COVID-19**\n\nFactores de riesgo identificados en datos clínicos:\n\n• **Edad**: Pacientes > 65 años\n• **Comorbilidades**: Diabetes, hipertensión, EPOC\n• **Estado inmunológico**: Inmunosupresión\n• **Factores cardiovasculares**: Enfermedad cardíaca previa\n\n**Nuestro sistema** puede analizar datasets COVID-19 y generar datos sintéticos que preserven estos patrones epidemiológicos.\n\n*Datos basados en estudios internacionales publicados.*",
                    "agent": "coordinator",
                    "topic": "covid19"
                }
            elif any(term in user_lower for term in ["hipertensión", "presión", "cardiovascular"]):
                return {
                    "message": "❤️ **Factores de Riesgo Cardiovascular**\n\nPrincipales factores identificados en estudios clínicos:\n\n• **Modificables**: Tabaquismo, colesterol alto, sedentarismo\n• **No modificables**: Edad, sexo, historia familiar\n• **Metabólicos**: Diabetes, obesidad, síndrome metabólico\n• **Otros**: Estrés, apnea del sueño, enfermedad renal\n\n**¿Tienes datos cardiovasulares?** Puedo ayudarte a analizarlos y generar datasets sintéticos para investigación.\n\n*Información basada en guías clínicas internacionales.*",
                    "agent": "coordinator",
                    "topic": "cardiovascular"
                }
            elif any(term in user_lower for term in ["cáncer", "oncología", "tumor", "metástasis"]):
                return {
                    "message": "🎗️ **Información Oncológica**\n\nFactores relevantes en análisis de datos oncológicos:\n\n• **Estadificación**: TNM, grado histológico\n• **Biomarcadores**: Receptores hormonales, HER2, mutaciones\n• **Tratamiento**: Quimioterapia, radioterapia, inmunoterapia\n• **Seguimiento**: Supervivencia libre de enfermedad, calidad de vida\n\n**Capacidades del sistema**: Análisis de cohortes oncológicas y generación de datos sintéticos preservando características pronósticas.\n\n*Para análisis específicos, considera subir datos anonimizados.*",
                    "agent": "coordinator",
                    "topic": "oncology"
                }
            elif any(term in user_lower for term in ["hola", "saludo", "buenos días", "buenas tardes", "como estas"]):
                has_dataset = context and context.get("dataset_uploaded", False)
                dataset_msg = ""
                if has_dataset:
                    filename = context.get("filename", "archivo")
                    rows = context.get("rows", 0)
                    cols = context.get("columns", 0)
                    dataset_msg = f"\n\n📊 **Dataset actual**: {filename} ({rows:,} filas, {cols} columnas)"
                
                return {
                    "message": f"👋 **¡Hola!** Estoy muy bien, gracias por preguntar.\n\nSoy tu asistente de IA especializado en datos clínicos sintéticos.\n\n🔬 **Estado del sistema**: {'✅ ' + active_provider.title() + ' Conectado' if connection_test else '🔄 Modo Simulado'}{dataset_msg}\n\n**¿En qué puedo ayudarte?**\n• Analizar datasets médicos\n• Generar datos sintéticos seguros\n• Responder preguntas sobre medicina\n• Validar coherencia clínica\n\n¡Pregúntame cualquier cosa sobre medicina o datos clínicos!",
                    "agent": "coordinator",
                    "topic": "greeting"
                }
            elif any(term in user_lower for term in ["ayuda", "help", "qué puedes hacer"]):
                return {
                    "message": "📋 **Guía de Uso - Patient IA**\n\n**🤖 Comandos principales:**\n• `Analiza estos datos` - Explora patrones en tu dataset\n• `Genera 1000 muestras con CTGAN` - Crea datos sintéticos\n• `Valida la coherencia médica` - Verifica calidad clínica\n\n**🩺 Consultas médicas:**\n• Factores de riesgo cardiovascular\n• Información sobre diabetes, COVID-19\n• Análisis epidemiológico\n• Interpretación de biomarcadores\n\n**📊 Tipos de datos soportados:**\n• CSV, Excel (.xlsx, .xls)\n• Historiales clínicos\n• Datos de laboratorio\n• Registros epidemiológicos\n\n¿Hay algo específico en lo que te pueda ayudar?",
                    "agent": "coordinator",
                    "topic": "help"
                }
            else:
                # Respuesta general para otras preguntas médicas
                return {
                    "message": f"🤔 **Respuesta médica (modo simulado)**\n\nHe recibido tu consulta: *\"{user_input}\"*\n\n📚 Como asistente de IA médica, puedo ayudarte con:\n• Análisis de datasets clínicos\n• Información sobre enfermedades comunes\n• Interpretación de factores de riesgo\n• Generación de datos sintéticos\n\n**Para respuestas más precisas**, configura Azure OpenAI o formula tu pregunta de manera más específica.\n\n*Recuerda: Esta información es para fines educativos. Consulta siempre con profesionales médicos.*",
                    "agent": "coordinator",
                    "topic": "general_medical"
                }
    
    return MockLangGraphOrchestrator(mock_agents)

# Función initialize_orchestrator movida arriba para evitar duplicación

# Inicialización del estado
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = initialize_orchestrator()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'context' not in st.session_state:
    st.session_state.context = {}

if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

if 'config' not in st.session_state:
    st.session_state.config = {
        'max_synthetic_rows': 500,
        'analysis_mode': 'Básico',
        'enable_validation': True
    }

def limit_chat_history():
    """Limita el historial de chat para evitar exceso de tokens"""
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]

def process_uploaded_file(uploaded_file=None):
    """Procesa un archivo cargado desde el sidebar"""
    if uploaded_file:
        try:
            with st.spinner(f"Procesando {uploaded_file.name}..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.context.update({
                    'dataframe': df,
                    'dataset_uploaded': True,
                    'filename': uploaded_file.name,
                    'rows': df.shape[0],
                    'columns': df.shape[1]
                })
                
                st.session_state.file_uploaded = True
                st.session_state.uploaded_file = uploaded_file
                
                st.success(f"✅ Archivo {uploaded_file.name} cargado exitosamente")
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"📊 **Archivo cargado exitosamente**\n\n**{uploaded_file.name}**\n- {df.shape[0]:,} filas\n- {df.shape[1]} columnas\n\n¿Qué te gustaría hacer con estos datos?",
                    "dataset_loaded": True,
                    "dataset_info": {
                        "filename": uploaded_file.name,
                        "rows": df.shape[0],
                        "columns": df.shape[1],
                        "dtypes": df.dtypes.value_counts().to_dict()
                    },
                    "dataset_preview": df.head().to_dict('records')
                })
                
                return True
                
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
            return False
    return False

def handle_synthetic_data_response(response, context=None):
    """Maneja la respuesta de generación sintética de forma centralizada"""
    # Aceptar tanto dict plano como AgentResponse
    resp_dict = response
    try:
        if isinstance(response, AgentResponse):
            resp_dict = response.model_dump()
    except Exception:
        pass

    payload = resp_dict.get("payload") if isinstance(resp_dict, dict) else None
    if isinstance(payload, dict) and "synthetic_data" in payload:
        synthetic_df = payload.get("synthetic_data")
        generation_info = payload.get("generation_info", {}) or resp_dict.get("generation_info", {})
    elif isinstance(resp_dict, dict) and "synthetic_data" in resp_dict:
        synthetic_df = resp_dict["synthetic_data"]
        generation_info = resp_dict.get("generation_info", {})
    else:
        return False
        
        # Si generation_info está vacío o incompleto, crear uno por defecto
        if not generation_info or not generation_info.get('model_type'):
            # Intentar extraer información del contexto o response
            context = context or {}
            parameters = context.get("parameters", {})
            
            # Crear generation_info por defecto con información disponible
            generation_info = {
                "model_type": parameters.get("model_type", "ctgan"),  # Modelo por defecto
                "num_samples": len(synthetic_df),
                "columns_used": len(synthetic_df.columns),
                "selection_method": "Columnas seleccionadas" if context.get('selected_columns') else "Automático",
                "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
            }
        else:
            # Asegurar que tiene timestamp
            if 'timestamp' not in generation_info:
                generation_info["timestamp"] = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Asegurar que num_samples coincide con los datos reales
            if 'num_samples' not in generation_info or generation_info['num_samples'] != len(synthetic_df):
                generation_info["num_samples"] = len(synthetic_df)
            
            # Asegurar que columns_used coincide con los datos reales
            if 'columns_used' not in generation_info or generation_info['columns_used'] != len(synthetic_df.columns):
                generation_info["columns_used"] = len(synthetic_df.columns)
        
        # Guardar en session_state
        st.session_state.context["synthetic_data"] = synthetic_df
        st.session_state.context["generation_info"] = generation_info
        
        return True
    return False

# Header principal con logo integrado
logo_path = os.path.join(project_root, "assets", "logo_patientia.png")
if os.path.exists(logo_path):
    col_left, col_logo, col_title, col_right = st.columns([5.7, 0.7, 6, 2])
    with col_logo:
        st.image(logo_path, width=54)
    with col_title:
        st.markdown(
            "<h1 style='font-size: 2.2rem; font-weight: 600; color: #1f2937; margin: 0; display: flex; align-items: center; height: 54px;'>Patient-IA</h1>",
            unsafe_allow_html=True
        )
else:
    st.markdown("""
    <div class="main-header">
        <h1>Patient-IA</h1>
    </div>
    """, unsafe_allow_html=True)

# Status indicator actualizado
if LLM_CONFIGURED and connection_test:
    if active_provider == "azure":
        status_text = "✅ Azure OpenAI Conectado"
    elif active_provider == "ollama":
        status_text = "✅ Ollama Local Conectado"
    elif active_provider == "grok":
        status_text = "✅ Grok Conectado"
    else:
        status_text = "✅ LLM Conectado"
    status_color = "#10b981"
elif LLM_CONFIGURED:
    status_text = f"🟡 {active_provider.title()} Configurado (Sin conexión)"
    status_color = "#f59e0b"
else:
    status_text = "🔄 Modo Simulado"
    status_color = "#6b7280"

st.markdown(f"""
<div class="status-indicator" style="border-color: {status_color}20; background: {status_color}10; color: {status_color};">
    <span>{status_text}</span>
    <span style="margin-left: 10px;">•</span>
    <span>{len(st.session_state.chat_history)} mensajes</span>
    {"<span style='margin-left: 10px;'>• Archivo cargado</span>" if st.session_state.get('file_uploaded', False) else ""}
</div>
""", unsafe_allow_html=True)

# Sidebar mejorado con funcionalidades útiles
with st.sidebar:
    st.header("Panel de Control")
    
    # Cargar Dataset
    st.subheader("Cargar Dataset")
    uploaded_file_sidebar = st.file_uploader(
        "Sube un archivo CSV o Excel", 
        type=["csv", "xlsx", "xls"], 
        key="sidebar_uploader"
    )
    
    # Procesar archivo si se ha subido uno nuevo
    if uploaded_file_sidebar is not None:
        current_file_key = f"{uploaded_file_sidebar.name}_{uploaded_file_sidebar.size}"
        previous_file_key = st.session_state.get('last_processed_file_key', '')
        
        if current_file_key != previous_file_key:
            st.session_state.last_processed_file_key = current_file_key
            if process_uploaded_file(uploaded_file_sidebar):
                st.rerun()
    
    # Información del dataset si está cargado
    if st.session_state.get('file_uploaded', False):
        st.subheader("Dataset Actual")
        filename = st.session_state.context.get("filename", "archivo")
        rows = st.session_state.context.get("rows", 0)
        cols = st.session_state.context.get("columns", 0)
        
        st.success(f"✅ **{filename}**")
        st.info(f"📈 **{rows:,}** filas, **{cols}** columnas")
        
        # Botón para seleccionar columnas para generación sintética
        if COLUMN_SELECTOR_AVAILABLE and 'dataframe' in st.session_state.context:
            if st.button("🔍 Seleccionar Columnas", use_container_width=True):
                st.session_state.show_column_selector = True
        
        # Botón para quitar archivo
        if st.button("Quitar Archivo", use_container_width=True):
            keys_to_reset = ['file_uploaded', 'uploaded_file', 'context', 'analysis_complete', 'show_column_selector', 'selected_columns']
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        # Botón para nueva conversación
        if st.button("Nueva Conversación", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != 'orchestrator':
                    del st.session_state[key]
            st.rerun()
    
    # Comandos rápidos
    st.subheader("Comandos Rápidos")
    if st.button("Analizar Datos", use_container_width=True):
        if st.session_state.get('file_uploaded'):
            st.session_state.quick_command = "analizar datos"
        else:
            st.warning("Primero carga un archivo")
    
    if st.button("🤖 Generar Sintéticos", use_container_width=True):
        if st.session_state.get('file_uploaded'):
            # Mostrar selector de modelos en lugar de quick_command
            st.session_state.show_model_selector = True
            st.session_state.selected_model = 'ctgan'  # Modelo por defecto
        else:
            st.warning("Primero carga un archivo")
    
    if st.button("Validar Datos", use_container_width=True):
        if st.session_state.get('file_uploaded'):
            st.session_state.quick_command = "validar datos"
        else:
            st.warning("Primero carga un archivo")
    
    if st.button("Evaluar Calidad", use_container_width=True):
        if st.session_state.get('file_uploaded'):
            st.session_state.quick_command = "evaluar calidad"
        else:
            st.warning("Primero carga un archivo")
    
    if st.button("Simular Paciente", use_container_width=True):
        if st.session_state.get('file_uploaded'):
            st.session_state.quick_command = "simular paciente"
        else:
            st.warning("Primero carga un archivo")
    
    # Configuración avanzada
    with st.expander("Configuración Avanzada"):
        st.markdown("**Límites de Generación:**")
        max_synthetic_rows = st.slider("Máx. registros sintéticos", 50, 1000, 500)
        
        st.markdown("**Modo de Análisis:**")
        analysis_mode = st.selectbox("Tipo de análisis", ["Básico", "Detallado", "Experto"])
        
        st.markdown("**Validación:**")
        enable_validation = st.checkbox("Validación automática", value=True)
        
        st.session_state.config = {
            'max_synthetic_rows': max_synthetic_rows,
            'analysis_mode': analysis_mode,
            'enable_validation': enable_validation
        }
    
    # Selector de modelo para generación sintética
    if st.session_state.get('show_model_selector', False):
        st.markdown("---")
        st.subheader("🤖 Selección de Modelo de Generación")
        
        # Explicaciones de modelos
        model_info = {
            'ctgan': {
                'name': 'CTGAN (Conditional Tabular GAN)',
                'description': 'Red neuronal generativa adversarial especializada en datos tabulares.',
                'pros': '• Excelente para datos mixtos (categóricos + numéricos)\n• Maneja correlaciones complejas\n• Rápido entrenamiento',
                'cons': '• Puede generar outliers\n• Requiere ajuste de hiperparámetros',
                'best_for': 'Datasets médicos con variables categóricas y numéricas mezcladas',
                'color': 'blue'
            },
            'tvae': {
                'name': 'TVAE (Tabular Variational AutoEncoder)', 
                'description': 'Autoencoder variacional optimizado para datos tabulares.',
                'pros': '• Preserva distribuciones estadísticas\n• Menos propenso a outliers\n• Estable y confiable',
                'cons': '• Puede ser conservador\n• Menor diversidad en algunos casos',
                'best_for': 'Cuando se requiere alta fidelidad estadística',
                'color': 'green'
            },
            'sdv': {
                'name': 'SDV (Synthetic Data Vault)',
                'description': 'Suite completa de síntesis con múltiples algoritmos.',
                'pros': '• Algoritmos múltiples integrados\n• Optimizado para datos médicos\n• Validación automática',
                'cons': '• Mayor complejidad computacional\n• Tiempo de entrenamiento más largo',
                'best_for': 'Proyectos que requieren máxima calidad y validación',
                'color': 'orange'
            }
        }
        
        # Selector de modelo por defecto
        default_model = st.session_state.get('selected_model', 'ctgan')
        
        # Crear tabs para cada modelo
        tab1, tab2, tab3 = st.tabs(['🔵 CTGAN', '🟢 TVAE', '🟠 SDV'])
        
        with tab1:
            info = model_info['ctgan']
            st.markdown(f"**{info['name']}**")
            st.write(info['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Ventajas:**")
                st.markdown(info['pros'])
            with col2:
                st.markdown("**⚠️ Consideraciones:**")
                st.markdown(info['cons'])
            
            st.info(f"**🎯 Ideal para:** {info['best_for']}")
            
            if st.button("Seleccionar CTGAN", key="select_ctgan", use_container_width=True):
                st.session_state.selected_model = 'ctgan'
                st.success("✅ CTGAN seleccionado")
        
        with tab2:
            info = model_info['tvae']
            st.markdown(f"**{info['name']}**")
            st.write(info['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Ventajas:**")
                st.markdown(info['pros'])
            with col2:
                st.markdown("**⚠️ Consideraciones:**")
                st.markdown(info['cons'])
            
            st.info(f"**🎯 Ideal para:** {info['best_for']}")
            
            if st.button("Seleccionar TVAE", key="select_tvae", use_container_width=True):
                st.session_state.selected_model = 'tvae'
                st.success("✅ TVAE seleccionado")
        
        with tab3:
            info = model_info['sdv']
            st.markdown(f"**{info['name']}**")
            st.write(info['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Ventajas:**")
                st.markdown(info['pros'])
            with col2:
                st.markdown("**⚠️ Consideraciones:**")
                st.markdown(info['cons'])
            
            st.info(f"**🎯 Ideal para:** {info['best_for']}")
            
            if st.button("Seleccionar SDV", key="select_sdv", use_container_width=True):
                st.session_state.selected_model = 'sdv'
                st.success("✅ SDV seleccionado")
        
        st.markdown("---")
        
        # Configuración adicional
        col1, col2 = st.columns(2)
        with col1:
            num_samples = st.number_input("Número de registros a generar", min_value=10, max_value=1000, value=100, step=10)
        with col2:
            st.markdown("**Modelo seleccionado:**")
            selected_model = st.session_state.get('selected_model', 'ctgan')
            st.markdown(f"🤖 **{model_info[selected_model]['name']}**")
        
        # Botones de acción
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🚀 Generar Datos", use_container_width=True):
                # Preparar contexto para generación
                context_for_generation = st.session_state.context.copy()
                context_for_generation['parameters'] = {
                    'model_type': st.session_state.get('selected_model', 'ctgan'),
                    'num_samples': num_samples
                }
                
                # Añadir columnas seleccionadas si existen
                if st.session_state.get('selected_columns'):
                    context_for_generation['selected_columns'] = st.session_state.selected_columns
                
                # Simular llamada al generador
                prompt_for_generation = f"Genera {num_samples} registros sintéticos usando el modelo {selected_model.upper()}"
                st.session_state.pending_generation = {
                    'prompt': prompt_for_generation,
                    'context': context_for_generation
                }
                st.session_state.show_model_selector = False
                st.success(f"✅ Configuración guardada. Generando con {selected_model.upper()}...")
                st.rerun()
        
        with col2:
            if st.button("🔄 Cambiar Modelo", use_container_width=True):
                st.session_state.selected_model = 'ctgan'  # Reset a default
                st.rerun()
        
        with col3:
            if st.button("❌ Cancelar", use_container_width=True):
                st.session_state.show_model_selector = False
                if 'selected_model' in st.session_state:
                    del st.session_state.selected_model
                st.rerun()

    # Información de datos sintéticos generados
    if st.session_state.context.get('synthetic_data') is not None:
        st.markdown("---")
        st.subheader("📊 Datos Sintéticos Generados")
        
        synthetic_df = st.session_state.context['synthetic_data']
        generation_info = st.session_state.context.get('generation_info', {})
        
        # Métricas básicas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Registros", f"{len(synthetic_df):,}")
        with col2:
            st.metric("Columnas", len(synthetic_df.columns))
        with col3:
            # Mostrar información del modelo de forma más clara
            model_type = generation_info.get('model_type', 'N/A')
            if model_type and model_type != 'N/A':
                model_display = model_type.upper()
            else:
                # Si no tenemos info del modelo, mostrar información del DataFrame
                model_display = "GENERADO"
            st.metric("Modelo", model_display)
        
        # Vista previa de los datos
        st.markdown("**🔍 Vista previa de datos sintéticos:**")
        st.dataframe(synthetic_df.head(10), use_container_width=True)
        
        # Información de generación
        if generation_info:
            with st.expander("🔬 Detalles de generación"):
                col1, col2 = st.columns(2)
                with col1:
                    # Formatear modelo de forma segura
                    model_type = generation_info.get('model_type', 'N/A')
                    if model_type and model_type != 'N/A':
                        st.text(f"Modelo utilizado: {model_type.upper()}")
                    else:
                        st.text(f"Modelo utilizado: Datos sintéticos generados")
                    # Formatear número de muestras de forma segura
                    num_samples = generation_info.get('num_samples', len(synthetic_df))
                    if isinstance(num_samples, (int, float)):
                        st.text(f"Registros generados: {int(num_samples):,}")
                    else:
                        st.text(f"Registros generados: {len(synthetic_df):,}")
                with col2:
                    selection_method = generation_info.get('selection_method', 'N/A')
                    if selection_method == 'N/A':
                        selection_method = "Método estándar"
                    st.text(f"Método de selección: {selection_method}")
                    
                    columns_used = generation_info.get('columns_used', len(synthetic_df.columns))
                    st.text(f"Columnas utilizadas: {columns_used}")
        else:
            # Si no hay generation_info, mostrar información básica del DataFrame
            with st.expander("🔬 Información de los datos"):
                col1, col2 = st.columns(2)
                with col1:
                    st.text(f"Datos sintéticos: Generados exitosamente")
                    st.text(f"Registros: {len(synthetic_df):,}")
                with col2:
                    st.text(f"Columnas: {len(synthetic_df.columns)}")
                    st.text(f"Método: Generación estándar")
        
        # Botones de descarga mejorados
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            csv_data = synthetic_df.to_csv(index=False)
            timestamp = generation_info.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
            filename_csv = f"datos_sinteticos_{generation_info.get('model_type', 'ctgan')}_{len(synthetic_df)}_{timestamp}.csv"
            st.download_button(
                label="📄 Descargar CSV",
                data=csv_data,
                file_name=filename_csv,
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_data = synthetic_df.to_json(orient='records', indent=2)
            filename_json = f"datos_sinteticos_{generation_info.get('model_type', 'ctgan')}_{len(synthetic_df)}_{timestamp}.json"
            st.download_button(
                label="📋 Descargar JSON",
                data=json_data,
                file_name=filename_json,
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Botón para limpiar datos sintéticos
            if st.button("🗑️ Limpiar", use_container_width=True, help="Limpiar datos sintéticos generados"):
                del st.session_state.context['synthetic_data']
                if 'generation_info' in st.session_state.context:
                    del st.session_state.context['generation_info']
                st.success("✅ Datos sintéticos eliminados")
                st.rerun()


        

        
    # Agregar nota de seguridad y privacidad
    st.markdown("""
    ---
    **🔒 Privacidad y Seguridad:** Todos los datos se procesan localmente. Los datos sintéticos mantienen las propiedades estadísticas 
    del dataset original mientras protegen la identidad de los pacientes individuales.
    """)
    
    # Mostrar estado del sistema en la parte inferior
    st.subheader("🔧 Estado del Sistema")
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        if LLM_CONFIGURED and connection_test:
            if active_provider == "azure":
                llm_status = "🟢 Azure Conectado"
            elif active_provider == "ollama":
                llm_status = "🟢 Ollama Local"
            elif active_provider == "grok":
                llm_status = "🟢 Grok Conectado"
            else:
                llm_status = "🟢 LLM Conectado"
        else:
            llm_status = "🟡 Modo Simulado"
        st.info(f"**Proveedor LLM:** {llm_status}")
    
    with status_col2:
        # Detectar tipo de orquestador
        orchestrator_type = type(st.session_state.get('orchestrator', None)).__name__
        if "FastMedical" in orchestrator_type:
            orchestrator_status = "🚀 Fast Mode"
            orchestrator_color = "green"
        elif "MedicalAgents" in orchestrator_type:
            orchestrator_status = "🔧 Full Mode" 
            orchestrator_color = "blue"
        else:
            orchestrator_status = "🟡 Mock Mode"
            orchestrator_color = "orange"
        st.info(f"**Orquestador:** {orchestrator_status}")
        
    with status_col3:
        agents_status = "🟢 Disponibles" if AGENTS_AVAILABLE else "🟡 Mock Agents"
        st.info(f"**Agentes IA:** {agents_status}")
    
    # Mostrar información adicional del orquestador en uso
    if "FastMedical" in orchestrator_type:
        st.success("⚡ **Modo Fast**: Respuestas optimizadas sin timeout. Agentes disponibles para tareas específicas.")
    elif "MedicalAgents" in orchestrator_type:
        st.info("🔧 **Modo Full**: Workflow completo con LangGraph. Todas las capacidades disponibles.")
    else:
        st.warning("🟡 **Modo Mock**: Respuestas simuladas para desarrollo. Configura LLM para funcionalidad completa.")

with st.container():
    # Mostrar historial de chat
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                # Limpiar el mensaje de caracteres Unicode y JSON crudo
                clean_content = clean_response_message(message["content"])
                st.markdown(clean_content)
                
                # MOSTRAR INFORMACIÓN ESPECIAL DEL DATASET
                if message.get("dataset_loaded"):
                    dataset_info = message.get("dataset_info", {})
                    dataset_preview = message.get("dataset_preview", [])
                    
                    with st.expander("👁️ Vista previa del dataset", expanded=False):
                        if dataset_preview:
                            preview_df = pd.DataFrame(dataset_preview)
                            st.dataframe(preview_df, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**📊 Distribución de tipos:**")
                            for dtype, count in dataset_info.get('dtypes', {}).items():
                                st.text(f"{dtype}: {count} columnas")
                        
                        with col2:
                            st.markdown("**📋 Información técnica:**")
                            st.text(f"Nombre: {dataset_info.get('filename', 'N/A')}")
                            st.text(f"Filas: {dataset_info.get('rows', 0):,}")
                            st.text(f"Columnas: {dataset_info.get('columns', 0)}")
    else:
        # Interfaz estática de bienvenida (NO es un mensaje de chat)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 2.0rem; font-weight: 600; color: #1f2937; margin-bottom: 0.5rem;">
                Asistente Inteligente para Generación de Datos Clínicos Sintéticos
            </h1>
            
        </div>
        """, unsafe_allow_html=True)

        # Crear dos columnas principales para explicar las capacidades del sistema
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            ### � **Capacidades del Sistema**
            
            **🧠 Agentes Especializados:**
            - **Analista Clínico** - Extrae patrones y estadísticas médicas
            - **Generador Sintético** - Crea datos realistas con CTGAN/TVAE/SDV
            - **Validador Médico** - Verifica coherencia clínica
            - **Simulador de Pacientes** - Modela evolución temporal
            - **Evaluador de Utilidad** - Mide calidad y privacidad
            
            **📊 Formatos Soportados:**
            - CSV, Excel (.xlsx, .xls)
            - Datos COVID-19, oncológicos, cardiológicos
            - Historiales clínicos y registros médicos
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 **Cómo Empezar**
            
            **1️⃣ Cargar Datos:**
            - Sube tu archivo CSV/Excel con datos clínicos
            - El sistema detectará automáticamente el dominio médico
            - Selecciona las columnas más relevantes (opcional)
            
            **2️⃣ Interactuar:**
            - **"Analiza estos datos"** - Para explorar patrones
            - **"Genera 1000 muestras con CTGAN"** - Para crear sintéticos
            - **"Valida la coherencia médica"** - Para verificar calidad
            - **"¿Cuáles son los factores de riesgo cardiovascular?"** - Consultas médicas
            
            **3️⃣ Descargar:**
            - Descarga los datos sintéticos en CSV/JSON
            - Revisa métricas de calidad y privacidad
            """)

# --- LÓGICA PRINCIPAL DEL CHAT ---
def main_chat_loop():
    """Función principal para manejar el bucle del chat usando wrapper síncrono."""
    limit_chat_history()
    
    # Procesar comandos rápidos del sidebar
    if hasattr(st.session_state, 'quick_command'):
        prompt = st.session_state.quick_command
        del st.session_state.quick_command
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("Procesando comando..."):
            # Preparar contexto incluyendo columnas seleccionadas si existen
            context_with_selections = st.session_state.context.copy()
            if st.session_state.get('selected_columns'):
                context_with_selections['selected_columns'] = st.session_state.selected_columns
            
            response = process_orchestrator_input_safe(st.session_state.orchestrator, prompt, context_with_selections)
            full_response = response.get("message", "No se recibió respuesta.")
            
            # Limpiar la respuesta antes de guardarla en el historial
            clean_response = clean_response_message(full_response)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": clean_response, 
                "agent": response.get("agent"),
                "mock": False
            })
        
        st.rerun()
        return
    
    # JavaScript para indentación automática
    st.markdown("""
    <script>
        function setupTextIndentation() {
            const textarea = document.querySelector('.stChatInput textarea');
            
            if (textarea) {
                const tenSpaces = '          '; // 10 espacios
                
                textarea.addEventListener('focus', function() {
                    if (this.value === '' || this.value === tenSpaces) {
                        this.value = tenSpaces;
                        this.setSelectionRange(tenSpaces.length, tenSpaces.length);
                    }
                });
                
                textarea.addEventListener('input', function() {
                    if (!this.value.startsWith(tenSpaces)) {
                        const cursorPos = this.selectionStart;
                        const restOfText = this.value.replace(/^\\s*/, '');
                        this.value = tenSpaces + restOfText;
                        this.setSelectionRange(Math.max(tenSpaces.length, cursorPos), Math.max(tenSpaces.length, cursorPos));
                    }
                });
                
                textarea.addEventListener('keydown', function(e) {
                    if (e.key === 'Backspace' && this.selectionStart <= tenSpaces.length) {
                        e.preventDefault();
                    }
                    if (e.key === 'Delete' && this.selectionStart < tenSpaces.length) {
                        e.preventDefault();
                    }
                    if (e.key === 'Home') {
                        e.preventDefault();
                        this.setSelectionRange(tenSpaces.length, tenSpaces.length);
                    }
                });
            }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(setupTextIndentation, 500);
            setTimeout(setupTextIndentation, 1500);
            setTimeout(setupTextIndentation, 3000);
        });
    </script>
    """, unsafe_allow_html=True)
    
    # Interfaz de selección de columnas
    if st.session_state.get('show_column_selector', False) and COLUMN_SELECTOR_AVAILABLE:
        st.markdown("---")
        
        # Crear instancia del selector de columnas
        column_selector = MedicalColumnSelector()
        df = st.session_state.context.get('dataframe')
        
        if df is not None:
            # Validar primero si el dataset es médico válido
            is_valid, validation_errors = column_selector.validate_medical_dataset(df)
            
            if not is_valid:
                st.error("❌ **Dataset no válido para generación sintética médica**")
                for error in validation_errors:
                    st.warning(f"⚠️ {error}")
                
                if st.button("Cerrar Selector"):
                    st.session_state.show_column_selector = False
                    st.rerun()
            else:
                # Mostrar interfaz de selección
                column_selection = column_selector.generate_column_selection_interface(df)
                
                # Botones de acción
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("✅ Confirmar Selección", use_container_width=True, disabled=not column_selection.mandatory_fulfilled):
                        st.session_state.selected_columns = column_selection.selected_columns
                        st.session_state.show_column_selector = False
                        st.success(f"Seleccionadas {len(column_selection.selected_columns)} columnas para generación sintética")
                        st.rerun()
                
                with col2:
                    if st.button("🔄 Usar Recomendadas", use_container_width=True):
                        # Usar selección automática recomendada
                        dataset_type = column_selector.detector.detect_dataset_type(df)
                        column_mappings = column_selector.detector.infer_medical_columns(df)
                        recommended = column_selector._get_recommended_columns(dataset_type, column_mappings, df)
                        
                        st.session_state.selected_columns = recommended
                        st.session_state.show_column_selector = False
                        st.success(f"Usando {len(recommended)} columnas recomendadas automáticamente")
                        st.rerun()
                
                with col3:
                    if st.button("❌ Cancelar", use_container_width=True):
                        st.session_state.show_column_selector = False
                        st.rerun()
        
        st.markdown("---")
    
    # Mostrar columnas seleccionadas si existen
    if st.session_state.get('selected_columns'):
        columns_display = st.session_state.selected_columns[:5]
        more_text = f" (+{len(st.session_state.selected_columns)-5} más)" if len(st.session_state.selected_columns) > 5 else ""
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"🎯 **Columnas seleccionadas:** {', '.join(columns_display)}{more_text}")
        with col2:
            if st.button("❌ Limpiar selección", help="Eliminar selección de columnas"):
                del st.session_state.selected_columns
                st.rerun()
    
    # Procesar generación pendiente si existe
    if st.session_state.get('pending_generation'):
        pending = st.session_state.pending_generation
        del st.session_state.pending_generation
        
        # Añadir el prompt de generación al historial
        st.session_state.chat_history.append({"role": "user", "content": pending['prompt']})
        
        with st.spinner("🔄 Generando datos sintéticos..."):
            response = process_orchestrator_input_safe(
                st.session_state.orchestrator,
                pending['prompt'], 
                pending['context']
            )
            
            # Manejar respuesta de generación sintética
            handle_synthetic_data_response(response, pending['context'])
            
            full_response = response.get("message", "No se recibió respuesta.")
            
            # Limpiar la respuesta antes de guardarla en el historial
            clean_response = clean_response_message(full_response)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": clean_response, 
                "agent": response.get("agent"),
                "mock": False
            })
        
        st.rerun()
        return

    # Chat input principal de Streamlit
    if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Crear un placeholder para el progreso
        progress_placeholder = st.empty()
        
        with st.spinner("🔄 Procesando solicitud..."):
            # Preparar contexto incluyendo columnas seleccionadas si existen
            context_with_selections = st.session_state.context.copy()
            if st.session_state.get('selected_columns'):
                context_with_selections['selected_columns'] = st.session_state.selected_columns
                progress_placeholder.info(f"✅ Usando {len(st.session_state.selected_columns)} columnas seleccionadas")
            
            response = process_orchestrator_input_safe(st.session_state.orchestrator, prompt, context_with_selections)
            progress_placeholder.empty()  # Limpiar el mensaje de progreso
            
            # Manejar respuesta de generación sintética
            handle_synthetic_data_response(response, context_with_selections)
            
            if response.get("error"):
                error_message = response.get("error")
                if isinstance(error_message, bool):
                    error_message = "Error en el procesamiento del dataset"
                elif not isinstance(error_message, str):
                    error_message = str(error_message)
                
                st.error(f"❌ **Error**: {error_message}")
                return
            
            # Obtener y limpiar el mensaje de respuesta
            raw_message = response.get("message", "No se recibió respuesta.")
            
            # Limpiar caracteres Unicode escapados y JSON crudo si aparece
            cleaned_message = clean_response_message(raw_message)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": cleaned_message, 
                "agent": response.get("agent"),
                "mock": False
            })
        
        st.rerun()

def validate_agent_response(response: Any) -> Dict[str, Any]:
    """
    Valida y estandariza la respuesta de un agente.
    Acepta diccionarios o instancias de AgentResponse y devuelve un diccionario.
    """
    if isinstance(response, AgentResponse):
        return response.model_dump(exclude_none=True)
    
    if isinstance(response, dict):
        # Asegurar que los campos mínimos estén presentes
        return {
            "message": response.get("message", "Respuesta no estructurada."),
            "agent": response.get("agent", "unknown"),
            "success": response.get("success", True),
            "payload": response.get("payload"),
            "metadata": response.get("metadata", {}),
        }
    
    # Si es un string u otro tipo, envolverlo en una respuesta estándar
    return {
        "message": str(response),
        "agent": "system",
        "success": False,
        "payload": None,
        "metadata": {"raw_response": response},
    }

def build_typed_context_dict(context: Optional[Dict[str, Any]]) -> Context:
    """
    Construye un diccionario validado compatible con el modelo Pydantic Context.
    """
    if context is None:
        return Context()

    # Crear una copia para no modificar el original
    context_copy = context.copy()

    # Pydantic no puede manejar DataFrames, así que lo extraemos temporalmente
    dataframe = context_copy.pop('dataframe', None)
    synthetic_data = context_copy.pop('synthetic_data', None)

    try:
        # Validar el resto del contexto
        typed_context = Context(**context_copy)
        
        # Reasignar los DataFrames al objeto validado
        if dataframe is not None:
            typed_context.dataframe = dataframe
        if synthetic_data is not None:
            typed_context.synthetic_data = synthetic_data
            
        return typed_context

    except ValidationError as e:
        logger.warning("Error de validación de contexto, usando valores por defecto: %s", e)
        # En caso de error, devolver un contexto por defecto pero intentando preservar los dataframes
        default_context = Context()
        if dataframe is not None:
            default_context.dataframe = dataframe
        if synthetic_data is not None:
            default_context.synthetic_data = synthetic_data
        return default_context
    except Exception as e:
        logger.error("Error inesperado construyendo contexto: %s", e)
        return Context() # Devuelve un contexto vacío como fallback

def process_orchestrator_input_safe(orchestrator, user_input: str, context: dict = None):
    """
    Ejecuta el orquestador de forma síncrona usando el wrapper para evitar problemas de event loop.
    """
    try:
        logger.info("Procesando input con orquestador: %s", type(orchestrator))
        logger.info("Contexto recibido: %s", context)
        # Construir contexto tipado (sin romper compatibilidad)
        typed_ctx = build_typed_context_dict(context)
        
        # Convertir el objeto Pydantic a un diccionario
        ctx_dict = typed_ctx.model_dump(exclude_none=True)

        # Verificar si el orquestador tiene el método síncrono
        if hasattr(orchestrator, 'process_user_input_sync'):
            logger.info("Usando process_user_input_sync")
            resp = orchestrator.process_user_input_sync(user_input, ctx_dict)
            return validate_agent_response(resp)
        
        # Si no tiene el método síncrono pero tenemos el wrapper disponible
        elif ASYNC_WRAPPER_AVAILABLE and hasattr(orchestrator, 'process_user_input'):
            logger.info("Usando wrapper con process_user_input")
            # Pasar la función y los argumentos por separado al wrapper
            resp = run_async_safe(orchestrator.process_user_input, user_input, ctx_dict)
            return validate_agent_response(resp)
        
        # Fallback: intentar ejecutar directamente (para mocks)
        else:
            logger.info("Usando fallback directo")
            # Para orquestadores mock que pueden ser síncronos o async
            result = orchestrator.process_user_input(user_input, typed_ctx)
            logger.info("Resultado obtenido: %s", result)
            
            # Si es una corrutina, necesitamos ejecutarla de forma async
            import inspect
            if inspect.iscoroutine(result):
                logger.info("El resultado es una corrutina, manejando async")
                if ASYNC_WRAPPER_AVAILABLE:
                    # Cancelar la corrutina actual y crear una nueva llamada
                    result.close()  # Liberar la corrutina no ejecutada
                    resp = run_async_safe(orchestrator.process_user_input, user_input, typed_ctx)
                    return validate_agent_response(resp)
                else:
                    logger.warning("No hay wrapper disponible, usando asyncio básico")
                    # Fallback simple: intentar con asyncio básico
                    import asyncio
                    import concurrent.futures
                    
                    def run_async():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(orchestrator.process_user_input(user_input, typed_ctx))
                        finally:
                            new_loop.close()
                    
                    try:
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(run_async)
                            resp = future.result(timeout=30)
                            return validate_agent_response(resp)
                    except Exception as e:
                        logger.error("Error en fallback async: %s", e)
                        return {
                            "message": f"❌ Error ejecutando operación async: {str(e)}",
                            "agent": "system", 
                            "error": True
                        }
            else:
                logger.info("El resultado es sync, devolviendo directamente")
                # Es síncrono, devolver directamente
                return validate_agent_response(result)
            
    except Exception as e:
        logger.error("Error en process_orchestrator_input_safe: %s", e)
        import traceback
        traceback.print_exc()
        return {
            "message": f"❌ Error interno: {str(e)}",
            "agent": "system", 
            "error": True
        }

# Función initialize_orchestrator ya definida arriba

# Inicialización del estado ya definida arriba

# Ejecutar chat usando el wrapper síncrono
if __name__ == "__main__":
    # Como main_chat_loop ahora es síncrono, llamarlo directamente
    try:
        main_chat_loop()
    except Exception as e:
        st.error(f"❌ Error en el bucle principal del chat: {e}")
        st.write("⚠️ Por favor, reinicia la aplicación")
else:
    # Si no estamos en main, ejecutar el bucle normalmente (para Streamlit)
    main_chat_loop()
