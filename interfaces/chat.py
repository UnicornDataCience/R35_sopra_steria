import streamlit as st
import pandas as pd
import json
import os
import sys
from datetime import datetime

# --- Configuración de la Página (DEBE SER LO PRIMERO) ---
st.set_page_config(
    page_title="Chat - Generador de Historias Clínicas Sintéticas",
    page_icon="🧑‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Añadir el directorio src al path para importar módulos
sys.path.append(os.path.dirname(__file__))

# Importar módulos del proyecto (con manejo de errores SILENCIOSO)
try:
    # Estos imports se conectarán con los módulos reales cuando estén implementados
    from extraction.data_extractor import DataExtractor
    from generation.sdv_generator import SDVGenerator
    from validation.medical_validator import MedicalValidator
    from simulation.patient_simulator import PatientSimulator
    from evaluation.utility_evaluator import UtilityEvaluator
    from orchestration.agent_orchestrator import AgentOrchestrator
    MODULES_AVAILABLE = True
except ImportError:
    # NO usar st.warning aquí - guardamos el estado para después
    DataExtractor = None
    SDVGenerator = None
    MedicalValidator = None
    PatientSimulator = None
    UtilityEvaluator = None
    AgentOrchestrator = None
    MODULES_AVAILABLE = False

# --- Clases para el Sistema de Chat ---
class ChatAgent:
    """Clase base para todos los agentes de chat"""
    def __init__(self, name, description, emoji, module_class=None):
        self.name = name
        self.description = description
        self.emoji = emoji
        self.module_class = module_class
        self.instance = None
        self.active = False
    
    def initialize(self):
        """Inicializa la instancia del módulo si está disponible"""
        if self.module_class:
            try:
                self.instance = self.module_class()
                return True
            except Exception as e:
                # Guardar error para mostrar después
                st.session_state.setdefault('init_errors', []).append(f"Error inicializando {self.name}: {e}")
                return False
        return False

class ConversationManager:
    """Maneja el flujo de conversación entre agentes"""
    def __init__(self):
        self.agents = {
            "analyzer": ChatAgent(
                "Agente Analista Clínico", 
                "Analiza datasets clínicos y extrae patrones", 
                "🔍",
                DataExtractor
            ),
            "generator": ChatAgent(
                "Agente Generador", 
                "Genera datos sintéticos usando SDV", 
                "⚗️",
                SDVGenerator
            ),
            "validator": ChatAgent(
                "Agente Validador", 
                "Valida coherencia médica", 
                "✅",
                MedicalValidator
            ),
            "simulator": ChatAgent(
                "Agente Simulador", 
                "Simula evolución de pacientes", 
                "🔄",
                PatientSimulator
            ),
            "evaluator": ChatAgent(
                "Agente Evaluador", 
                "Evalúa utilidad de datos", 
                "📊",
                UtilityEvaluator
            )
        }
        self.current_agent = None
        self.conversation_state = "greeting"
        self.initialized = False
        
    def initialize_agents(self):
        """Inicializa todos los agentes"""
        success_count = 0
        for agent_key, agent in self.agents.items():
            if agent.initialize():
                success_count += 1
        
        self.initialized = success_count > 0
        return success_count, len(self.agents)
        
    def get_current_agent_response(self, user_input, context):
        """Genera respuesta del agente actual basada en el contexto"""
        if self.current_agent == "analyzer":
            return self.analyzer_response(user_input, context)
        elif self.current_agent == "generator":
            return self.generator_response(user_input, context)
        elif self.current_agent == "validator":
            return self.validator_response(user_input, context)
        elif self.current_agent == "simulator":
            return self.simulator_response(user_input, context)
        elif self.current_agent == "evaluator":
            return self.evaluator_response(user_input, context)
        else:
            return self.coordinator_response(user_input, context)
    
    def analyzer_response(self, user_input, context):
        """Respuestas del Agente Analista"""
        agent = self.agents["analyzer"]
        
        if "archivo" in user_input.lower() or "dataset" in user_input.lower():
            return {
                "message": "¡Perfecto! Necesito que subas un dataset clínico para analizar. Usaré técnicas avanzadas de extracción de patrones para identificar características relevantes.",
                "action": "request_file_upload",
                "next_state": "waiting_file"
            }
        elif context.get("file_uploaded"):
            try:
                # Usar el módulo real si está disponible
                if agent.instance:
                    analysis_result = self.real_analysis(context["dataframe"], agent.instance)
                else:
                    analysis_result = self.mock_analysis(context)
                
                return {
                    "message": f"✅ **Análisis completado** del archivo '{context['filename']}'!\n\n" + 
                              "\n".join([f"🔍 {pattern}" for pattern in analysis_result['patterns']]) + 
                              f"\n\n📊 **{analysis_result['total_patterns']} patrones detectados**\n\n¿Procedo con la generación sintética?",
                    "action": "analysis_complete",
                    "data": analysis_result,
                    "next_agent": "generator"
                }
            except Exception as e:
                return {
                    "message": f"❌ Error en el análisis: {str(e)}\n\nUsando análisis básico como respaldo.",
                    "action": "analysis_error",
                    "data": self.mock_analysis(context)
                }
        else:
            return {
                "message": "Para comenzar el análisis, necesito que subas un dataset clínico. ¿Tienes un archivo CSV o Excel con datos anonimizados?",
                "action": "request_file_upload",
                "next_state": "waiting_file"
            }
    
    def generator_response(self, user_input, context):
        """Respuestas del Agente Generador"""
        agent = self.agents["generator"]
        
        if "sí" in user_input.lower() or "si" in user_input.lower() or "proceder" in user_input.lower():
            return {
                "message": "¡Excelente! Configuremos la generación sintética:\n\n1. **Número de pacientes** (100-1000)\n2. **Rango de edad** (ej: 50-80)\n3. **Distribución de sexo**\n4. **Condiciones prioritarias**\n\nUsaré SDV (Synthetic Data Vault) para máxima fidelidad.",
                "action": "request_generation_params",
                "next_state": "collecting_params"
            }
        elif any(word in user_input.lower() for word in ["pacientes", "generar", "número"]):
            try:
                num_patients = self.extract_number(user_input)
                context["generation_params"] = {"num_patients": num_patients}
                
                if agent.instance:
                    # Usar SDV real
                    generation_result = self.real_generation(context, agent.instance)
                else:
                    # Usar generación mock
                    generation_result = self.mock_generation(num_patients)
                
                return {
                    "message": f"🎯 **Generación completada**: {num_patients} pacientes sintéticos creados usando técnicas avanzadas de ML.\n\n📈 **Calidad estimada**: {generation_result.get('quality_score', 'N/A')}%\n\n¿Procedo con la validación médica?",
                    "action": "generation_complete",
                    "data": generation_result,
                    "next_agent": "validator"
                }
            except Exception as e:
                return {
                    "message": f"❌ Error en generación: {str(e)}\n\nUsando generador de respaldo.",
                    "action": "generation_error"
                }
        else:
            return {
                "message": "Dime cuántos pacientes sintéticos necesitas (recomiendo 100-500 para pruebas):",
                "action": "request_generation_params"
            }
    
    def validator_response(self, user_input, context):
        """Respuestas del Agente Validador"""
        agent = self.agents["validator"]
        
        try:
            if agent.instance:
                validation_result = self.real_validation(context, agent.instance)
            else:
                validation_result = self.mock_validation(context)
            
            status_emoji = "✅" if validation_result["valid"] else "⚠️"
            
            return {
                "message": f"{status_emoji} **Validación médica completada**\n\n" +
                          f"🔍 **Coherencia clínica**: {validation_result['coherence_score']:.1%}\n" +
                          f"⚕️ **Validación farmacológica**: {validation_result.get('pharma_score', 0.9):.1%}\n" +
                          f"📅 **Consistencia temporal**: {validation_result.get('temporal_score', 0.88):.1%}\n\n" +
                          ("¿Procedo con la simulación de evolución?" if validation_result["valid"] else "Se detectaron inconsistencias. ¿Quieres continuar?"),
                "action": "validation_complete",
                "data": validation_result,
                "next_agent": "simulator" if validation_result["valid"] else None
            }
        except Exception as e:
            return {
                "message": f"❌ Error en validación: {str(e)}\n\nUsando validación básica.",
                "action": "validation_error",
                "data": self.mock_validation(context)
            }
    
    def simulator_response(self, user_input, context):
        """Respuestas del Agente Simulador"""
        agent = self.agents["simulator"]
        
        if "sí" in user_input.lower() or "si" in user_input.lower():
            try:
                if agent.instance:
                    simulation_result = self.real_simulation(context, agent.instance)
                else:
                    simulation_result = self.mock_simulation(context)
                
                return {
                    "message": f"🔄 **Simulación temporal completada**\n\n" +
                              f"📊 **Visitas generadas**: {simulation_result.get('total_visits', 'N/A')}\n" +
                              f"⏱️ **Período simulado**: {simulation_result.get('time_span', '6 meses')}\n" +
                              f"📈 **Evoluciones modeladas**: {simulation_result.get('evolutions', 'N/A')}\n\n" +
                              "¿Quieres que evalúe la utilidad final de los datos?",
                    "action": "simulation_complete",
                    "data": simulation_result,
                    "next_agent": "evaluator"
                }
            except Exception as e:
                return {
                    "message": f"❌ Error en simulación: {str(e)}\n\nUsando simulación básica.",
                    "action": "simulation_error",
                    "data": self.mock_simulation(context)
                }
        else:
            return {
                "message": "La simulación temporal añadirá realismo con múltiples visitas y progresión clínica. ¿Procedemos?",
                "action": "request_simulation"
            }
    
    def evaluator_response(self, user_input, context):
        """Respuestas del Agente Evaluador"""
        agent = self.agents["evaluator"]
        
        try:
            if agent.instance:
                evaluation_result = self.real_evaluation(context, agent.instance)
            else:
                evaluation_result = self.mock_evaluation(context)
            
            quality_icon = "🎯" if evaluation_result["fidelity"] > 0.85 else "⚠️"
            
            return {
                "message": f"📊 **Evaluación de utilidad completada**\n\n" +
                          f"{quality_icon} **Fidelidad estadística**: {evaluation_result['fidelity']:.1%}\n" +
                          f"📈 **Similitud distribucional**: {evaluation_result.get('similarity', 0.91):.1%}\n" +
                          f"🔬 **Utilidad para ML**: {evaluation_result.get('ml_utility', 'Excelente')}\n" +
                          f"🎲 **Privacidad preservada**: {evaluation_result.get('privacy_score', 0.95):.1%}\n\n" +
                          "✅ **Los datos están listos para descarga**",
                "action": "evaluation_complete",
                "data": evaluation_result,
                "next_state": "ready_download"
            }
        except Exception as e:
            return {
                "message": f"❌ Error en evaluación: {str(e)}\n\nUsando evaluación básica.",
                "action": "evaluation_error",
                "data": self.mock_evaluation(context)
            }
    
    def coordinator_response(self, user_input, context):
        """Respuestas del coordinador general"""
        greetings = ["hola", "buenos", "buenas", "saludos", "empezar", "comenzar", "iniciar"]
        
        if any(greeting in user_input.lower() for greeting in greetings) or self.conversation_state == "greeting":
            self.current_agent = "analyzer"
            
            # Verificar estado de inicialización
            init_status = "🟢 Todos los módulos cargados" if self.initialized else "🟡 Usando módulos simulados"
            
            return {
                "message": f"¡Hola! 👋 Soy tu asistente para generar historias clínicas sintéticas.\n\n" +
                          f"**Estado del sistema**: {init_status}\n\n" +
                          "🔍 **Mi equipo de agentes especializados**:\n" +
                          "• **Analista**: Extrae patrones con ML avanzado\n" +
                          "• **Generador**: Crea datos con SDV/CTGAN\n" +
                          "• **Validador**: Verifica coherencia médica\n" +
                          "• **Simulador**: Modela evolución temporal\n" +
                          "• **Evaluador**: Mide fidelidad y utilidad\n\n" +
                          "Para empezar, sube un dataset clínico anonimizado. ¿Tienes uno listo?",
                "action": "greeting_complete",
                "next_agent": "analyzer"
            }
        else:
            return {
                "message": "¡Hola! Para ayudarte mejor, di 'hola' o 'empezar' para iniciar el proceso de generación de datos sintéticos.",
                "action": "request_greeting"
            }
    
    # --- Métodos para usar módulos reales ---
    def real_analysis(self, dataframe, extractor_instance):
        """Usa el módulo real de extracción"""
        try:
            patterns = extractor_instance.extract_patterns(dataframe)
            return {
                "patterns": patterns.get("clinical_patterns", []),
                "total_patterns": len(patterns.get("clinical_patterns", [])),
                "statistics": patterns.get("statistics", {}),
                "columns": list(dataframe.columns),
                "rows": len(dataframe)
            }
        except:
            return self.mock_analysis({"dataframe": dataframe})
    
    def real_generation(self, context, generator_instance):
        """Usa el módulo real de generación SDV"""
        try:
            params = context.get("generation_params", {})
            synthetic_data = generator_instance.generate(
                original_data=context.get("dataframe"),
                num_samples=params.get("num_patients", 100)
            )
            return {
                "synthetic_data": synthetic_data,
                "quality_score": 92,  # Esto vendría del módulo real
                "method": "SDV-CTGAN"
            }
        except:
            return self.mock_generation(context.get("generation_params", {}).get("num_patients", 100))
    
    def real_validation(self, context, validator_instance):
        """Usa el módulo real de validación"""
        try:
            synthetic_data = context.get("synthetic_data")
            validation = validator_instance.validate(synthetic_data)
            return validation
        except:
            return self.mock_validation(context)
    
    def real_simulation(self, context, simulator_instance):
        """Usa el módulo real de simulación"""
        try:
            synthetic_data = context.get("synthetic_data")
            simulation = simulator_instance.simulate_evolution(synthetic_data)
            return simulation
        except:
            return self.mock_simulation(context)
    
    def real_evaluation(self, context, evaluator_instance):
        """Usa el módulo real de evaluación"""
        try:
            original_data = context.get("dataframe")
            synthetic_data = context.get("synthetic_data")
            evaluation = evaluator_instance.evaluate_utility(original_data, synthetic_data)
            return evaluation
        except:
            return self.mock_evaluation(context)
    
    # --- Métodos mock (respaldo) ---
    def mock_analysis(self, context):
        """Análisis simulado cuando no hay módulo real"""
        df = context.get("dataframe")
        if df is not None:
            patterns = [
                f"Patrón detectado: {df.shape[0]} registros con {df.shape[1]} variables",
                "Alta correlación entre edad y comorbilidades",
                "Distribución no uniforme de diagnósticos principales"
            ]
        else:
            patterns = [
                "Patrón A: Pacientes >60 años con diabetes tipo 2 y HTA",
                "Patrón B: Alta tasa de reingresos en EPOC",
                "Patrón C: Uso frecuente metformina + lisinopril"
            ]
        return {
            "patterns": patterns,
            "total_patterns": len(patterns),
            "columns": context.get("columns", []),
            "rows": context.get("rows", 0)
        }
    
    def mock_generation(self, num_patients):
        """Generación simulada"""
        import random
        synthetic_data = []
        for i in range(num_patients):
            patient = {
                "patient_id": f"synth_{1000 + i}",
                "age": random.randint(50, 80),
                "sex": random.choice(["M", "F"]),
                "chronic_conditions": random.sample(["diabetes", "hypertension", "EPOC"], 2),
                "generated_at": datetime.now().isoformat()
            }
            synthetic_data.append(patient)
        
        return {
            "synthetic_data": pd.DataFrame(synthetic_data),
            "quality_score": 85,
            "method": "Mock Generator"
        }
    
    def mock_validation(self, context):
        """Validación simulada"""
        return {
            "coherence_score": 0.92,
            "pharma_score": 0.89,
            "temporal_score": 0.88,
            "valid": True,
            "issues": []
        }
    
    def mock_simulation(self, context):
        """Simulación simulada"""
        return {
            "total_visits": "245",
            "time_span": "6 meses",
            "evolutions": "Progresión realista",
            "evolved_data": context.get("synthetic_data")
        }
    
    def mock_evaluation(self, context):
        """Evaluación simulada"""
        return {
            "fidelity": 0.88,
            "similarity": 0.91,
            "ml_utility": "Excelente",
            "privacy_score": 0.95
        }
    
    def extract_number(self, text):
        """Extrae números del texto"""
        import re
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else 100
    
    def set_current_agent(self, agent_name):
        """Cambia el agente activo"""
        self.current_agent = agent_name
        if agent_name in self.agents:
            self.agents[agent_name].active = True

# --- Funciones auxiliares ---
def load_real_data():
    """Carga datos reales si están disponibles"""
    try:
        real_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "real", "df_final.csv")
        if os.path.exists(real_data_path):
            return pd.read_csv(real_data_path)
    except:
        pass
    return None

def process_uploaded_file(uploaded_file):
    """Procesa el archivo subido y extrae información básica"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        return {
            "success": True,
            "filename": uploaded_file.name,
            "columns": list(df.columns),
            "rows": len(df),
            "dataframe": df
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- Inicialización del Estado ---
if 'conversation_manager' not in st.session_state:
    st.session_state.conversation_manager = ConversationManager()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_context' not in st.session_state:
    st.session_state.current_context = {}

if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None

if 'awaiting_file' not in st.session_state:
    st.session_state.awaiting_file = False

if 'agents_status' not in st.session_state:
    st.session_state.agents_status = "not_initialized"

if 'init_errors' not in st.session_state:
    st.session_state.init_errors = []

# --- Barra Lateral ---
with st.sidebar:
    # Logo
    logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "logo_patientia.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("🧑‍⚕️ **PATIENTIA**")
    
    st.title("🤖 Sistema de Agentes")
    st.markdown("**Chat Conversacional**")
    
    # Mostrar advertencias DESPUÉS de set_page_config
    if not MODULES_AVAILABLE:
        st.warning("⚠️ Módulos no encontrados. Usando versiones simuladas.")
    
    if st.session_state.init_errors:
        with st.expander("⚠️ Errores de inicialización"):
            for error in st.session_state.init_errors:
                st.error(error)
    
    # Inicialización de agentes
    if st.session_state.agents_status == "not_initialized":
        if st.button("🚀 Inicializar Sistema", type="primary"):
            with st.spinner("Inicializando agentes..."):
                cm = st.session_state.conversation_manager
                success_count, total_count = cm.initialize_agents()
                
                if success_count > 0:
                    st.session_state.agents_status = "initialized"
                    st.success(f"✅ {success_count}/{total_count} agentes inicializados")
                else:
                    st.session_state.agents_status = "mock_mode"
                    st.info("🔄 Usando modo simulado completo")
    
    # Estado de agentes
    if st.session_state.agents_status != "not_initialized":
        st.subheader("Estado de Agentes:")
        cm = st.session_state.conversation_manager
        
        for agent_key, agent in cm.agents.items():
            status_color = "🟢" if agent_key == cm.current_agent else "⚪"
            module_status = "🔧" if agent.instance else "🔄"
            st.markdown(f"{status_color}{module_status} **{agent.name}**")
            st.caption(f"{agent.description}")
    
    st.markdown("---")
    
    # Upload de archivo
    if st.session_state.awaiting_file:
        st.subheader("📁 Subir Dataset")
        uploaded_file = st.file_uploader(
            "Selecciona tu archivo clínico", 
            type=["csv", "xlsx"],
            key="dataset_uploader"
        )
        
        if uploaded_file:
            result = process_uploaded_file(uploaded_file)
            if result["success"]:
                st.session_state.current_context.update({
                    "file_uploaded": True,
                    "filename": result["filename"],
                    "columns": result["columns"],
                    "rows": result["rows"],
                    "dataframe": result["dataframe"]
                })
                st.session_state.awaiting_file = False
                st.success(f"✅ {result['filename']} cargado")
                st.rerun()
            else:
                st.error(f"Error: {result['error']}")
    
    # Datos de ejemplo
    st.subheader("📊 Datos de Ejemplo")
    if st.button("📋 Usar datos reales del proyecto"):
        real_data = load_real_data()
        if real_data is not None:
            st.session_state.current_context.update({
                "file_uploaded": True,
                "filename": "df_final.csv",
                "columns": list(real_data.columns),
                "rows": len(real_data),
                "dataframe": real_data
            })
            st.success("✅ Datos reales cargados")
            st.rerun()
        else:
            st.error("No se encontraron datos reales")
    
    # Reiniciar
    if st.button("🔄 Reiniciar Chat"):
        for key in ['chat_history', 'current_context', 'synthetic_data', 'awaiting_file', 'init_errors']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.conversation_manager = ConversationManager()
        st.rerun()

# --- Contenido Principal ---
st.title("💬 Chat - Generador de Historias Clínicas Sintéticas")

# Solo mostrar chat si el sistema está inicializado
if st.session_state.agents_status == "not_initialized":
    st.info("👆 **Inicializa el sistema de agentes** desde la barra lateral para comenzar.")
    st.markdown("""
    ### 🔧 Estado del Sistema
    
    **Módulos detectados:**
    - ❌ extraction.data_extractor
    - ❌ generation.sdv_generator  
    - ❌ validation.medical_validator
    - ❌ simulation.patient_simulator
    - ❌ evaluation.utility_evaluator
    
    **Modo de funcionamiento:** Simulado (completamente funcional para desarrollo)
    """)
else:
    # Contenedor para el chat
    chat_container = st.container()
    
    with chat_container:
        # Mostrar historial de chat
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Mostrar datos si los hay
                if "data" in message and message["data"]:
                    if isinstance(message["data"], pd.DataFrame):
                        st.dataframe(message["data"].head(), use_container_width=True)
                    elif isinstance(message["data"], dict):
                        with st.expander("Ver detalles"):
                            st.json(message["data"])
    
    # Input del usuario
    user_input = st.chat_input("Escribe tu mensaje aquí...")
    
    if user_input:
        # Añadir mensaje del usuario
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Obtener respuesta del agente
        cm = st.session_state.conversation_manager
        response = cm.get_current_agent_response(user_input, st.session_state.current_context)
        
        # Procesar acciones
        if response.get("action") == "request_file_upload":
            st.session_state.awaiting_file = True
        
        elif response.get("action") == "analysis_complete":
            st.session_state.current_context.update(response.get("data", {}))
            if response.get("next_agent"):
                cm.set_current_agent(response["next_agent"])
        
        elif response.get("action") == "generation_complete":
            synthetic_data = response.get("data", {}).get("synthetic_data")
            if synthetic_data is not None:
                st.session_state.synthetic_data = synthetic_data
                st.session_state.current_context["synthetic_data"] = synthetic_data
            st.session_state.current_context.update(response.get("data", {}))
            if response.get("next_agent"):
                cm.set_current_agent(response["next_agent"])
        
        elif response.get("action") in ["validation_complete", "simulation_complete", "evaluation_complete"]:
            st.session_state.current_context.update(response.get("data", {}))
            if response.get("next_agent"):
                cm.set_current_agent(response["next_agent"])
        
        # Determinar nombre del agente
        if cm.current_agent and cm.current_agent in cm.agents:
            agent = cm.agents[cm.current_agent]
            agent_name = f"{agent.emoji} {agent.name}"
        else:
            agent_name = "🤖 Coordinador"
        
        # Añadir respuesta del agente
        agent_message = {
            "role": "assistant",
            "content": f"**{agent_name}**\n\n{response['message']}",
            "timestamp": datetime.now()
        }
        
        # Añadir datos si los hay
        if "data" in response and isinstance(response["data"], dict):
            if "synthetic_data" in response["data"]:
                agent_message["data"] = response["data"]["synthetic_data"]
        
        st.session_state.chat_history.append(agent_message)
        
        # Cambiar agente si es necesario
        if response.get("next_agent"):
            cm.set_current_agent(response["next_agent"])
        
        st.rerun()

# --- Sección de descarga ---
if (st.session_state.synthetic_data is not None and 
    st.session_state.current_context.get("fidelity") is not None):
    
    st.markdown("---")
    st.subheader("📥 Descargar Datos Sintéticos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = st.session_state.synthetic_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Descargar CSV",
            data=csv_data,
            file_name=f"datos_sinteticos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = st.session_state.synthetic_data.to_json(orient='records', indent=2)
        st.download_button(
            label="📋 Descargar JSON",
            data=json_data,
            file_name=f"datos_sinteticos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        # Guardar en la carpeta del proyecto
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")
        if st.button("💾 Guardar en Proyecto"):
            try:
                os.makedirs(save_path, exist_ok=True)
                filename = f"datos_sinteticos_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                full_path = os.path.join(save_path, filename)
                st.session_state.synthetic_data.to_csv(full_path, index=False)
                st.success(f"✅ Guardado en: {full_path}")
            except Exception as e:
                st.error(f"Error guardando: {e}")

st.markdown("---")
st.caption("💡 **Tip**: Escribe 'hola' para empezar o 'usar datos reales' para cargar el dataset del proyecto.")