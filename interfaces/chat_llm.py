import streamlit as st
import pandas as pd
import asyncio
import os
import sys
import re
from datetime import datetime
from dotenv import load_dotenv

# Añadir la ruta del proyecto al path de Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Cargar variables de entorno
load_dotenv()

# Importar el selector de columnas médicas
try:
    from src.adapters.medical_column_selector import MedicalColumnSelector
    COLUMN_SELECTOR_AVAILABLE = True
except ImportError as e:
    COLUMN_SELECTOR_AVAILABLE = False
    print(f"⚠️ Selector de columnas no disponible: {e}")

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

# Importar configuración de Azure
try:
    from src.config.azure_config import azure_config
    AZURE_CONFIGURED = True
    try:
        connection_test = azure_config.test_connection()
        print("✅ Azure OpenAI conectado correctamente")
    except Exception as e:
        connection_test = False
        print(f"⚠️ Azure configurado pero sin conexión: {e}")
except Exception as e:
    AZURE_CONFIGURED = False
    connection_test = False
    print(f"⚠️ Error de configuración Azure: {e}")

# Importar LangGraph Orchestrator y agentes con manejo de errores
try:
    from src.orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator, AgentState
    from src.agents.base_agent import BaseLLMAgent, BaseAgentConfig
    from src.agents.coordinator_agent import CoordinatorAgent
    from src.agents.analyzer_agent import ClinicalAnalyzerAgent
    from src.agents.generator_agent import SyntheticGeneratorAgent
    
    # Importar otros agentes con manejo individual de errores
    try:
        from src.agents.validator_agent import MedicalValidatorAgent
        VALIDATOR_AVAILABLE = True
    except Exception as e:
        print(f"⚠️ Validator agent no disponible: {e}")
        VALIDATOR_AVAILABLE = False
    
    try:
        from src.agents.simulator_agent import PatientSimulatorAgent
        SIMULATOR_AVAILABLE = True
    except Exception as e:
        print(f"⚠️ Simulator agent no disponible: {e}")
        SIMULATOR_AVAILABLE = False
    
    try:
        from src.agents.evaluator_agent import UtilityEvaluatorAgent
        EVALUATOR_AVAILABLE = True
    except Exception as e:
        print(f"⚠️ Evaluator agent no disponible: {e}")
        EVALUATOR_AVAILABLE = False

    AGENTS_AVAILABLE = True
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph Orchestrator y Agentes cargados correctamente")
    
except Exception as e:
    AGENTS_AVAILABLE = False
    LANGGRAPH_AVAILABLE = False
    VALIDATOR_AVAILABLE = False
    SIMULATOR_AVAILABLE = False
    EVALUATOR_AVAILABLE = False
    print(f"❌ Error cargando agentes: {e}")

# Agente mock para desarrollo
class MockAgent:
    def __init__(self, name):
        self.name = name
        self.config = type('Config', (), {'name': name})()
    
    async def process(self, input_text, context=None):
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
                "message": f"👋 **¡Hola!** Soy tu asistente de IA para generar datos clínicos sintéticos.\n\n🔬 **Estado:** {'✅ Azure OpenAI Conectado' if connection_test else '🔄 Modo Simulado'}{dataset_msg}\n\n**🧠 Mi equipo especializado:**\n• **Analista** - Extrae patrones clínicos\n• **Generador** - Crea datos sintéticos con SDV\n• **Validador** - Verifica coherencia médica\n• **Simulador** - Modela evolución temporal\n• **Evaluador** - Mide calidad y utilidad\n\n¿En qué puedo ayudarte hoy?",
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
                rows = context.get("rows", 0)
                return {
                    "message": f"""🧬 **Generación sintética completada (modo simulado)**

**📊 Dataset base:** {rows:,} registros
**🎯 Registros generados:** 100 (simulado)
**🔬 Modelo utilizado:** CTGAN (simulado)

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

@st.cache_resource
def initialize_langgraph_orchestrator():
    """Inicializa el orquestador LangGraph con agentes"""
    if AGENTS_AVAILABLE and LANGGRAPH_AVAILABLE and AZURE_CONFIGURED:
        try:
            agents = {
                "coordinator": CoordinatorAgent(),
                "analyzer": ClinicalAnalyzerAgent(),
                "generator": SyntheticGeneratorAgent(),
            }
            
            # Agregar agentes opcionales solo si están disponibles
            if VALIDATOR_AVAILABLE:
                agents["validator"] = MedicalValidatorAgent()
                print("✅ Validator agent agregado")
            
            if SIMULATOR_AVAILABLE:
                agents["simulator"] = PatientSimulatorAgent()
                print("✅ Simulator agent agregado")
            
            if EVALUATOR_AVAILABLE:
                agents["evaluator"] = UtilityEvaluatorAgent()
                print("✅ Evaluator agent agregado")
            
            orchestrator = MedicalAgentsOrchestrator(agents)
            print("✅ LangGraph Orchestrator inicializado con agentes reales")
            return orchestrator
        except Exception as e:
            st.error(f"Error inicializando LangGraph Orchestrator: {e}")
            print(f"❌ Error en LangGraph: {e}")
    
    print("⚠️ Usando orquestador mock")
    return create_mock_orchestrator()

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
            self.state["context"] = context or {}
            
            # Detectar intención y ejecutar agente correspondiente
            if any(word in user_input.lower() for word in ["analizar", "análisis", "analiza"]):
                self.state["current_agent"] = "analyzer"
                agent = self.agents["analyzer"]
                response = await agent.process(user_input, context)
                return response
            elif any(word in user_input.lower() for word in ["generar", "sintético", "sintéticos", "genera"]):
                self.state["current_agent"] = "generator"
                agent = self.agents["generator"]
                response = await agent.process(user_input, context)
                return response
            elif any(word in user_input.lower() for word in ["validar", "valida", "validación"]):
                self.state["current_agent"] = "validator"
                return {"message": "✅ Validación completada (modo simulado)\n\nDatos validados exitosamente.", "agent": "validator"}
            elif any(word in user_input.lower() for word in ["evaluar", "evalúa", "calidad"]):
                self.state["current_agent"] = "evaluator"
                return {"message": "📊 Evaluación completada (modo simulado)\n\nCalidad de datos: Excelente", "agent": "evaluator"}
            elif any(word in user_input.lower() for word in ["simular", "simula", "paciente"]):
                self.state["current_agent"] = "simulator"
                return {"message": "🏥 Simulación completada (modo simulado)\n\nEvolución de pacientes simulada.", "agent": "simulator"}
            else:
                self.state["current_agent"] = "coordinator"
                agent = self.agents["coordinator"]
                response = await agent.process(user_input, context)
                return response
    
    return MockLangGraphOrchestrator(mock_agents)

@st.cache_resource
def initialize_orchestrator():
    """Inicializa el orquestador principal"""
    return initialize_langgraph_orchestrator()

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
if AZURE_CONFIGURED and connection_test:
    status_text = "✅ Azure OpenAI Conectado"
    status_color = "#10b981"
elif AZURE_CONFIGURED:
    status_text = "🟡 Azure Configurado (Sin conexión)"
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
    
    if st.button("Generar Sintéticos", use_container_width=True):
        if st.session_state.get('file_uploaded'):
            st.session_state.quick_command = "generar datos sintéticos"
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

# Container principal del chat
with st.container():
    # Mostrar historial de chat
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
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
        # Mensaje de bienvenida centrado
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 120px;">
                <strong style="font-size: 1.5rem; text-align: center;">Bienvenido a Patient AI</strong>
                <span style="font-size: 1.1rem; color: #374151; text-align: center;">Asistente de Generación de datos clínicos sintéticos.</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Crear dos columnas internas centradas para el contenido
        left, center1, center2, right = st.columns([1.5,2,2,0.7])
        with center1:
            st.markdown("""
            **¿Qué puedo hacer por ti?**  
            - Analizar datasets médicos existentes  
            - Generar datos sintéticos realistas y seguros  
            - Validar la coherencia clínica de los datos  
            - Simular la evolución temporal de pacientes  
            """)
        with center2:
            st.markdown("""
            **Para comenzar:**  
            - Sube un archivo CSV/Excel con datos clínicos
            - Pregunta sobre generación de datos sintéticos 
            - Solicita análisis o validación de datos 
            - Escribe "ayuda" para ver más opciones  
            """)

# --- LÓGICA PRINCIPAL DEL CHAT ---
async def main_chat_loop():
    """Función asíncrona para manejar el bucle principal del chat."""
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
            
            response = await st.session_state.orchestrator.process_user_input(prompt, context_with_selections)
            full_response = response.get("message", "No se recibió respuesta.")
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": full_response, 
                "agent": response.get("agent"),
                "dataset_type": response.get("dataset_type")
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
        st.info(f"🎯 **Columnas seleccionadas para generación sintética:** {', '.join(st.session_state.selected_columns[:5])}{'...' if len(st.session_state.selected_columns) > 5 else ''}")
    
    # Chat input principal de Streamlit
    if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.spinner("Procesando..."):
            # Preparar contexto incluyendo columnas seleccionadas si existen
            context_with_selections = st.session_state.context.copy()
            if st.session_state.get('selected_columns'):
                context_with_selections['selected_columns'] = st.session_state.selected_columns
            
            response = await st.session_state.orchestrator.process_user_input(prompt, context_with_selections)
            full_response = response.get("message", "No se recibió respuesta.")
            
            if response.get("error"):
                error_message = response.get("error")
                if isinstance(error_message, bool):
                    error_message = "Error en el procesamiento del dataset"
                elif not isinstance(error_message, str):
                    error_message = str(error_message)
                
                st.error(f"❌ **Error**: {error_message}")
                return
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": full_response, 
                "agent": response.get("agent"),
                "dataset_type": response.get("dataset_type")
            })
        
        st.rerun()

# --- EJECUCIÓN DEL BUCLE ASÍNCRONO ---
if __name__ == "__main__":
    try:
        asyncio.run(main_chat_loop())
    except Exception as e:
        st.error(f"Ocurrió un error en la aplicación: {e}")
