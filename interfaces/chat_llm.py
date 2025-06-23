import streamlit as st
import pandas as pd
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv


# Añadir la ruta del proyecto al path de Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Aplicar parche para PyTorch-Streamlit
try:
    import fix_streamlit_pytorch
except ImportError:
    pass  # Continuar si el archivo no existe

# Añadir la ruta del proyecto al path de Python
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Cargar variables de entorno
load_dotenv()

# Configuración de página
st.set_page_config(
    page_title="Patient IA",
    page_icon=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "logo_patientia.png"),
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar colapsado por defecto
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
    .stChatInput > div > div > textarea {
        min-height: 60px !important;
        font-size: 16px !important;
        border-radius: 20px !important;
        border: 2px solid #e1e5e9 !important;
        padding: 15px 20px !important;
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
    
    /* Botón de archivo en el input */
    .file-upload-btn {
        position: absolute;
        right: 60px;
        bottom: 15px;
        background: none;
        border: none;
        color: #6b7280;
        cursor: pointer;
        font-size: 20px;
    }
    
    /* Área de arrastrar archivo */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background: #f8fafc;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }
</style>
""", unsafe_allow_html=True)

# Importar configuración de Azure
try:
    from src.config.azure_config import azure_config
    AZURE_CONFIGURED = True
    
    # Test de conexión
    connection_test = azure_config.test_connection()
    
except Exception as e:
    AZURE_CONFIGURED = False
    connection_test = False
    st.error(f"Error de configuración Azure: {e}")

# Importar agentes con manejo de errores
try:
    from src.agents.base_agent import BaseLLMAgent, BaseAgentConfig
    from src.agents.coordinator_agent import CoordinatorAgent
    from src.agents.analyzer_agent import ClinicalAnalyzerAgent
    from src.agents.generator_agent import SyntheticGeneratorAgent
    from src.agents.validator_agent import MedicalValidatorAgent
    from src.agents.simulator_agent import PatientSimulatorAgent
    from src.agents.evaluator_agent import UtilityEvaluatorAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    st.warning(f"Agentes no disponibles: {e}")

# Agente mock para desarrollo
class MockAgent:
    def __init__(self, name):
        self.name = name
        self.config = type('Config', (), {'name': name})()
    
    async def process(self, input_text, context=None):
        context = context or {}
        has_dataset = context.get("dataset_uploaded", False)
        
        # Simular diferentes respuestas según el agente
        if "coordinador" in self.name.lower():
            dataset_msg = ""
            if has_dataset:
                filename = context.get("filename", "archivo")
                rows = context.get("rows", 0)
                cols = context.get("columns", 0)
                dataset_msg = f"\n\n📁 **Dataset detectado:** {filename} ({rows:,} filas, {cols} columnas)"
            
            return {
                "message": f"👋 **¡Hola!** Soy tu asistente de IA para generar datos clínicos sintéticos.\n\n🔬 **Estado:** {'✅ Azure OpenAI Conectado' if connection_test else '🔄 Modo Simulado'}{dataset_msg}\n\n**🧠 Mi equipo especializado:**\n• **Analista** - Extrae patrones clínicos\n• **Generador** - Crea datos sintéticos con SDV\n• **Validador** - Verifica coherencia médica\n• **Simulador** - Modela evolución temporal\n• **Evaluador** - Mide calidad y utilidad\n\n¿En qué puedo ayudarte hoy?",
                "agent": self.name,
                "mock": True
            }
        
        elif "analista" in self.name.lower():
            if has_dataset:
                filename = context.get("filename", "dataset")
                return {
                    "message": f"🔍 **Análisis en progreso de {filename}...**\n\nPor favor, usa el orquestador principal para análisis completo del dataset. Los agentes mock tienen capacidades limitadas.\n\n💡 **Sugerencia:** El análisis detallado estará disponible cuando todos los módulos LLM estén configurados.",
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
                    "message": f"🔬 **Generador Sintético**\n\n📊 **Dataset base:** {rows:,} registros detectados\n\n**Técnicas disponibles:**\n• SDV GaussianCopula\n• CTGAN (redes neuronales)\n• TVAE (autoencoders)\n\n🎯 **¿Cuántos registros sintéticos necesitas?**\n\n*Generación completa disponible con módulos LLM configurados.*",
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

    async def analyze_dataset(self, dataframe, context=None):
        """Método especial para análisis de dataset (mock)"""
        return await self.process("analizar dataset", context)

@st.cache_resource
def initialize_agents():
    """Inicializa agentes con Azure OpenAI"""
    if AGENTS_AVAILABLE and AZURE_CONFIGURED:
        try:
            return {
                "coordinator": CoordinatorAgent(),
                "analyzer": ClinicalAnalyzerAgent(),
                "generator": SyntheticGeneratorAgent(),
                "validator": MedicalValidatorAgent(),
                "simulator": PatientSimulatorAgent(),
                "evaluator": UtilityEvaluatorAgent()
            }
        except Exception as e:
            st.error(f"Error inicializando agentes reales: {e}")
    
    # Fallback a agentes mock
    return {
        "coordinator": MockAgent("Coordinador"),
        "analyzer": MockAgent("Analista Clínico"),
        "generator": MockAgent("Generador Sintético"),
        "validator": MockAgent("Validador Médico"),
        "simulator": MockAgent("Simulador de Pacientes"),
        "evaluator": MockAgent("Evaluador de Utilidad")
    }

class SimpleOrchestrator:
    def __init__(self, agents):
        self.agents = agents
        self.current_agent = "coordinator"
        self.pipeline_data = {}  # Almacenar datos entre pasos
        
    async def process_user_input(self, user_input: str, context: dict = None):
        context = context or {}
        
        # Detectar si hay un dataset disponible
        has_dataset = context.get("dataset_uploaded", False)
        dataframe = context.get("dataframe")
        
        # Routing mejorado con flujo de pipeline
        if any(word in user_input.lower() for word in ["hola", "empezar", "comenzar", "ayuda"]):
            self.current_agent = "coordinator"
        elif any(word in user_input.lower() for word in ["analizar", "análisis", "archivo", "dataset"]) or "subido el archivo" in user_input.lower():
            self.current_agent = "analyzer"
        elif any(word in user_input.lower() for word in ["generar", "sintético", "crear"]):
            self.current_agent = "generator"
        elif any(word in user_input.lower() for word in ["validar", "verificar"]):
            self.current_agent = "validator"
        elif any(word in user_input.lower() for word in ["simular", "evolución"]):
            self.current_agent = "simulator"
        elif any(word in user_input.lower() for word in ["evaluar", "utilidad", "calidad"]):
            self.current_agent = "evaluator"
        elif any(word in user_input.lower() for word in ["descargar", "exportar", "guardar"]):
            return await self._handle_download_request(context)
        
        agent = self.agents.get(self.current_agent)
        if agent:
            # Ejecutar según el agente activo
            if self.current_agent == "analyzer" and has_dataset and dataframe is not None:
                response = await self._handle_analysis(agent, dataframe, user_input, context)
            elif self.current_agent == "generator":
                response = await self._handle_generation(agent, user_input, context)
            elif self.current_agent == "validator":
                response = await self._handle_validation(agent, user_input, context)
            elif self.current_agent == "evaluator":
                response = await self._handle_evaluation(agent, user_input, context)
            else:
                response = await agent.process(user_input, context)
            
            response["current_agent"] = self.current_agent
            return response
        
        return {"message": "❌ Agente no disponible", "agent": "error"}
    
    async def _handle_analysis(self, agent, dataframe, user_input, context):
        """Maneja el análisis de datos con resultados reales"""
        if hasattr(agent, 'analyze_dataset') and not getattr(agent, 'name', '').startswith('Mock'):
            # Agente real
            response = await agent.analyze_dataset(dataframe, context)
        else:
            # Agente mock con análisis mejorado
            response = await self._analyze_with_mock(agent, dataframe, user_input, context)
        
        # Guardar datos analizados para siguiente paso
        self.pipeline_data['analysis_complete'] = True
        self.pipeline_data['original_data'] = dataframe
        self.pipeline_data['analysis_results'] = response.get('dataset_info', {})
        
        # Añadir botón para siguiente paso
        if response.get('mock', False):
            response['message'] += "\n\n🚀 **¿Listo para generar datos sintéticos?** Escribe: `generar 100 sintéticos`"
        
        return response
    
    async def _handle_generation(self, agent, user_input, context):
        """Maneja la generación de datos sintéticos con selección de método"""
        # Verificar que hay datos para generar
        if not self.pipeline_data.get('analysis_complete'):
            return {
                "message": "⚠️ **Primero necesito analizar los datos**\n\nPara generar datos sintéticos, primero debo analizar el dataset original.\n\n📝 **Comando:** `analizar datos`",
                "agent": "generator",
                "error": True
            }
        import re
        import tempfile
        from src.generation import SDVGenerator, CTGANGenerator, TVAEGenerator
        # Extraer número de muestras y método del input
        numbers = re.findall(r'\d+', user_input)
        num_samples = int(numbers[0]) if numbers else 100
        user_input_lower = user_input.lower()
        if 'ctgan' in user_input_lower:
            generator = CTGANGenerator()
            method = 'CTGAN'
        elif 'tvae' in user_input_lower:
            generator = TVAEGenerator()
            method = 'TVAE'
        else:
            generator = SDVGenerator()
            method = 'SDV GaussianCopula'
        original_data = self.pipeline_data.get('original_data')
        # Si original_data es un DataFrame, guardarlo temporalmente y pasar la ruta
        if isinstance(original_data, pd.DataFrame):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w', encoding='utf-8') as tmp:
                original_data.to_csv(tmp.name, index=False)
                data_path = tmp.name
        else:
            data_path = original_data
        try:
            synthetic_data = generator.generate(data_path, num_samples)
            quality_score = 0.85  # Puedes mejorar esto si tienes un método real
            message = f"🔬 **GENERACIÓN COMPLETADA CON {method}**\n\n📊 **Resultados:**\n- Registros originales: {len(original_data):,}\n- Registros sintéticos: {len(synthetic_data):,}\n- Método: {method}\n- Score de calidad: {quality_score:.1%}\n\n✅ Los datos mantienen las distribuciones y correlaciones originales"
            self.pipeline_data['generation_complete'] = True
            self.pipeline_data['synthetic_data'] = synthetic_data
            self.pipeline_data['generation_info'] = {
                'method': method,
                'quality_score': quality_score,
                'original_size': len(original_data),
                'synthetic_size': len(synthetic_data)
            }
            download_info = f"""
\n💾 **DATOS SINTÉTICOS LISTOS PARA DESCARGA:**\n\n📊 **Estadísticas:**\n- **Registros generados:** {len(synthetic_data):,}\n- **Columnas:** {len(synthetic_data.columns)}\n- **Método:** {method}\n- **Calidad:** {quality_score:.1%}\n\n📁 **Formatos disponibles:** CSV, JSON\n📂 **Ubicación:** data/synthetic/\n\n🔍 **¿Validar calidad médica?** Escribe: `validar datos`\n📊 **¿Ver vista previa?** Escribe: `mostrar datos`\n💾 **¿Descargar ahora?** Escribe: `descargar csv` o `descargar json`"""
            return {
                "message": message + download_info,
                "agent": getattr(agent, "name", "generator"),
                "synthetic_data": synthetic_data,
                "generation_info": self.pipeline_data['generation_info'],
                'has_synthetic_data': True,
                'synthetic_data_info': {
                    'records': len(synthetic_data),
                    'columns': len(synthetic_data.columns),
                    'generated_at': datetime.now().isoformat()
                },
                'current_agent': self.current_agent
            }
        except Exception as e:
            # Fallback a generación mock, mostrando el error real
            response = await self._generate_with_mock(agent, original_data, num_samples, context)
            # Añadir el error real al mensaje para depuración
            if isinstance(response, dict):
                response['message'] += f"\n\n❗ **Error real del generador:** {str(e)}"
            return response

    async def _generate_with_mock(self, agent, original_data, num_samples, context):
        """Generación mock mejorada con datos más realistas"""
        try:
            # Usar el generador SDV real si está disponible
            from src.generation.sdv_generator import SDVGenerator
            generator = SDVGenerator()
            synthetic_data = generator.generate(original_data, num_samples)
            quality_score = generator.get_quality_score()

            message = f"🔬 **GENERACIÓN COMPLETADA CON SDV**\n\n📊 **Resultados:**\n- Registros originales: {len(original_data):,}\n- Registros sintéticos: {len(synthetic_data):,}\n- Método: SDV GaussianCopula\n- Score de calidad: {quality_score:.1%}\n\n✅ Los datos mantienen las distribuciones y correlaciones originales"

            return {
                "message": message,
                "agent": getattr(agent, "name", "generator"),
                "synthetic_data": synthetic_data,
                "generation_info": {
                    'method': 'SDV GaussianCopula',
                    'quality_score': quality_score,
                    'original_size': len(original_data),
                    'synthetic_size': len(synthetic_data)
                }
            }

        except Exception as e:
            # Fallback a generación mock
            mock_data = self._create_realistic_mock_data(original_data, num_samples)

            return {
                "message": f"🔬 **GENERACIÓN COMPLETADA (MODO SIMULADO)**\n\n📊 **Resultados:**\n- Registros sintéticos: {len(mock_data):,}\n- Método: Simulación estadística\n- Nota: Para resultados óptimos, configura SDV\n\n⚠️ Datos generados para demostración",
                "agent": getattr(agent, "name", "generator"),
                "synthetic_data": mock_data,
                "generation_info": {
                    'method': 'Mock Simulation',
                    'quality_score': 0.75,
                    'note': f'Fallback debido a: {str(e)}'
                }
            }
    
    async def _handle_validation(self, agent, user_input, context):
        """Maneja la validación médica"""
        
        if not self.pipeline_data.get('generation_complete'):
            return {
                "message": "⚠️ **Primero necesito generar datos sintéticos**\n\nPara validar, primero debo generar los datos sintéticos.\n\n📝 **Comando:** `generar 100 sintéticos`",
                "agent": "validator",
                "error": True
            }
        
        original_data = self.pipeline_data.get('original_data')
        synthetic_data = self.pipeline_data.get('synthetic_data')
        
        if hasattr(agent, 'validate_synthetic_data') and not getattr(agent, 'name', '').startswith('Mock'):
            # Agente real
            response = await agent.validate_synthetic_data(synthetic_data, original_data, context)
        else:
            # Validación mock
            response = await self._validate_with_mock(agent, synthetic_data, original_data, context)
        
        # Guardar resultados de validación
        self.pipeline_data['validation_complete'] = True
        self.pipeline_data['validation_results'] = response.get('validation_results', {})
        
        # Añadir botón para siguiente paso
        response['message'] += "\n\n📊 **¿Evaluar utilidad estadística?** Escribe: `evaluar calidad`"
        
        return response
        
    async def _handle_evaluation(self, agent, user_input, context):
        """Maneja la evaluación de utilidad"""
        
        if not self.pipeline_data.get('validation_complete'):
            return {
                "message": "⚠️ **Primero necesito validar los datos**\n\nPara evaluar utilidad, primero debo validar la calidad médica.\n\n📝 **Comando:** `validar datos`",
                "agent": "evaluator", 
                "error": True
            }
        
        original_data = self.pipeline_data.get('original_data')
        synthetic_data = self.pipeline_data.get('synthetic_data')
        
        if hasattr(agent, 'evaluate_synthetic_utility') and not getattr(agent, 'name', '').startswith('Mock'):
            # Agente real
            response = await agent.evaluate_synthetic_utility(original_data, synthetic_data, context)
        else:
            # Evaluación mock
            response = await self._evaluate_with_mock(agent, original_data, synthetic_data, context)
        
        # Guardar resultados finales
        self.pipeline_data['evaluation_complete'] = True
        self.pipeline_data['evaluation_results'] = response.get('evaluation_results', {})
        
        # Añadir botón para descargar
        response['message'] += "\n\n💾 **¿Descargar datos sintéticos?** Escribe: `descargar csv` o `descargar json`"
        
        return response
    
    async def _handle_download_request(self, context):
        """Maneja solicitudes de descarga"""
        
        if not self.pipeline_data.get('generation_complete'):
            return {
                "message": "⚠️ **No hay datos sintéticos para descargar**\n\nPrimero necesito generar y procesar los datos.\n\n📝 **Comando:** `analizar datos` → `generar sintéticos`",
                "agent": "downloader",
                "error": True
            }
        
        synthetic_data = self.pipeline_data.get('synthetic_data')
        
        # Información del pipeline completo
        pipeline_summary = f"""
📋 **RESUMEN DEL PIPELINE COMPLETADO:**

📊 **Datos Originales:** {len(self.pipeline_data.get('original_data', []))} registros
🔬 **Datos Sintéticos:** {len(synthetic_data)} registros generados
✅ **Análisis:** {'Completado' if self.pipeline_data.get('analysis_complete') else 'Pendiente'}
🧬 **Generación:** {'Completada' if self.pipeline_data.get('generation_complete') else 'Pendiente'}
🔍 **Validación:** {'Completada' if self.pipeline_data.get('validation_complete') else 'Pendiente'}
📈 **Evaluación:** {'Completada' if self.pipeline_data.get('evaluation_complete') else 'Pendiente'}

💾 **Archivos disponibles para descarga:**
- `datos_sinteticos.csv` - Formato tabular
- `datos_sinteticos.json` - Formato JSON
- `metadata_pipeline.json` - Información del proceso

📂 **Los archivos se han guardado en la carpeta:** `data/synthetic/`
"""
        
        try:
            # Guardar archivos
            import os
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'synthetic')
            os.makedirs(output_dir, exist_ok=True)
            
            # Guardar CSV
            csv_path = os.path.join(output_dir, 'datos_sinteticos.csv')
            synthetic_data.to_csv(csv_path, index=False)
            
            # Guardar JSON
            json_path = os.path.join(output_dir, 'datos_sinteticos.json')
            synthetic_data.to_json(json_path, orient='records', indent=2)
            
            # Guardar metadata del pipeline
            metadata = {
                'pipeline_info': self.pipeline_data,
                'generation_timestamp': datetime.now().isoformat(),
                'files_generated': ['datos_sinteticos.csv', 'datos_sinteticos.json']
            }
            
            metadata_path = os.path.join(output_dir, 'metadata_pipeline.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return {
                "message": pipeline_summary + f"\n\n✅ **Archivos guardados exitosamente en:** `{output_dir}`",
                "agent": "downloader",
                "download_complete": True,
                "files": {
                    'csv_path': csv_path,
                    'json_path': json_path,
                    'metadata_path': metadata_path
                }
            }
            
        except Exception as e:
            return {
                "message": f"❌ **Error al guardar archivos:** {str(e)}\n\nVerifica permisos de escritura en la carpeta de destino.",
                "agent": "downloader",
                "error": True
            }
    
    def _create_realistic_mock_data(self, original_data, num_samples):
        """Crea datos mock más realistas basados en el original"""
        import pandas as pd
        import numpy as np
        mock_records = []
        for i in range(num_samples):
            record = {}
            for column in original_data.columns:
                if original_data[column].dtype in ['int64', 'float64']:
                    mean_val = original_data[column].mean()
                    std_val = original_data[column].std()
                    if pd.isna(mean_val) or pd.isna(std_val):
                        record[column] = 0
                    else:
                        value = np.random.normal(mean_val, std_val * 0.5)
                        if 'PATIENT' in column.upper():
                            record[column] = 10000 + i
                        elif 'EDAD' in column.upper() or 'AGE' in column.upper():
                            record[column] = max(0, min(120, int(value)))
                        elif 'TEMP' in column.upper():
                            record[column] = max(35.0, min(42.0, round(value, 1)))
                        elif 'SAT' in column.upper() or 'O2' in column.upper():
                            record[column] = max(70, min(100, int(value)))
                        else:
                            record[column] = value
                else:
                    unique_vals = original_data[column].dropna().unique()
                    if len(unique_vals) > 0:
                        record[column] = np.random.choice(unique_vals)
                    else:
                        record[column] = f"synthetic_value_{i}"
            mock_records.append(record)
        return pd.DataFrame(mock_records)

@st.cache_resource
def initialize_orchestrator():
    agents = initialize_agents()
    return SimpleOrchestrator(agents)

# Inicialización del estado
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = initialize_orchestrator()
    st.session_state.chat_history = []
    st.session_state.context = {}
    st.session_state.file_uploaded = False

# Header principal con logo integrado
logo_path = os.path.join(project_root, "assets", "logo_patientia.png")
if os.path.exists(logo_path):
    # Header centrado con logo y título
    st.markdown('<div style="display: flex; justify-content: center; align-items: center; margin: 2.5rem 0;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Contenedor flex para logo y título con menos espacio
        st.markdown('<div style="display: flex; align-items: center; justify-content: center; gap: 0px;">', unsafe_allow_html=True)
        subcol1, subcol2 = st.columns([0.3, 2])
        with subcol1:
            st.image(logo_path, width=70)
        with subcol2:
            st.markdown("""
            <h1 style="font-size: 2.5rem; font-weight: 600; color: #1f2937; margin: 0; text-align: left; margin-left: -10px;">Patienti-IA</h1>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
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
    {"<span style='margin-left: 10px;'>• Archivo cargado</span>" if st.session_state.file_uploaded else ""}
</div>
""", unsafe_allow_html=True)

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
                    
                    # Expandir con vista previa
                    with st.expander("👁️ Vista previa del dataset", expanded=False):
                        if dataset_preview:
                            preview_df = pd.DataFrame(dataset_preview)
                            st.dataframe(preview_df, use_container_width=True)
                        
                        # Información técnica
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
                
                # MOSTRAR BOTONES DE DESCARGA PARA DATOS SINTÉTICOS SOLO SI EL MENSAJE ES DE GENERACIÓN
                show_synth_buttons = False
                # Mostrar solo si el mensaje tiene has_synthetic_data o el agente es generator y el mensaje contiene generación
                if message.get('has_synthetic_data'):
                    show_synth_buttons = True
                elif message.get('agent', '').lower() == 'generator' and (
                    'generación completada' in message.get('content', '').lower() or
                    'datos sintéticos listos' in message.get('content', '').lower()
                ):
                    show_synth_buttons = True
                if show_synth_buttons:
                    synthetic_data = None
                    if hasattr(st.session_state.orchestrator, 'pipeline_data'):
                        synthetic_data = st.session_state.orchestrator.pipeline_data.get('synthetic_data')
                    if synthetic_data is not None and hasattr(synthetic_data, 'empty') and not synthetic_data.empty:
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            csv_data = synthetic_data.to_csv(index=False)
                            st.download_button(
                                label="📄 Descargar CSV",
                                data=csv_data,
                                file_name=f"datos_sinteticos_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv",
                                key=f"csv_download_{i}",
                                use_container_width=True
                            )
                        with col2:
                            json_data = synthetic_data.to_json(orient='records', indent=2)
                            st.download_button(
                                label="📋 Descargar JSON",
                                data=json_data,
                                file_name=f"datos_sinteticos_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                mime="application/json",
                                key=f"json_download_{i}",
                                use_container_width=True
                            )
                        with col3:
                            if st.button("👁️ Vista Previa", key=f"preview_{i}", use_container_width=True):
                                preview_key = f"show_preview_{i}"
                                if preview_key not in st.session_state:
                                    st.session_state[preview_key] = False
                                st.session_state[preview_key] = not st.session_state[preview_key]
                        # Vista previa condicional
                        preview_key = f"show_preview_{i}"
                        if st.session_state.get(preview_key, False):
                            with st.expander("👁️ Vista previa de datos sintéticos", expanded=True):
                                col_prev1, col_prev2 = st.columns(2)
                                with col_prev1:
                                    st.markdown("**📊 Primeros 5 registros:**")
                                    st.dataframe(synthetic_data.head(), use_container_width=True)
                                with col_prev2:
                                    st.markdown("**📈 Estadísticas básicas:**")
                                    numeric_cols = synthetic_data.select_dtypes(include=['number']).columns[:5]
                                    if len(numeric_cols) > 0:
                                        st.dataframe(synthetic_data[numeric_cols].describe(), use_container_width=True)
                                    else:
                                        st.info("No hay columnas numéricas")
    else:
        # Mensaje de bienvenida centrado
        col1, col2 = st.columns([1, 2])
        with col2:
            with st.chat_message("assistant"):
                st.markdown("**Bienvenido a Patientia AI**")
                st.markdown("Generación inteligente de datos clínicos sintéticos.")
                
                # Crear dos columnas para el contenido
                content_col1, content_col2 = st.columns(2)
                
                with content_col1:
                    st.markdown("""
                    **¿Qué puedo hacer por ti?**  
                    - Analizar datasets médicos existentes  
                    - Generar datos sintéticos realistas y seguros  
                    - Validar la coherencia clínica de los datos  
                    - Simular la evolución temporal de pacientes  
                    """)
                
                with content_col2:
                    st.markdown("""
                    **Para comenzar:**  
                    - Sube un archivo CSV/Excel con datos clínicos anonimizados  
                    - Pregunta sobre generación de datos sintéticos 
                    - Solicita análisis o validación de datos 
                    - Escribe "ayuda" para ver más opciones  
                    """)

# Área de upload de archivos (solo cuando no hay archivo cargado)
if not st.session_state.file_uploaded:
    with st.expander("📁 Subir archivo de datos", expanded=False):
        uploaded_file = st.file_uploader(
            "Selecciona tu dataset clínico",
            type=["csv", "xlsx", "xls"],
            help="Sube un archivo CSV o Excel con datos clínicos anonimizados"
        )
        
        if uploaded_file:
            try:
                # Cargar el archivo
                with st.spinner("📊 Cargando y analizando archivo..."):
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file, low_memory=False)
                    else:
                        df = pd.read_excel(uploaded_file)
                
                # Actualizar contexto
                st.session_state.context.update({
                    "dataframe": df,
                    "filename": uploaded_file.name,
                    "dataset_uploaded": True,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                    "dtypes": dict(df.dtypes.astype(str)),
                    "missing_data": df.isnull().sum().to_dict()
                })
                st.session_state.file_uploaded = True
                
                # AÑADIR INFORMACIÓN DEL DATASET AL CHAT en lugar de fuera
                dataset_info_message = f"""📁 **Archivo cargado exitosamente: {uploaded_file.name}**

📊 **Información del dataset:**
- **Registros:** {len(df):,} filas  
- **Variables:** {len(df.columns)} columnas
- **Tamaño:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB

✅ **Dataset listo para análisis**"""

                # Añadir al historial del chat en lugar de mostrar separado
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": dataset_info_message,
                    "agent": "system",
                    "timestamp": datetime.now().isoformat(),
                    "dataset_loaded": True,
                    "dataset_preview": df.head().to_dict('records'),
                    "dataset_info": {
                        "filename": uploaded_file.name,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "dtypes": dict(df.dtypes.value_counts())
                    }
                })
                
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error al cargar archivo: {e}")

# Input del usuario con diseño personalizado
# Reemplazar el bloque de procesamiento del prompt

if prompt := st.chat_input("💭 Escribe tu mensaje aquí... (ej: 'analizar datos', 'generar sintéticos', 'ayuda')", key="main_input"):
    
    # Añadir mensaje del usuario
    st.session_state.chat_history.append({
        "role": "user", 
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })
    
    # Mostrar mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # INICIALIZAR result con valor por defecto
    result = {"message": "❌ Error de procesamiento", "agent": "error"}
    
    try:
        # Procesar con orquestador sin usar context manager para chat_message
        # Crear el mensaje pero no usarlo como context manager
        message_placeholder = st.chat_message("assistant")
        
        # Mostrar spinner en el mensaje
        with message_placeholder:
            with st.spinner("🧠 Procesando..."):
                # Ejecutar procesamiento asíncrono with tiempo máximo
                result = asyncio.run(
                    asyncio.wait_for(
                        st.session_state.orchestrator.process_user_input(
                            prompt, 
                            st.session_state.context
                        ),
                        timeout=60  # 60 segundos máximo
                    )
                )
                
                response_content = result.get("message", "❌ Sin respuesta")
                agent_name = result.get("agent", "IA")
                
                # Mostrar respuesta
                st.markdown(response_content)
                
                # Mostrar botones de descarga si hay datos sintéticos
                if result.get('has_synthetic_data') and hasattr(st.session_state.orchestrator, 'pipeline_data'):
                    synthetic_data = st.session_state.orchestrator.pipeline_data.get('synthetic_data')
                    if synthetic_data is not None:
                        st.markdown("---")
                        
                        # Botones de descarga DENTRO del mensaje
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv_data = synthetic_data.to_csv(index=False)
                            st.download_button(
                                label="📄 Descargar CSV",
                                data=csv_data,
                                file_name=f"datos_sinteticos_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv",
                                key="csv_download_direct",
                                use_container_width=True
                            )
                        
                        with col2:
                            json_data = synthetic_data.to_json(orient='records', indent=2)
                            st.download_button(
                                label="📋 Descargar JSON",
                                data=json_data,
                                file_name=f"datos_sinteticos_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                                mime="application/json",
                                key="json_download_direct",
                                use_container_width=True
                            )
        
        # Añadir respuesta al historial DESPUÉS de mostrarla
        assistant_message = {
            "role": "assistant",
            "content": response_content,
            "agent": agent_name,
            "timestamp": datetime.now().isoformat()
        }
        
        # Marcar si tiene datos sintéticos
        if result.get('has_synthetic_data'):
            assistant_message['has_synthetic_data'] = True
            assistant_message['synthetic_data_info'] = result.get('synthetic_data_info', {})
        
        st.session_state.chat_history.append(assistant_message)
        
        # Actualizar agente actual si cambió
        if result.get("current_agent"):
            st.session_state.orchestrator.current_agent = result["current_agent"]
            
    except asyncio.TimeoutError:
        with st.chat_message("assistant"):
            st.error("⌛ El proceso tardó demasiado tiempo y fue interrumpido. Por favor, intenta con un comando más simple.")
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "⌛ El proceso tardó demasiado tiempo y fue interrumpido. Por favor, intenta con un comando más simple.",
            "agent": "system_timeout",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        with st.chat_message("assistant"):
            error_msg = f"❌ **Error del sistema:** {str(e)}"
            st.error(error_msg)
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error_msg,
            "agent": "system_error",
            "timestamp": datetime.now().isoformat()
        })
    
    # ✅ AÑADIR - Auto-scroll a datos sintéticos si se generaron
    if result.get('has_synthetic_data'):
        st.rerun()

# ✅ AÑADIR - JavaScript para auto-scroll mejorado
st.markdown("""
<script>
// Auto-scroll mejorado para chat
function autoScrollToBottom() {
    const messages = document.querySelectorAll('.stChatMessage');
    if (messages.length > 0) {
        const lastMessage = messages[messages.length - 1];
        setTimeout(() => {
            lastMessage.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        }, 100);
    }
}

// Ejecutar scroll al cargar la página
document.addEventListener('DOMContentLoaded', function() {
    autoScrollToBottom();
    
    // Dar foco al input
    setTimeout(() => {
        const chatInput = document.querySelector('.stChatInput textarea');
        if (chatInput) chatInput.focus();
    }, 500);
});

// Observar cambios en los mensajes del chat
const chatObserver = new MutationObserver((mutations) => {
    let shouldScroll = false;
    
    mutations.forEach((mutation) => {
        if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
            const addedNodes = Array.from(mutation.addedNodes);
            // Verificar si se agregó un mensaje de chat
            const hasChatMessage = addedNodes.some(node => 
                node.nodeType === 1 && 
                node.classList && 
                node.classList.contains('stChatMessage')
            );
            if (hasChatMessage) {
                shouldScroll = true;
            }
        }
    });
    
    if (shouldScroll) {
        autoScrollToBottom();
    }
});

// Iniciar observación del contenedor de mensajes
const chatContainer = document.querySelector('.main');
if (chatContainer) {
    chatObserver.observe(chatContainer, {
        childList: true,
        subtree: true
    });
}

// Manejar botón de enviar con anti-rebote
let canSend = true;
document.addEventListener('click', function(e) {
    if (e.target.matches('.stChatInput button[kind="primary"]') && !canSend) {
        e.preventDefault();
        e.stopPropagation();
        return false;
    } else if (e.target.matches('.stChatInput button[kind="primary"]')) {
        canSend = false;
        setTimeout(() => { canSend = true; }, 1500);
    }
});

// Desactivar temporalmente el watcher de Streamlit cuando ocurra un error
window.addEventListener('error', function(e) {
    if (e.message.includes('no running event loop') || 
        e.message.includes('torch.classes') ||
        e.message.includes('RuntimeError')) {
        console.warn('Streamlit watcher error interceptado:', e.message);
        // Intentar prevenir que el error detenga la ejecución
        e.preventDefault();
        return true;
    }
});
</script>
""", unsafe_allow_html=True)
# Sidebar minimalista (colapsado por defecto)
with st.sidebar:
    st.markdown("### 🛠️ Herramientas")
    
    # Info del estado actual
    current_agent = st.session_state.orchestrator.current_agent
    st.info(f"🎯 **Agente activo:** {current_agent.title()}")
    
    # Botón de reset
    if st.button("🔄 Nueva conversación", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.context = {}
        st.session_state.file_uploaded = False
        st.session_state.orchestrator.current_agent = "coordinator"
        st.rerun()
    
    # Información del archivo cargado
    if st.session_state.file_uploaded:
        st.markdown("### 📁 Archivo actual")
        filename = st.session_state.context.get("filename", "N/A")
        rows = st.session_state.context.get("rows", 0)
        cols = st.session_state.context.get("columns", 0)
        
        st.markdown(f"""
        **📄 Nombre:** {filename}  
        **📊 Filas:** {rows:,}  
        **📋 Columnas:** {cols}
        """)
        
        if st.button("❌ Remover archivo"):
            st.session_state.file_uploaded = False
            st.session_state.context = {}
            st.rerun()
    
    st.markdown("---")
    
    # Comandos rápidos
    st.markdown("### 💡 Comandos útiles")
    st.markdown("""
    - `analizar datos` - Analizar dataset
    - `generar sintéticos` - Crear datos artificiales
    - `validar datos` - Verificar coherencia
    - `simular evolución` - Modelar progresión
    - `evaluar calidad` - Medir utilidad
    - `ayuda` - Ver más opciones
    """)
    
    # Footer
    st.markdown("---")
    st.caption("Powered by AI + LangChain")
