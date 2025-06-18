import streamlit as st
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Cargar variables de entorno
load_dotenv()

# Configuración de página
st.set_page_config(
    page_title="Chat Básico - Azure OpenAI",
    page_icon="💬",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .status-good { 
        color: #10b981; 
        font-weight: bold;
    }
    
    .status-error { 
        color: #ef4444; 
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Función para inicializar Azure OpenAI
@st.cache_resource
def initialize_azure_llm():
    """Inicializa el cliente Azure OpenAI"""
    try:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4-TFM")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        if not all([endpoint, api_key, deployment]):
            return None, "Configuración Azure OpenAI incompleta"
        
        llm = AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            temperature=0.3,
            max_tokens=2000
        )
        
        return llm, "Conexión exitosa"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Función asíncrona para generar respuesta
async def generate_response(llm, messages):
    """Genera respuesta usando Azure OpenAI"""
    try:
        response = await llm.ainvoke(messages)
        return response.content, None
    except Exception as e:
        return None, str(e)

# Función para test de conexión
async def test_connection(llm):
    """Prueba la conexión con Azure OpenAI"""
    try:
        response = await llm.ainvoke("Di 'Test exitoso'")
        return response.content, None
    except Exception as e:
        return None, str(e)

# Inicializar estado
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'llm' not in st.session_state:
    llm, status = initialize_azure_llm()
    st.session_state.llm = llm
    st.session_state.connection_status = status

# Header
st.markdown("""
<div class="main-header">
    <h1>💬 Chat Básico con Azure OpenAI</h1>
    <p>Prueba directa de conexión sin agentes complejos</p>
</div>
""", unsafe_allow_html=True)

# Sidebar con información
with st.sidebar:
    st.title("🔧 Estado del Sistema")
    
    # Estado de conexión
    if st.session_state.llm:
        st.markdown('<p class="status-good">✅ Azure OpenAI Conectado</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-error">❌ Azure OpenAI No Disponible</p>', unsafe_allow_html=True)
        st.error(st.session_state.connection_status)
    
    # Información de configuración
    st.subheader("📋 Configuración")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "No configurado")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "No configurado")
    
    st.info(f"""
    **Endpoint:** {endpoint[:50]}...
    **Deployment:** {deployment}
    **Mensajes:** {len(st.session_state.chat_history)}
    """)
    
    # Botón de test
    if st.button("🧪 Test de Conexión", use_container_width=True):
        if st.session_state.llm:
            with st.spinner("Probando conexión..."):
                # Usar asyncio.run() para ejecutar función asíncrona
                result, error = asyncio.run(test_connection(st.session_state.llm))
                
                if result:
                    st.success(f"✅ Test OK: {result}")
                else:
                    st.error(f"❌ Test falló: {error}")
        else:
            st.error("No hay conexión para probar")
    
    # Reset
    if st.button("🔄 Reiniciar Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Área principal de chat
st.subheader("💬 Conversación")

# Mostrar mensajes
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("timestamp"):
            st.caption(f"🕒 {message['timestamp']}")

# Input del usuario
if prompt := st.chat_input("Escribe tu mensaje..."):
    
    # Verificar conexión
    if not st.session_state.llm:
        st.error("❌ No hay conexión con Azure OpenAI. Verifica tu configuración.")
        st.stop()
    
    # Añadir mensaje del usuario
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.chat_history.append(user_message)
    
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.write(prompt)
        st.caption(f"🕒 {user_message['timestamp']}")
    
    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("🧠 Generando respuesta..."):
            
            # Crear contexto de conversación
            messages = [
                SystemMessage(content="""Eres un asistente médico experto en datos sintéticos y análisis clínico. 

Características:
- Especializado en COVID-19 y datos hospitalarios
- Conocimiento de SDV, CTGAN y generación sintética
- Comunicación clara y profesional
- Respuestas concisas pero informativas

Responde de manera útil y orientada a la acción.""")
            ]
            
            # Añadir historial reciente (últimos 5 mensajes de usuario)
            recent_history = st.session_state.chat_history[-6:-1]  # Excluir el último (que acabamos de añadir)
            for msg in recent_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
            
            # Añadir mensaje actual
            messages.append(HumanMessage(content=prompt))
            
            # Generar respuesta usando asyncio.run()
            response_content, error = asyncio.run(
                generate_response(st.session_state.llm, messages)
            )
            
            if response_content:
                # Mostrar respuesta
                st.write(response_content)
                
                # Añadir respuesta al historial
                assistant_message = {
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                st.session_state.chat_history.append(assistant_message)
                
                st.caption(f"🕒 {assistant_message['timestamp']}")
                
            else:
                # Manejar error
                error_msg = f"❌ Error generando respuesta: {error}"
                st.error(error_msg)
                
                # Añadir error al historial
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })

# Footer
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.caption("🧠 **Azure OpenAI** - Test de conexión básica")

with col2:
    st.caption("💡 **Tip:** Prueba preguntas sobre datos médicos sintéticos")

# Mensajes de ayuda
if not st.session_state.chat_history:
    st.info("""
    **💡 Sugerencias para probar:**
    
    - `"¿Qué son los datos sintéticos médicos?"`
    - `"Explícame SDV y CTGAN"`
    - `"Genera un ejemplo de paciente COVID-19"`
    - `"¿Cómo validar datos sintéticos?"`
    
    **🔧 Si hay errores de conexión:**
    1. Verifica tu archivo .env
    2. Confirma que tu API key es válida
    3. Usa el botón "Test de Conexión" en la barra lateral
    """)