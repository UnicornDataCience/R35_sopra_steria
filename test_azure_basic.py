import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Cargar variables de entorno
load_dotenv()

async def test_azure_openai_basic():
    """Test básico de conexión con Azure OpenAI"""
    
    print("🧪 **PRUEBA BÁSICA DE AZURE OPENAI**")
    print("=" * 50)
    
    # 1. Verificar variables de entorno
    print("\n🔧 **CONFIGURACIÓN:**")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4-TFM")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment}")
    print(f"API Version: {api_version}")
    print(f"API Key: {'✅ Configurada' if api_key else '❌ No encontrada'}")
    
    if not all([endpoint, api_key, deployment]):
        print("\n❌ **ERROR:** Configuración incompleta")
        print("Verifica tu archivo .env")
        return False
    
    try:
        print("\n🚀 **CONECTANDO...**")
        
        # Crear cliente Azure OpenAI
        llm = AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            temperature=0.1,
            max_tokens=1000
        )
        
        print("✅ Cliente creado exitosamente")
        
        # Test 1: Mensaje simple
        print("\n🔄 **TEST 1:** Mensaje simple...")
        response = await llm.ainvoke("Di 'Hola' en una frase simple")
        print(f"✅ Respuesta: {response.content}")
        
        # Test 2: Conversación médica básica
        print("\n🔄 **TEST 2:** Contexto médico...")
        messages = [
            SystemMessage(content="Eres un asistente médico especializado en datos sintéticos."),
            HumanMessage(content="¿Qué son los datos sintéticos en medicina?")
        ]
        response = await llm.ainvoke(messages)
        print(f"✅ Respuesta: {response.content[:200]}...")
        
        # Test 3: Generación de datos mock
        print("\n🔄 **TEST 3:** Generación de ejemplo...")
        mock_prompt = """Genera un ejemplo de paciente COVID-19 sintético con estos campos:
        - ID del paciente
        - Edad
        - Sexo
        - Diagnóstico
        - Medicamento
        - Días en UCI
        - Temperatura de ingreso
        - Saturación O2
        - Resultado
        - Motivo de alta
        
        Responde solo con el ejemplo, sin explicaciones."""
        
        response = await llm.ainvoke(mock_prompt)
        print(f"✅ Ejemplo generado:\n{response.content}")
        
        print("\n🎉 **TODAS LAS PRUEBAS EXITOSAS**")
        print("La conexión con Azure OpenAI está funcionando correctamente")
        return True
        
    except Exception as e:
        print(f"\n❌ **ERROR EN CONEXIÓN:** {e}")
        print("\n🔍 **POSIBLES CAUSAS:**")
        print("• API key incorrecta o expirada")
        print("• Endpoint incorrecto")
        print("• Deployment no disponible")
        print("• Problemas de red/firewall")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_azure_openai_basic())
    if result:
        print("\n✅ Tu configuración Azure OpenAI está lista para usar!")
    else:
        print("\n❌ Revisa tu configuración antes de continuar")