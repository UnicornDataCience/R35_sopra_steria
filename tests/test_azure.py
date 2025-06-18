from src.config.azure_config import azure_config
import asyncio

async def test_azure_connection():
    """Prueba la conexión a Azure OpenAI"""
    
    print("🔍 Probando configuración de Azure OpenAI...")
    print(f"📊 Estado: {azure_config.status_info}")
    
    try:
        # Crear LLM
        llm = azure_config.create_llm(temperature=0.1)
        
        # Prueba simple
        response = await llm.ainvoke("Di 'Hola' en una frase")
        
        print("✅ Conexión exitosa!")
        print(f"📝 Respuesta: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_azure_connection())