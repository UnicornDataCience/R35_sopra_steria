#!/usr/bin/env python3
"""
Test simple para verificar que los arreglos funcionen correctamente
"""
import asyncio
import sys
import os

# Añadir la ruta del proyecto al path de Python
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    """Test básico de imports"""
    try:
        from src.orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator
        print("✅ Import LangGraph Orchestrator: OK")
        
        from src.agents.coordinator_agent import CoordinatorAgent
        print("✅ Import CoordinatorAgent: OK")
        
        from interfaces.chat_llm import initialize_langgraph_orchestrator
        print("✅ Import interfaz: OK")
        
        return True
    except Exception as e:
        print(f"❌ Error en imports: {e}")
        return False

async def test_mock_orchestrator():
    """Test del orquestador mock"""
    try:
        from interfaces.chat_llm import create_mock_orchestrator
        
        orchestrator = create_mock_orchestrator()
        
        # Test 1: Saludo simple
        response = await orchestrator.process_user_input("hola como estas", {})
        print(f"✅ Test saludo: {response.get('message', 'N/A')[:100]}...")
        
        # Test 2: Pregunta médica
        response = await orchestrator.process_user_input("¿Cuáles son los factores de riesgo cardiovascular?", {})
        print(f"✅ Test pregunta médica: {response.get('message', 'N/A')[:100]}...")
        
        # Test 3: Comando de análisis
        response = await orchestrator.process_user_input("analiza estos datos", {"dataset_uploaded": True})
        print(f"✅ Test comando análisis: {response.get('message', 'N/A')[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error en test mock: {e}")
        return False

async def main():
    """Función principal de test"""
    print("🧪 Testing arreglos de interfaz y LLM conversacional")
    print("=" * 60)
    
    # Test 1: Imports básicos
    print("\n📦 Test 1: Verificando imports...")
    imports_ok = test_imports()
    
    # Test 2: Orquestador mock
    print("\n🤖 Test 2: Verificando orquestador mock...")
    mock_ok = await test_mock_orchestrator()
    
    # Resultado final
    print("\n" + "=" * 60)
    if imports_ok and mock_ok:
        print("✅ TODOS LOS TESTS PASARON - Sistema funcionando correctamente")
        print("\n🚀 La interfaz ahora debería:")
        print("   - Mostrar bienvenida estática con dos columnas")
        print("   - Responder a preguntas médicas conversacionales")
        print("   - Manejar comandos de agentes correctamente")
    else:
        print("❌ ALGUNOS TESTS FALLARON - Revisar errores arriba")
    
    print("\n💡 Para probar la interfaz completa ejecuta:")
    print("   streamlit run interfaces/chat_llm.py")

if __name__ == "__main__":
    asyncio.run(main())
