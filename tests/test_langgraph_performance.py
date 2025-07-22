#!/usr/bin/env python3
"""
Script de prueba para diagnosticar problemas de rendimiento en LangGraph Orchestrator
"""

import sys
import os
import time
import datetime
import asyncio

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Probar la importación de módulos"""
    print(f"🧪 [{datetime.datetime.now().strftime('%H:%M:%S')}] Probando imports...")
    start_time = time.time()
    
    try:
        from src.orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator
        from src.agents.coordinator_agent import CoordinatorAgent
        from src.agents.analyzer_agent import ClinicalAnalyzerAgent
        from src.agents.generator_agent import SyntheticGeneratorAgent
        from src.agents.validator_agent import MedicalValidatorAgent
        
        end_time = time.time()
        print(f"✅ [{datetime.datetime.now().strftime('%H:%M:%S')}] Imports completados en {end_time - start_time:.2f}s")
        return True
    except Exception as e:
        end_time = time.time()
        print(f"❌ [{datetime.datetime.now().strftime('%H:%M:%S')}] Error en imports después de {end_time - start_time:.2f}s: {e}")
        return False

def test_agent_initialization():
    """Probar la inicialización de agentes individuales"""
    print(f"🧪 [{datetime.datetime.now().strftime('%H:%M:%S')}] Probando inicialización de agentes...")
    start_time = time.time()
    
    try:
        from src.agents.coordinator_agent import CoordinatorAgent
        
        print(f"🧪 [{datetime.datetime.now().strftime('%H:%M:%S')}] Creando CoordinatorAgent...")
        coordinator = CoordinatorAgent()
        
        end_time = time.time()
        print(f"✅ [{datetime.datetime.now().strftime('%H:%M:%S')}] Agente creado en {end_time - start_time:.2f}s")
        return coordinator
    except Exception as e:
        end_time = time.time()
        print(f"❌ [{datetime.datetime.now().strftime('%H:%M:%S')}] Error creando agente después de {end_time - start_time:.2f}s: {e}")
        return None

def test_orchestrator_initialization():
    """Probar la inicialización del orquestador"""
    print(f"🧪 [{datetime.datetime.now().strftime('%H:%M:%S')}] Probando inicialización del orquestador...")
    start_time = time.time()
    
    try:
        from src.orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator
        from src.agents.coordinator_agent import CoordinatorAgent
        
        # Crear agentes mock mínimos
        agents = {
            "coordinator": CoordinatorAgent(),
            "analyzer": None,  # Mock básico
            "generator": None,
            "validator": None
        }
        
        print(f"🧪 [{datetime.datetime.now().strftime('%H:%M:%S')}] Creando MedicalAgentsOrchestrator...")
        orchestrator = MedicalAgentsOrchestrator(agents)
        
        end_time = time.time()
        print(f"✅ [{datetime.datetime.now().strftime('%H:%M:%S')}] Orquestador creado en {end_time - start_time:.2f}s")
        return orchestrator
    except Exception as e:
        end_time = time.time()
        print(f"❌ [{datetime.datetime.now().strftime('%H:%M:%S')}] Error creando orquestador después de {end_time - start_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_simple_workflow(orchestrator):
    """Probar un workflow simple (solo conversación)"""
    print(f"🧪 [{datetime.datetime.now().strftime('%H:%M:%S')}] Probando workflow simple...")
    start_time = time.time()
    
    try:
        # Test simple de conversación que debería terminar rápido
        response = await orchestrator.process_user_input("hola", {})
        
        end_time = time.time()
        print(f"✅ [{datetime.datetime.now().strftime('%H:%M:%S')}] Workflow simple completado en {end_time - start_time:.2f}s")
        print(f"📤 Respuesta: {response}")
        return True
    except Exception as e:
        end_time = time.time()
        print(f"❌ [{datetime.datetime.now().strftime('%H:%M:%S')}] Error en workflow después de {end_time - start_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sync_wrapper(orchestrator):
    """Probar el wrapper síncrono"""
    print(f"🧪 [{datetime.datetime.now().strftime('%H:%M:%S')}] Probando wrapper síncrono...")
    start_time = time.time()
    
    try:
        # Test con timeout corto para identificar el problema
        response = orchestrator.process_user_input_sync("hola", {})
        
        end_time = time.time()
        print(f"✅ [{datetime.datetime.now().strftime('%H:%M:%S')}] Wrapper síncrono completado en {end_time - start_time:.2f}s")
        print(f"📤 Respuesta: {response}")
        return True
    except Exception as e:
        end_time = time.time()
        print(f"❌ [{datetime.datetime.now().strftime('%H:%M:%S')}] Error en wrapper después de {end_time - start_time:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal de pruebas"""
    print(f"🚀 [{datetime.datetime.now().strftime('%H:%M:%S')}] Iniciando diagnóstico de rendimiento LangGraph...")
    
    # Test 1: Imports
    if not test_imports():
        return
    
    # Test 2: Inicialización de agente individual
    agent = test_agent_initialization()
    if not agent:
        return
    
    # Test 3: Inicialización del orquestador
    orchestrator = test_orchestrator_initialization()
    if not orchestrator:
        return
    
    # Test 4: Workflow simple async
    if not await test_simple_workflow(orchestrator):
        return
    
    # Test 5: Wrapper síncrono
    if not test_sync_wrapper(orchestrator):
        return
    
    print(f"🎉 [{datetime.datetime.now().strftime('%H:%M:%S')}] Todos los tests completados exitosamente!")

if __name__ == "__main__":
    asyncio.run(main())
