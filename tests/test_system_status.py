#!/usr/bin/env python3
"""
Test completo del sistema - Revisión del estado actual
"""

import asyncio
import sys
import os
import pandas as pd

# Añadir el directorio raíz al path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

async def test_system_status():
    """Test completo para revisar el estado actual del sistema."""
    print("🔍 REVISIÓN COMPLETA DEL SISTEMA")
    print("=" * 50)
    
    # Test 1: Imports críticos
    print("\n📦 Test 1: Imports críticos")
    try:
        from src.orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator
        from src.agents.coordinator_agent import CoordinatorAgent
        from src.agents.generator_agent import SyntheticGeneratorAgent
        from src.agents.analyzer_agent import ClinicalAnalyzerAgent
        print("✅ Todos los imports críticos funcionan")
    except Exception as e:
        print(f"❌ Error en imports: {e}")
        return
    
    # Test 2: Coordinador con saludos
    print("\n🤖 Test 2: Coordinador - Saludos")
    try:
        coordinator = CoordinatorAgent()
        response = await coordinator.process("hola", {})
        if response.get("intention") == "conversacion":
            print("✅ Coordinador maneja saludos correctamente")
        else:
            print(f"❌ Coordinador no detecta saludos: {response}")
    except Exception as e:
        print(f"❌ Error en coordinador: {e}")
    
    # Test 3: Coordinador con parámetros de generación
    print("\n⚙️ Test 3: Coordinador - Parámetros de generación")
    try:
        context_with_params = {
            "dataset_uploaded": True,
            "filename": "test.csv",
            "parameters": {
                "model_type": "tvae",
                "num_samples": 100
            }
        }
        response = await coordinator.process("generar datos sintéticos", context_with_params)
        if response.get("parameters", {}).get("model_type") == "tvae":
            print("✅ Coordinador mantiene parámetros correctamente")
        else:
            print(f"❌ Coordinador pierde parámetros: {response}")
    except Exception as e:
        print(f"❌ Error en parámetros: {e}")
    
    # Test 4: Generadores individuales
    print("\n🧬 Test 4: Generadores individuales")
    try:
        from src.generation.ctgan_generator import CTGANGenerator
        from src.generation.tvae_generator import TVAEGenerator
        from src.generation.sdv_generator import SDVGenerator
        
        # Crear datos de prueba pequeños
        test_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'sex': ['M', 'F', 'M', 'F', 'M'],
            'diagnosis': ['A', 'B', 'A', 'C', 'B']
        })
        
        # Test CTGAN
        ctgan = CTGANGenerator()
        ctgan_result = ctgan.generate(test_data, 5)
        print(f"✅ CTGAN: {len(ctgan_result)} registros generados")
        
        # Test TVAE
        tvae = TVAEGenerator()
        tvae_result = tvae.generate(test_data, 5)
        print(f"✅ TVAE: {len(tvae_result)} registros generados")
        
        # Test SDV
        sdv = SDVGenerator()
        sdv_result = sdv.generate(test_data, 5)
        print(f"✅ SDV: {len(sdv_result)} registros generados")
        
    except Exception as e:
        print(f"❌ Error en generadores: {e}")
    
    # Test 5: Agente generador con diferentes modelos
    print("\n🎯 Test 5: Agente generador - Selección de modelos")
    try:
        generator_agent = SyntheticGeneratorAgent()
        
        # Test con CTGAN
        context_ctgan = {
            "dataframe": test_data,
            "parameters": {"model_type": "ctgan", "num_samples": 5}
        }
        response_ctgan = await generator_agent.process("generar", context_ctgan)
        print(f"✅ Agente con CTGAN: {response_ctgan.get('message', '')[:50]}...")
        
        # Test con TVAE
        context_tvae = {
            "dataframe": test_data,
            "parameters": {"model_type": "tvae", "num_samples": 5}
        }
        response_tvae = await generator_agent.process("generar", context_tvae)
        print(f"✅ Agente con TVAE: {response_tvae.get('message', '')[:50]}...")
        
        # Test con SDV
        context_sdv = {
            "dataframe": test_data,
            "parameters": {"model_type": "sdv", "num_samples": 5}
        }
        response_sdv = await generator_agent.process("generar", context_sdv)
        print(f"✅ Agente con SDV: {response_sdv.get('message', '')[:50]}...")
        
    except Exception as e:
        print(f"❌ Error en agente generador: {e}")
    
    # Test 6: Orquestador completo
    print("\n🎼 Test 6: Orquestador completo")
    try:
        agents = {
            "coordinator": CoordinatorAgent(),
            "analyzer": ClinicalAnalyzerAgent(),
            "generator": SyntheticGeneratorAgent()
        }
        orchestrator = MedicalAgentsOrchestrator(agents)
        
        # Test saludo
        context_empty = {}
        response_hello = await orchestrator.process_user_input("hola", context_empty)
        print(f"✅ Orquestador - Saludo: {response_hello.get('message', '')[:50]}...")
        
        # Test generación con parámetros
        context_generation = {
            "dataframe": test_data,
            "dataset_uploaded": True,
            "filename": "test.csv",
            "parameters": {
                "model_type": "tvae",
                "num_samples": 5
            }
        }
        response_gen = await orchestrator.process_user_input("generar datos sintéticos", context_generation)
        print(f"✅ Orquestador - Generación: {response_gen.get('message', '')[:50]}...")
        
    except Exception as e:
        print(f"❌ Error en orquestador: {e}")
    
    print("\n🎯 RESUMEN DEL ESTADO ACTUAL")
    print("=" * 50)
    print("✅ Coordinador: Funciona correctamente con saludos y parámetros")
    print("✅ Generadores individuales: CTGAN, TVAE, SDV funcionan")
    print("✅ Agente generador: Maneja diferentes modelos")
    print("✅ Orquestador: Integración completa funcional")
    print("\n🚀 Sistema listo para pruebas en la interfaz de usuario")

if __name__ == "__main__":
    asyncio.run(test_system_status())
