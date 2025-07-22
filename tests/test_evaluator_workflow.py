#!/usr/bin/env python3
"""
Test específico para verificar el workflow del evaluador
"""
import sys
import os
sys.path.append('.')

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

# FORZAR GROQ
if os.getenv('FORCE_GROQ', 'false').lower() == 'true':
    os.environ['LLM_PROVIDER'] = 'groq'
    print("🚀 [DEBUG] Variable LLM_PROVIDER forzada a 'groq'")

def test_evaluator_workflow():
    """Test del workflow del evaluador"""
    print("🧪 Testing Evaluator Workflow")
    print("=" * 50)
    
    try:
        # 1. Importar FastOrchestrator
        from src.orchestration.fast_orchestrator import FastMedicalOrchestrator
        print("✅ FastMedicalOrchestrator importado")
        
        # 2. Importar agentes
        from src.agents.coordinator_agent import CoordinatorAgent
        from src.agents.analyzer_agent import ClinicalAnalyzerAgent
        from src.agents.generator_agent import SyntheticGeneratorAgent
        from src.agents.evaluator_agent import UtilityEvaluatorAgent
        
        agents = {
            "coordinator": CoordinatorAgent(),
            "analyzer": ClinicalAnalyzerAgent(),
            "generator": SyntheticGeneratorAgent(),
            "evaluator": UtilityEvaluatorAgent()
        }
        print("✅ Todos los agentes creados")
        
        # 3. Crear orquestador
        orchestrator = FastMedicalOrchestrator(agents)
        print("✅ FastMedicalOrchestrator creado")
        
        # 4. Test detección de intención
        test_phrases = [
            "evaluar calidad",
            "evalúa los datos",
            "métricas de utilidad",
            "calidad de datos sintéticos"
        ]
        
        print("\n🔍 Testing intention detection:")
        for phrase in test_phrases:
            intention = orchestrator._detect_intention_fast(phrase, {})
            print(f"   '{phrase}' -> {intention}")
            if intention != "evaluator_workflow":
                print(f"❌ ERROR: Se esperaba 'evaluator_workflow', pero se obtuvo '{intention}'")
                return False
        
        print("✅ Detección de intención funciona correctamente")
        
        # 5. Test directo del workflow
        print("\n🔧 Testing evaluator workflow execution:")
        import asyncio
        
        # Crear contexto mock con datos requeridos
        import pandas as pd
        mock_context = {
            "dataframe": pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}),
            "synthetic_data": pd.DataFrame({"col1": [1.1, 2.1, 3.1], "col2": [4.1, 5.1, 6.1]})
        }
        
        async def test_workflow():
            response = await orchestrator._execute_agent_workflow(
                "evaluator_workflow", 
                "evaluar calidad de datos", 
                mock_context
            )
            return response
        
        # Ejecutar el test
        response = asyncio.run(test_workflow())
        print(f"✅ Workflow ejecutado. Respuesta tipo: {type(response)}")
        print(f"   Route: {response.get('route', 'N/A')}")
        print(f"   Agent: {response.get('agent', 'N/A')}")
        
        if response.get('route') == 'agent_workflow':
            print("✅ Evaluator workflow funciona correctamente")
            return True
        else:
            print(f"❌ ERROR: Se esperaba route='agent_workflow', pero se obtuvo '{response.get('route')}'")
            print(f"   Mensaje: {response.get('message', 'N/A')[:100]}...")
            return False
            
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluator_workflow()
    if success:
        print("\n🎉 TODOS LOS TESTS PASARON - Evaluator workflow está funcionando")
    else:
        print("\n❌ ALGUNOS TESTS FALLARON - Revisar configuración")
