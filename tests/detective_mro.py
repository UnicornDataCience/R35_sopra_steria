"""
Test específico para detectar el error MRO exacto
"""
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

def test_base_agent():
    try:
        from agents.base_agent import BaseLLMAgent, BaseAgentConfig
        print("✅ base_agent importado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en base_agent: {e}")
        return False

def test_coordinator():
    try:
        from agents.coordinator_agent import CoordinatorAgent
        print("✅ coordinator_agent importado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en coordinator_agent: {e}")
        return False

def test_analyzer():
    try:
        from agents.analyzer_agent import ClinicalAnalyzerAgent
        print("✅ analyzer_agent importado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en analyzer_agent: {e}")
        return False

def test_generator():
    try:
        from agents.generator_agent import SyntheticGeneratorAgent
        print("✅ generator_agent importado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en generator_agent: {e}")
        return False

def test_validator():
    try:
        from agents.validator_agent import MedicalValidatorAgent
        print("✅ validator_agent importado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en validator_agent: {e}")
        return False

def test_simulator():
    try:
        from agents.simulator_agent import PatientSimulatorAgent
        print("✅ simulator_agent importado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en simulator_agent: {e}")
        return False

def test_evaluator():
    try:
        from agents.evaluator_agent import UtilityEvaluatorAgent
        print("✅ evaluator_agent importado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en evaluator_agent: {e}")
        return False

def test_orchestrator():
    try:
        from orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator
        print("✅ orchestrator importado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en orchestrator: {e}")
        return False

if __name__ == "__main__":
    print("=== DETECTIVE MRO ===")
    
    tests = [
        test_base_agent,
        test_coordinator,
        test_analyzer, 
        test_generator,
        test_validator,
        test_simulator,
        test_evaluator,
        test_orchestrator
    ]
    
    for test in tests:
        test()
        print()
