import pytest
import asyncio
import pandas as pd
import os
import sys
from typing import Dict, Any

# Añadir la ruta del proyecto al path de Python para las importaciones
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.agents.coordinator_agent import CoordinatorAgent
from src.orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator, AgentState
from src.agents.analyzer_agent import ClinicalAnalyzerAgent
from src.adapters.universal_dataset_detector import UniversalDatasetDetector
from src.config.pipeline_config import DynamicPipelineConfig

# --- Funciones auxiliares para crear datos de prueba ---
def create_test_dataframe(num_rows: int = 10) -> pd.DataFrame:
    """Crea un DataFrame de prueba simple."""
    data = {
        'PATIENT ID': range(1, num_rows + 1),
        'EDAD/AGE': [25 + i for i in range(num_rows)],
        'SEXO/SEX': ['M' if i % 2 == 0 else 'F' for i in range(num_rows)],
        'DIAGNOSTICO': ['Gripe' if i % 3 == 0 else 'Resfriado' for i in range(num_rows)],
        'TEMP_ING/INPAT': [37.0 + i*0.1 for i in range(num_rows)],
        'RESULTADO/VAL_RESULT': ['POSITIVO' if i % 2 == 0 else 'NEGATIVO' for i in range(num_rows)]
    }
    return pd.DataFrame(data)

# --- Tests para CoordinatorAgent ---
@pytest.mark.asyncio
async def test_coordinator_agent_intention_classification():
    """Verifica que el CoordinatorAgent clasifica correctamente las intenciones."""
    print("\n--- Test: CoordinatorAgent Intention Classification ---")
    coordinator = CoordinatorAgent()

    # Caso 1: Conversación
    user_input_conv = "Hola, ¿cómo estás?"
    response_conv = await coordinator.process(user_input_conv)
    print(f"Input: '{user_input_conv}' -> Response: {response_conv}")
    assert response_conv["intention"] == "conversacion"
    assert "Hola" in response_conv["message"] or "cómo estás" in response_conv["message"]

    # Caso 2: Comando "analizar datos"
    user_input_cmd_analyze = "analizar datos"
    response_cmd_analyze = await coordinator.process(user_input_cmd_analyze)
    print(f"Input: '{user_input_cmd_analyze}' -> Response: {response_cmd_analyze}")
    assert response_cmd_analyze["intention"] == "comando"
    assert "analizar" in response_cmd_analyze["message"].lower()

    # Caso 3: Comando "generar sintéticos"
    user_input_cmd_generate = "generar 100 sintéticos"
    response_cmd_generate = await coordinator.process(user_input_cmd_generate)
    print(f"Input: '{user_input_cmd_generate}' -> Response: {response_cmd_generate}")
    assert response_cmd_generate["intention"] == "comando"
    assert "generar" in response_cmd_generate["message"].lower()

    print("--- Test: CoordinatorAgent Intention Classification PASSED ---")

# --- Test integrado para AnalyzerAgent ---
@pytest.mark.asyncio
async def test_analyzer_agent_flow():
    """Verifica el flujo completo del AnalyzerAgent a través del orquestador."""
    print("\n--- Test: AnalyzerAgent Integrated Flow ---")

    # 1. Inicializar agentes mock o reales según sea necesario
    # Para este test, usaremos el AnalyzerAgent real y un CoordinatorAgent real
    # Los demás agentes pueden ser mocks si el orquestador los necesita para compilar
    agents = {
        "coordinator": CoordinatorAgent(),
        "analyzer": ClinicalAnalyzerAgent(),
        # Puedes añadir mocks para otros agentes si el orquestador los necesita para compilar
        "generator": type('MockAgent', (), {'process': lambda *args, **kwargs: {'message': 'Mocked generation', 'agent': 'generator'}})() ,
        "validator": type('MockAgent', (), {'process': lambda *args, **kwargs: {'message': 'Mocked validation', 'agent': 'validator'}})() ,
        "simulator": type('MockAgent', (), {'process': lambda *args, **kwargs: {'message': 'Mocked simulation', 'agent': 'simulator'}})() ,
        "evaluator": type('MockAgent', (), {'process': lambda *args, **kwargs: {'message': 'Mocked evaluation', 'agent': 'evaluator'}})() ,
    }
    
    # Asegurarse de que los componentes universales estén disponibles para el orquestador
    # Esto es importante si el orquestador intenta instanciarlos
    MedicalAgentsOrchestrator.universal_detector = UniversalDatasetDetector()
    MedicalAgentsOrchestrator.pipeline_configurator = DynamicPipelineConfig()

    orchestrator = MedicalAgentsOrchestrator(agents)

    # 2. Simular carga de un dataset
    test_df = create_test_dataframe(num_rows=50)
    initial_context = {
        "dataframe": test_df,
        "dataset_uploaded": True,
        "filename": "test_data.csv",
        "rows": test_df.shape[0],
        "columns": test_df.shape[1]
    }
    print(f"Simulando carga de dataset: {initial_context['filename']} ({initial_context['rows']} filas)")

    # 3. Enviar comando "analizar datos" al orquestador
    user_input = "analizar datos"
    print(f"Enviando comando al orquestador: '{user_input}'")
    
    # El orquestador procesará el input, el coordinador lo clasificará como comando
    # y luego el orquestador debería enrutar al analyzer_agent
    response = await orchestrator.process_user_input(user_input, {"dataset_context": initial_context})
    
    print(f"Respuesta del orquestador: {response}")

    # 4. Verificar la respuesta y el estado
    assert response["agent"] == "Analista Clínico" # El agente final debería ser el Analista Clínico
    assert "análisis" in response["message"].lower() or "analizar" in response["message"].lower()

    # Verificar que el estado del análisis se actualizó
    final_state = response["state"]
    assert final_state.analysis_complete is True
    assert "analysis_result" in final_state.context
    assert final_state.context["analysis_result"].get("medical_context") is not None

    print("--- Test: AnalyzerAgent Integrated Flow PASSED ---")

@pytest.mark.asyncio
async def test_analyzer_agent_direct_call():
    """Verifica el comportamiento del AnalyzerAgent al ser llamado directamente."""
    print("\n--- Test: AnalyzerAgent Direct Call ---")
    analyzer = ClinicalAnalyzerAgent()
    test_df = create_test_dataframe(num_rows=20)
    
    # Simular un contexto básico
    context = {
        "dataset_type": "general_medical",
        "detected_columns": {},
        "domain_patterns": []
    }

    analysis_result = await analyzer.analyze_dataset(test_df, context)
    print(f"Resultado del análisis directo: {analysis_result}")

    assert "medical_context" in analysis_result
    assert analysis_result["medical_context"]["primary_context"] == "covid19"
    assert "statistics" in analysis_result
    assert "cleaned_dataframe" in analysis_result # El dataframe limpio debería estar en el contexto
    assert analysis_result["analysis_type"] == "covid19"  # Cambiado de "general" a "covid19"
    assert "message" in analysis_result # El mensaje del LLM

    print("--- Test: AnalyzerAgent Direct Call PASSED ---")