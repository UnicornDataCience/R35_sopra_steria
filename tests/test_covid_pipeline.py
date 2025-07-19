

import pytest
import pandas as pd
import os
import asyncio
import numpy as np
from src.agents.analyzer_agent import ClinicalAnalyzerAgent
from src.agents.generator_agent import SyntheticGeneratorAgent

# Marcar el test como asíncrono para anyio
@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_covid_data_flow():
    """
    Verifica el flujo completo desde el análisis hasta la generación para un dataset COVID-19.
    1. El analizador debe crear un archivo filtrado.
    2. El generador debe usar ese archivo filtrado.
    """
    # 1. Preparación del Entorno
    # Crear un DataFrame de prueba con más filas y varianza
    num_test_rows = 100
    data = {
        'PATIENT ID': list(range(1, num_test_rows + 1)),
        'EDAD/AGE': np.random.randint(18, 90, num_test_rows),
        'SAT_02_ING/INPAT': np.random.randint(85, 100, num_test_rows),
        'RESULTADO/VAL_RESULT': np.random.choice(['POSITIVO', 'NEGATIVO', 'PENDIENTE'], num_test_rows),
        'HIPER_ART/ART_HYPER': np.random.randint(0, 2, num_test_rows),
        'ENF_RESPIRA/RESPI_DISEASE': np.random.randint(0, 2, num_test_rows),
        'DIABETES/DIABETES': np.random.randint(0, 2, num_test_rows),
        'Columna_a_eliminar': np.random.choice(['A', 'B', 'C', 'D', 'E'], num_test_rows)  # Esta columna no está en la lista y debe ser eliminada
    }
    test_df = pd.DataFrame(data)

    # Instanciar los agentes
    analyzer_agent = ClinicalAnalyzerAgent()
    generator_agent = SyntheticGeneratorAgent()

    # 2. Ejecución del Agente Analizador
    # El analizador debería detectar COVID, filtrar columnas y guardar un archivo temporal
    analysis_context = {'dataframe': test_df}
    analysis_result = await analyzer_agent.analyze_dataset(test_df, context=analysis_context)

    # Verificar que el analizador identificó el contexto COVID y creó el archivo
    assert analysis_result['medical_context']['is_covid_dataset'] is True
    filtered_path = analysis_result['medical_context'].get('filtered_file_path')
    assert filtered_path is not None, "El analizador no generó la ruta del archivo filtrado."
    assert os.path.exists(filtered_path), f"El archivo filtrado no fue creado en la ruta: {filtered_path}"

    # 3. Ejecución del Agente Generador
    # El generador debe recibir el contexto del analizador y usar el archivo filtrado
    generation_result = await generator_agent.generate_synthetic_data(
        original_data=test_df,  # Pasamos el DF original, la lógica interna debe ignorarlo
        context=analysis_result['medical_context']
    )

    # 4. Verificación del Resultado
    # Comprobar que el generador usó el dataset filtrado
    assert 'generation_info' in generation_result, "El resultado del generador no contiene 'generation_info'."
    assert generation_result['generation_info']['source_dataset'] == 'filtered', "El generador no utilizó el dataset filtrado como se esperaba."

    # Comprobar que los datos sintéticos no contienen la columna eliminada
    synthetic_data = generation_result.get('synthetic_data')
    assert synthetic_data is not None, "No se generaron datos sintéticos."
    assert 'Columna_a_eliminar' not in synthetic_data.columns, "La columna que debía eliminarse sigue presente en los datos sintéticos."

    # Comprobar que los datos sintéticos contienen las 7 columnas esperadas de COVID-19
    expected_covid_columns = ['PATIENT ID', 'EDAD/AGE', 'SAT_02_ING/INPAT', 'RESULTADO/VAL_RESULT',
                              'HIPER_ART/ART_HYPER', 'ENF_RESPIRA/RESPI_DISEASE', 'DIABETES/DIABETES']
    for col in expected_covid_columns:
        assert col in synthetic_data.columns, f"La columna esperada '{col}' no está presente en los datos sintéticos."

    print("Test de flujo COVID superado con exito.")
    print(f"   - Archivo filtrado creado en: {filtered_path}")
    print(f"   - Fuente de datos del generador: {generation_result['generation_info']['source_dataset']}")

    # Limpieza: eliminar el archivo temporal creado
    if os.path.exists(filtered_path):
        os.remove(filtered_path)

