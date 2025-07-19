import pandas as pd
from src.agents.analyzer_agent import ClinicalAnalyzerAgent
from src.agents.generator_agent import SyntheticGeneratorAgent
import sys
import os

# Agregar la carpeta raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Crear un dataset de prueba para COVID-19
data_covid = pd.DataFrame({
    'DIAG': ['COVID-19', 'NEGATIVO', 'POSITIVO'],
    'FARMACO': ['Dexametasona', 'Paracetamol', 'Ibuprofeno'],
    'UCI': [1, 0, 1],
    'TEMP': [38.5, 36.7, 39.0],
    'SAT_02': [92, 98, 85],
    'PATIENT': ['P001', 'P002', 'P003'],
    'EDAD': [65, 45, 70]
})

# Crear un dataset genérico (diabetes)
data_diabetes = pd.DataFrame({
    'glucosa': [120, 150, 200],
    'embarazos': [1, 2, 3],
    'edad': [25, 30, 35],
    'imc': [22.5, 28.0, 35.0],
    'diagnostico': ['positivo', 'negativo', 'positivo']
})

def test_integration_flow():
    # Inicializar agentes
    analyzer = ClinicalAnalyzerAgent()
    generator = SyntheticGeneratorAgent()

    # Probar flujo con dataset COVID-19
    context_covid = {}
    analysis_results_covid = analyzer.analyze_dataset(data_covid, context_covid)
    assert 'is_covid_dataset' in context_covid and context_covid['is_covid_dataset'] is True
    assert 'filtered_file_path' in context_covid

    synthetic_data_covid = generator.generate_synthetic_data(
        original_data=data_covid,
        num_samples=10,
        context=context_covid
    )
    assert 'synthetic_data' in synthetic_data_covid

    # Probar flujo con dataset genérico (diabetes)
    context_diabetes = {}
    analysis_results_diabetes = analyzer.analyze_dataset(data_diabetes, context_diabetes)
    assert 'is_covid_dataset' in context_diabetes and context_diabetes['is_covid_dataset'] is False

    synthetic_data_diabetes = generator.generate_synthetic_data(
        original_data=data_diabetes,
        num_samples=10,
        context=context_diabetes
    )
    assert 'synthetic_data' in synthetic_data_diabetes

if __name__ == "__main__":
    test_integration_flow()
    print("All tests passed!")
