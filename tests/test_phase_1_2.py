"""
Script de pruebas para validar la implementación de FASE 1 y FASE 2
del plan de refactorización de Patientia

Este script verifica:
1. Funcionamiento del detector universal
2. Configuración dinámica de pipelines
3. Orquestador LangGraph refactorizado
4. Integración completa del sistema

Fecha: 2024-01-15
Versión: 1.0.0
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from typing import Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Añadir paths del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def create_test_datasets():
    """Crea datasets de prueba para diferentes dominios médicos"""

    # Dataset COVID-19
    covid_data = {
        'patient_id': range(1, 101),
        'age': np.random.randint(18, 90, 100),
        'sex': np.random.choice(['M', 'F'], 100),
        'fever': np.random.choice([True, False], 100),
        'cough': np.random.choice([True, False], 100),
        'dyspnea': np.random.choice([True, False], 100),
        'pcr_result': np.random.choice(['POSITIVE', 'NEGATIVE', 'PENDING'], 100),
        'ct_scan': np.random.choice(['NORMAL', 'PNEUMONIA', 'BILATERAL'], 100),
        'admission_date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'hospitalization_days': np.random.randint(1, 30, 100),
        'icu_admission': np.random.choice([True, False], 100),
        'outcome': np.random.choice(['DISCHARGED', 'DECEASED', 'TRANSFERRED'], 100)
    }

    # Dataset Diabetes
    diabetes_data = {
        'patient_id': range(1, 101),
        'age': np.random.randint(25, 80, 100),
        'gender': np.random.choice(['Male', 'Female'], 100),
        'bmi': np.random.uniform(18.5, 35.0, 100),
        'glucose': np.random.uniform(80, 300, 100),
        'hba1c': np.random.uniform(5.0, 12.0, 100),
        'blood_pressure_systolic': np.random.randint(100, 180, 100),
        'blood_pressure_diastolic': np.random.randint(60, 120, 100),
        'cholesterol': np.random.uniform(150, 300, 100),
        'diabetes_type': np.random.choice(['Type1', 'Type2', 'Gestational'], 100),
        'medication': np.random.choice(['Metformin', 'Insulin', 'Glyburide'], 100),
        'complications': np.random.choice(['None', 'Neuropathy', 'Retinopathy'], 100)
    }

    # Dataset Cardiovascular
    cardio_data = {
        'patient_id': range(1, 101),
        'age': np.random.randint(30, 85, 100),
        'sex': np.random.choice(['M', 'F'], 100),
        'chest_pain': np.random.choice([True, False], 100),
        'systolic_bp': np.random.randint(90, 200, 100),
        'diastolic_bp': np.random.randint(60, 130, 100),
        'cholesterol': np.random.uniform(120, 400, 100),
        'troponin': np.random.uniform(0.0, 50.0, 100),
        'ecg': np.random.choice(['Normal', 'Abnormal', 'ST_elevation'], 100),
        'diagnosis': np.random.choice(['Angina', 'Myocardial_infarction', 'Heart_failure'], 100),
        'treatment': np.random.choice(['Medical', 'PCI', 'CABG'], 100),
        'outcome': np.random.choice(['Improved', 'Stable', 'Worsened'], 100)
    }

    return {
        'covid19': pd.DataFrame(covid_data),
        'diabetes': pd.DataFrame(diabetes_data),
        'cardiovascular': pd.DataFrame(cardio_data)
    }

def test_universal_detector():
    """Prueba el detector universal de datasets"""

    print("🔍 PRUEBA 1: Detector Universal de Datasets")
    print("=" * 50)

    try:
        from src.adapters.universal_dataset_detector import UniversalDatasetDetector

        detector = UniversalDatasetDetector()
        test_datasets = create_test_datasets()

        for dataset_name, df in test_datasets.items():
            print(f"\n📊 Analizando dataset: {dataset_name}")
            print(f"   Dimensiones: {df.shape}")

            # Ejecutar análisis
            result = detector.analyze_dataset(df)

            print(f"   Tipo detectado: {result.get('dataset_type', 'unknown')}")
            print(f"   Columnas inferidas: {len(result.get('column_inference', {}))}")
            print(f"   Patrones de dominio: {len(result.get('domain_patterns', []))}")

            # Mostrar algunos patrones
            patterns = result.get('domain_patterns', [])[:3]
            for pattern in patterns:
                print(f"   - {pattern}")
            
            assert result.get('dataset_type') == dataset_name

        print("\n✅ Detector universal funcionando correctamente")

    except Exception as e:
        print(f"\n❌ Error en detector universal: {e}")
        assert False, f"test_universal_detector failed with {e}"

def test_dynamic_pipeline_config():
    """Prueba la configuración dinámica de pipelines"""

    print("\n🔧 PRUEBA 2: Configuración Dinámica de Pipelines")
    print("=" * 50)

    try:
        from src.config.pipeline_config import DynamicPipelineConfig

        configurator = DynamicPipelineConfig()
        test_datasets = create_test_datasets()

        for dataset_name, df in test_datasets.items():
            print(f"\n⚙️ Configurando pipeline para: {dataset_name}")

            # Simular columnas detectadas
            detected_columns = {
                col: {
                    'data_type': str(df[col].dtype),
                    'detected_type': 'categorical' if df[col].dtype == 'object' else 'numeric'
                }
                for col in df.columns
            }

            # Generar configuración completa
            config = configurator.generate_complete_pipeline_config(
                dataset_name,
                detected_columns,
                [f"{dataset_name}_pattern1", f"{dataset_name}_pattern2"],
                len(df)
            )

            print(f"   Configuración generada:")
            print(f"   - Análisis: {config['analysis']['clustering_method']}")
            print(f"   - Síntesis: {config['synthesis']['preferred_method']}")
            print(f"   - Validación: {config['validation']['validation_level']}")
            print(f"   - Simulación: {config['simulation']['simulation_type']}")
            
            assert 'analysis' in config
            assert 'synthesis' in config
            assert 'validation' in config
            assert 'simulation' in config

        print("\n✅ Configuración dinámica funcionando correctamente")

    except Exception as e:
        print(f"\n❌ Error en configuración dinámica: {e}")
        assert False, f"test_dynamic_pipeline_config failed with {e}"

def test_langgraph_orchestrator():
    """Prueba el orquestador LangGraph refactorizado"""

    print("\n🎭 PRUEBA 3: Orquestador LangGraph")
    print("=" * 50)

    try:
        from src.orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator, AgentState

        # Crear agentes mock para prueba
        mock_agents = {
            "coordinator": type('MockAgent', (), {
                'process': lambda self, input_text, context: {
                    'message': f"Coordinador procesando: {input_text}",
                    'agent': 'coordinator'
                }
            })(),
            "analyzer": type('MockAgent', (), {
                'analyze_dataset': lambda self, df, context: {
                    'message': f"Análisis completado para {len(df)} registros",
                    'agent': 'analyzer'
                }
            })()
        }

        # Crear orquestador
        orchestrator = MedicalAgentsOrchestrator(mock_agents)

        print("   Orquestador creado exitosamente")
        print("   Agentes disponibles:", list(orchestrator.agents.keys()))

        # Crear estado de prueba
        test_state = AgentState(
            user_input="Analizar dataset de prueba",
            dataset_type="covid19",
            context={}
        )

        print("   Estado de prueba creado")
        print(f"   Dataset tipo: {test_state.dataset_type}")
        
        assert orchestrator is not None
        assert "coordinator" in orchestrator.agents

        print("\n✅ Orquestador LangGraph funcionando correctamente")

    except Exception as e:
        print(f"\n❌ Error en orquestador LangGraph: {e}")
        assert False, f"test_langgraph_orchestrator failed with {e}"

def test_integration():
    """Prueba la integración completa del sistema"""

    print("\n🔗 PRUEBA 4: Integración Completa")
    print("=" * 50)

    try:
        from src.adapters.universal_dataset_detector import UniversalDatasetDetector
        from src.config.pipeline_config import DynamicPipelineConfig

        # Crear componentes
        detector = UniversalDatasetDetector()
        configurator = DynamicPipelineConfig()

        # Usar dataset COVID-19 de prueba
        test_datasets = create_test_datasets()
        covid_df = test_datasets['covid19']

        print(f"   Dataset de prueba: {covid_df.shape}")

        # Paso 1: Detección universal
        print("   🔍 Ejecutando detección universal...")
        detection_result = detector.analyze_dataset(covid_df)

        # Paso 2: Configuración dinámica
        print("   ⚙️ Generando configuración dinámica...")
        pipeline_config = configurator.generate_complete_pipeline_config(
            detection_result.get('dataset_type', 'unknown'),
            detection_result.get('column_inference', {}),
            detection_result.get('domain_patterns', []),
            len(covid_df)
        )

        # Paso 3: Validar configuración
        print("   📋 Validando configuración generada...")
        required_sections = ['analysis', 'synthesis', 'validation', 'simulation']
        for section in required_sections:
            assert section in pipeline_config, f"Sección faltante: {section}"

        # Paso 4: Mostrar resultados
        print("   📊 Resumen de integración:")
        print(f"      - Tipo detectado: {detection_result.get('dataset_type')}")
        print(f"      - Columnas inferidas: {len(detection_result.get('column_inference', {}))}")
        print(f"      - Patrones encontrados: {len(detection_result.get('domain_patterns', []))}")
        print(f"      - Configuración generada: {len(pipeline_config)} secciones")

        print("\n✅ Integración completa funcionando correctamente")

    except Exception as e:
        print(f"\n❌ Error en integración: {e}")
        assert False, f"test_integration failed with {e}"

def test_interface_components():
    """Prueba componentes de la interfaz"""

    print("\n🖥️ PRUEBA 5: Componentes de Interfaz")
    print("=" * 50)

    try:
        # Probar importación de la interfaz unificada
        interface_path = os.path.join(project_root, 'interfaces', 'chat_llm_unified.py')
        assert os.path.exists(interface_path), "Interfaz unificada no encontrada"
        print("   ✅ Interfaz unificada encontrada")

        # Probar clases auxiliares
        from interfaces.chat_llm_unified import SystemStatus, DatasetContext, MockAgent

        # Crear instancias de prueba
        system_status = SystemStatus()
        dataset_context = DatasetContext()
        mock_agent = MockAgent("Test Agent", "coordinator")

        print("   ✅ Clases auxiliares importadas correctamente")

        # Probar funcionalidad básica
        status_summary = system_status.get_status_summary()
        print(f"   📊 Estado del sistema: {len(status_summary)} componentes")
        
        assert system_status is not None
        assert dataset_context is not None
        assert mock_agent is not None

        print("\n✅ Componentes de interfaz funcionando correctamente")

    except Exception as e:
        print(f"\n❌ Error en componentes de interfaz: {e}")
        assert False, f"test_interface_components failed with {e}"

def generate_test_report(results: Dict[str, bool]):
    """Genera reporte de pruebas"""

    print("\n📋 REPORTE DE PRUEBAS - FASE 1 Y 2")
    print("=" * 50)

    total_tests = len(results)
    passed_tests = sum(results.values())

    print(f"Total de pruebas: {total_tests}")
    print(f"Pruebas exitosas: {passed_tests}")
    print(f"Pruebas fallidas: {total_tests - passed_tests}")
    print(f"Porcentaje de éxito: {(passed_tests/total_tests)*100:.1f}%")

    print("\nDetalle de resultados:")
    for test_name, passed in results.items():
        status = "✅ PASÓ" if passed else "❌ FALLÓ"
        print(f"  {test_name}: {status}")

    if passed_tests == total_tests:
        print("\n🎉 TODAS LAS PRUEBAS PASARON - FASE 1 Y 2 COMPLETADAS")
        print("✅ El sistema está listo para la siguiente fase")
    else:
        print("\n⚠️ ALGUNAS PRUEBAS FALLARON")
        print("❌ Revisa los errores antes de continuar")

    return passed_tests == total_tests

def main():
    """Función principal de pruebas"""

    print("🧬 PATIENTIA - PRUEBAS DE REFACTORIZACIÓN")
    print("🔬 Validando implementación de FASE 1 y FASE 2")
    print("=" * 60)

    # Ejecutar pruebas
    results = {}
    
    # Esta forma de ejecutar ya no es compatible con pytest
    # Se recomienda ejecutar con `pytest` desde la terminal
    print("Ejecutando pruebas con pytest...")


if __name__ == "__main__":
    try:
        # El main ya no ejecuta los tests directamente para ser compatible con pytest
        print("Para ejecutar las pruebas, usa el comando: pytest")
        exit(0)
    except Exception as e:
        print(f"\n💥 Error crítico en pruebas: {e}")
        exit(1)
