#!/usr/bin/env python3
"""
Test integral del flujo completo incluyendo el agente evaluador
"""

import asyncio
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Añadir src al path correctamente
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Importar directamente sin src prefix
from src.orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator
from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.analyzer_agent import ClinicalAnalyzerAgent
from src.agents.generator_agent import SyntheticGeneratorAgent
from src.agents.validator_agent import MedicalValidatorAgent
from src.agents.evaluator_agent import UtilityEvaluatorAgent
from src.adapters.universal_dataset_detector import UniversalDatasetDetector

def create_sample_covid_data():
    """Crea un dataset de muestra con datos COVID-19 sintéticos para testing"""
    np.random.seed(42)
    
    data = {
        'PATIENT ID': [f'TEST_{i:03d}' for i in range(1, 51)],
        'EDAD/AGE': np.random.randint(18, 85, 50),
        'SEXO/SEX': np.random.choice(['M', 'F'], 50),
        'DIAG ING/INPAT': np.random.choice([
            'COVID-19 PNEUMONIA', 'RESPIRATORY FAILURE', 'ACUTE RESPIRATORY DISTRESS',
            'VIRAL PNEUMONIA', 'SEPSIS', 'MULTI-ORGAN FAILURE'
        ], 50),
        'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME': np.random.choice([
            'DEXAMETASONA', 'AZITROMICINA', 'ENOXAPARINA', 'FUROSEMIDA',
            'REMDESIVIR', 'TOCILIZUMAB', 'METILPREDNISOLONA', 'HEPARINA'
        ], 50),
        'UCI_DIAS/ICU_DAYS': np.random.randint(0, 30, 50),
        'TEMP_ING/INPAT': np.round(np.random.uniform(36.0, 42.0, 50), 1),
        'SAT_02_ING/INPAT': np.random.randint(70, 100, 50),
        'RESULTADO/VAL_RESULT': np.random.choice([0, 1], 50, p=[0.7, 0.3]),  # 0=recuperado, 1=fallecido
        'MOTIVO_ALTA/DESTINY_DISCHARGE_ING': np.random.choice([
            'RECOVERED', 'TRANSFERRED', 'DECEASED', 'DISCHARGED HOME'
        ], 50)
    }
    
    return pd.DataFrame(data)

async def test_complete_workflow():
    """Prueba el flujo completo desde análisis hasta evaluación"""
    
    print("🧪 === TEST FLUJO COMPLETO CON EVALUADOR ===")
    print()
    
    # 1. Crear datos de prueba
    print("📊 1. Creando dataset de prueba...")
    df = create_sample_covid_data()
    print(f"   ✅ Dataset creado: {len(df)} registros, {len(df.columns)} columnas")
    print(f"   📋 Columnas: {list(df.columns)}")
    
    # 2. Inicializar orquestador
    print("\n🤖 2. Inicializando orquestador...")
    orchestrator = MedicalAgentsOrchestrator()
    print("   ✅ Orquestador inicializado")
    
    # 3. Análisis de dataset
    print("\n🔍 3. Ejecutando análisis del dataset...")
    input_data = {
        "query": "Analiza este dataset de COVID-19 y prepara para generación sintética",
        "dataframe": df,
        "agent_request": "analyzer"
    }
    
    result = await orchestrator.process_request(input_data)
    
    if "error" in result:
        print(f"   ❌ Error en análisis: {result.get('message', 'Error desconocido')}")
        return False
    
    print(f"   ✅ Análisis completado por: {result.get('agent', 'N/A')}")
    
    # 4. Generación sintética
    print("\n⚙️ 4. Ejecutando generación sintética...")
    generation_data = {
        "query": "Genera 30 registros sintéticos usando CTGAN",
        "dataframe": df,
        "agent_request": "generator",
        "generator_type": "ctgan",
        "num_samples": 30,
        "universal_analysis": result.get("universal_analysis", {})
    }
    
    gen_result = await orchestrator.process_request(generation_data)
    
    if "error" in gen_result:
        print(f"   ❌ Error en generación: {gen_result.get('message', 'Error desconocido')}")
        return False
    
    print(f"   ✅ Generación completada por: {gen_result.get('agent', 'N/A')}")
    
    synthetic_data = gen_result.get("synthetic_data")
    if synthetic_data is None or len(synthetic_data) == 0:
        print("   ❌ No se generaron datos sintéticos")
        return False
    
    print(f"   📊 Registros sintéticos generados: {len(synthetic_data)}")
    
    # 5. Validación médica
    print("\n✅ 5. Ejecutando validación médica...")
    validation_data = {
        "query": "Valida la coherencia médica de los datos sintéticos",
        "dataframe": df,
        "synthetic_data": synthetic_data,
        "agent_request": "validator",
        "universal_analysis": result.get("universal_analysis", {})
    }
    
    val_result = await orchestrator.process_request(validation_data)
    
    if "error" in val_result:
        print(f"   ❌ Error en validación: {val_result.get('message', 'Error desconocido')}")
        return False
    
    print(f"   ✅ Validación completada por: {val_result.get('agent', 'N/A')}")
    
    validation_results = val_result.get("validation_results", {})
    print(f"   📋 Tasa de validación: {validation_results.get('success_rate', 0):.1f}%")
    
    # 6. EVALUACIÓN DE UTILIDAD (NUEVO)
    print("\n📈 6. Ejecutando evaluación de utilidad...")
    evaluation_data = {
        "query": "Evalúa la calidad, fidelidad y utilidad de los datos sintéticos",
        "dataframe": df,
        "synthetic_data": synthetic_data,
        "agent_request": "evaluator",
        "validation_results": validation_results,
        "universal_analysis": result.get("universal_analysis", {})
    }
    
    eval_result = await orchestrator.process_request(evaluation_data)
    
    if "error" in eval_result:
        print(f"   ❌ Error en evaluación: {eval_result.get('message', 'Error desconocido')}")
        return False
    
    print(f"   ✅ Evaluación completada por: {eval_result.get('agent', 'N/A')}")
    
    # Extraer métricas de evaluación
    evaluation_results = eval_result.get("evaluation_results", {})
    quality_score = eval_result.get("quality_score", 0)
    recommendation = eval_result.get("recommendation", "N/A")
    
    print(f"   🎯 Score de calidad: {quality_score:.2f}")
    print(f"   💡 Recomendación: {recommendation}")
    
    # 7. Mostrar mensaje del evaluador
    print("\n📋 7. Reporte de evaluación:")
    print("=" * 60)
    eval_message = eval_result.get("message", "No hay reporte disponible")
    print(eval_message)
    print("=" * 60)
    
    # 8. Resumen final del flujo
    print("\n🎉 === RESUMEN DEL FLUJO COMPLETO ===")
    print(f"📊 Datos originales: {len(df)} registros")
    print(f"🔬 Datos sintéticos: {len(synthetic_data)} registros")
    print(f"✅ Validación médica: {validation_results.get('success_rate', 0):.1f}%")
    print(f"📈 Score de calidad: {quality_score:.2f}")
    print(f"💡 Recomendación: {recommendation}")
    print()
    print("✅ FLUJO COMPLETO EXITOSO - TODOS LOS AGENTES FUNCIONANDO")
    
    return True

async def test_individual_evaluator():
    """Prueba específica del agente evaluador"""
    
    print("\n🔬 === TEST ESPECÍFICO DEL EVALUADOR ===")
    
    # Crear datos simulados
    original_data = create_sample_covid_data()
    
    # Simular datos sintéticos (con algunas diferencias)
    synthetic_data = original_data.copy()
    
    # Modificar solo columnas numéricas de forma segura
    if 'EDAD/AGE' in synthetic_data.columns:
        synthetic_data['EDAD/AGE'] = synthetic_data['EDAD/AGE'] + np.random.randint(-5, 5, len(synthetic_data))
        synthetic_data['EDAD/AGE'] = synthetic_data['EDAD/AGE'].clip(lower=0)  # No edades negativas
    
    if 'TEMP_ING/INPAT' in synthetic_data.columns:
        synthetic_data['TEMP_ING/INPAT'] = synthetic_data['TEMP_ING/INPAT'] + np.random.normal(0, 0.3, len(synthetic_data))
        synthetic_data['TEMP_ING/INPAT'] = synthetic_data['TEMP_ING/INPAT'].clip(lower=30.0, upper=45.0)
    
    if 'SAT_02_ING/INPAT' in synthetic_data.columns:
        synthetic_data['SAT_02_ING/INPAT'] = synthetic_data['SAT_02_ING/INPAT'] + np.random.randint(-3, 3, len(synthetic_data))
        synthetic_data['SAT_02_ING/INPAT'] = synthetic_data['SAT_02_ING/INPAT'].clip(lower=50, upper=100)
    
    # Simular resultados de validación
    validation_results = {
        "success_rate": 87.5,
        "validation_errors": ["Medicamento no reconocido: FARMACO_X"],
        "domain": "covid",
        "valid_records": 35,
        "invalid_records": 5
    }
    
    domain_info = {
        "detected_domain": "covid",
        "confidence_score": 0.95,
        "key_features": ["UCI_DIAS/ICU_DAYS", "SAT_02_ING/INPAT"]
    }
    
    # Inicializar evaluador
    evaluator = UtilityEvaluatorAgent()
    
    # Preparar contexto
    context = {
        "dataframe": original_data,
        "synthetic_data": synthetic_data,
        "validation_results": validation_results,
        "universal_analysis": domain_info
    }
    
    # Ejecutar evaluación
    print("🔍 Ejecutando evaluación...")
    result = await evaluator.process(
        "Evalúa la calidad y utilidad de estos datos sintéticos", 
        context
    )
    
    if "error" in result:
        print(f"❌ Error: {result.get('message', 'Error desconocido')}")
        return False
    
    print("✅ Evaluación completada")
    print(f"🎯 Score: {result.get('quality_score', 0):.2f}")
    print(f"💡 Recomendación: {result.get('recommendation', 'N/A')}")
    
    print("\n📋 Reporte detallado:")
    print("-" * 40)
    print(result.get("message", "No hay reporte"))
    print("-" * 40)
    
    return True

async def main():
    """Función principal del test"""
    
    print("🚀 INICIANDO TESTS DEL SISTEMA PATIENTIA")
    print("=" * 60)
    
    try:
        # Test 1: Evaluador individual
        success1 = await test_individual_evaluator()
        
        if success1:
            print("\n" + "=" * 60)
            # Test 2: Flujo completo
            success2 = await test_complete_workflow()
        else:
            print("❌ Fallo en test individual, no ejecutando flujo completo")
            success2 = False
        
        # Resultado final
        print("\n" + "=" * 60)
        if success1 and success2:
            print("🎉 TODOS LOS TESTS EXITOSOS")
            print("✅ El agente evaluador está correctamente integrado")
            print("✅ El flujo completo funciona correctamente")
        else:
            print("❌ ALGUNOS TESTS FALLARON")
            if not success1:
                print("   - Test individual del evaluador: FALLÓ")
            if not success2:
                print("   - Test de flujo completo: FALLÓ")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
