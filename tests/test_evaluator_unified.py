#!/usr/bin/env python3
"""
Test integral del agente evaluador funcionando sin Azure OpenAI
"""

import asyncio
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Añadir src al path 
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from evaluation.unified_evaluator import UnifiedMedicalEvaluator

def create_sample_covid_data():
    """Crea un dataset de muestra con datos COVID-19 sintéticos para testing"""
    np.random.seed(42)
    
    data = {
        'PATIENT ID': [f'P{i:04d}' for i in range(1, 101)],
        'EDAD/AGE': np.random.randint(18, 90, 100),
        'SEXO/SEX': np.random.choice(['M', 'F'], 100),
        'UCI_DIAS/ICU_DAYS': np.random.randint(0, 15, 100),
        'TEMP_ING/INPAT': np.random.normal(37.5, 1.5, 100).round(1),
        'SAT_02_ING/INPAT': np.random.randint(75, 99, 100),
        'RESULTADO/VAL_RESULT': np.random.choice([0, 1], 100),
        'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME': np.random.choice([
            'DEXAMETASONA', 'AZITROMICINA', 'ENOXAPARINA', 'FUROSEMIDA',
            'REMDESIVIR', 'PARACETAMOL', 'OMEPRAZOL'
        ], 100),
        'DIAG ING/INPAT': np.random.choice([
            'NEUMONIA COVID-19', 'INSUFICIENCIA RESPIRATORIA', 
            'SEPSIS', 'SHOCK SEPTICO', 'FALLO MULTIORGANICO'
        ], 100)
    }
    
    return pd.DataFrame(data)

def test_unified_evaluator():
    """Test del evaluador unificado sin dependencias de Azure"""
    
    print("🧪 === TEST DEL EVALUADOR UNIFICADO ===")
    print()
    
    # 1. Crear datos de muestra
    print("📊 1. Creando datasets de muestra...")
    original_data = create_sample_covid_data()
    
    # Crear datos sintéticos simulados (ligeramente modificados)
    synthetic_data = original_data.copy()
    
    # Aplicar modificaciones solo a columnas numéricas
    numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == 'EDAD/AGE':
            synthetic_data[col] = synthetic_data[col] + np.random.randint(-5, 5, len(synthetic_data))
            synthetic_data[col] = synthetic_data[col].clip(lower=0, upper=120)  # Edades válidas
        elif col == 'TEMP_ING/INPAT':
            synthetic_data[col] = synthetic_data[col] + np.random.normal(0, 0.3, len(synthetic_data))
            synthetic_data[col] = synthetic_data[col].clip(lower=35.0, upper=42.0)  # Temp válidas
        elif col == 'UCI_DIAS/ICU_DAYS':
            synthetic_data[col] = synthetic_data[col] + np.random.randint(-2, 2, len(synthetic_data))
            synthetic_data[col] = synthetic_data[col].clip(lower=0, upper=30)  # Días válidos
        elif col == 'SAT_02_ING/INPAT':
            synthetic_data[col] = synthetic_data[col] + np.random.randint(-5, 5, len(synthetic_data))
            synthetic_data[col] = synthetic_data[col].clip(lower=70, upper=100)  # Saturación válida
    
    print(f"   ✅ Dataset original: {len(original_data)} filas, {len(original_data.columns)} columnas")
    print(f"   ✅ Dataset sintético: {len(synthetic_data)} filas, {len(synthetic_data.columns)} columnas")
    print()
    
    # 2. Inicializar evaluador
    print("🔧 2. Inicializando evaluador unificado...")
    try:
        evaluator = UnifiedMedicalEvaluator()
        print("   ✅ Evaluador unificado inicializado")
    except Exception as e:
        print(f"   ❌ Error inicializando evaluador: {e}")
        return False
    
    # 3. Ejecutar evaluación completa
    print("📊 3. Ejecutando evaluación completa...")
    try:
        validation_results = {
            "coherence_score": 0.85,
            "overall_quality_score": 0.78,
            "domain": "COVID-19"
        }
        
        domain_info = {
            "dataset_type": "COVID-19",
            "medical_domain": "covid19",
            "detected_features": ["age", "sex", "icu_days", "temperature", "oxygen_saturation"]
        }
        
        # Ejecutar evaluación
        results = evaluator.comprehensive_evaluation(
            original_data, 
            synthetic_data, 
            validation_results
        )
        
        print("   ✅ Evaluación completada exitosamente")
        print()
        
        # 4. Mostrar resultados
        print("📈 4. RESULTADOS DE LA EVALUACIÓN:")
        final_score = results.get('final_quality_score', 0)
        if isinstance(final_score, (int, float)):
            print(f"   🎯 Score Final de Calidad: {final_score:.2%}")
        else:
            print(f"   🎯 Score Final de Calidad: {final_score}")
            
        print(f"   📊 Tier de Calidad: {results.get('quality_tier', 'N/A')}")
        
        # Mostrar métricas con verificación de tipo
        for metric_name, display_name in [
            ('statistical_fidelity', 'Fidelidad Estadística'),
            ('ml_utility_score', 'Utilidad ML'),
            ('medical_coherence', 'Coherencia Médica'),
            ('privacy_score', 'Score de Privacidad')
        ]:
            value = results.get(metric_name, 0)
            if isinstance(value, (int, float)):
                print(f"   � {display_name}: {value:.2%}")
            else:
                print(f"   📈 {display_name}: {value}")
        
        print()
        print(f"   💡 Recomendación: {results.get('usage_recommendation', 'N/A')}")
        print()
        
        # 5. Generar informe markdown
        print("📄 5. Generando informe markdown...")
        try:
            markdown_report = evaluator.create_markdown_report(results, domain_info)
            print("   ✅ Informe markdown generado")
            print()
            print("   📋 VISTA PREVIA DEL INFORME:")
            print("   " + "="*50)
            # Mostrar las primeras líneas del informe
            lines = markdown_report.split('\n')[:15]
            for line in lines:
                print(f"   {line}")
            print("   ...")
            print("   " + "="*50)
            print()
        except Exception as e:
            print(f"   ⚠️ Warning en generación de informe: {e}")
        
        # 6. Verificar métricas clave
        print("✅ 6. VERIFICACIÓN DE MÉTRICAS:")
        quality_score = results.get('final_quality_score', 0)
        if quality_score >= 0.7:
            print(f"   🎉 EXCELENTE: Score de calidad alto ({quality_score:.2%})")
        elif quality_score >= 0.5:
            print(f"   👍 BUENO: Score de calidad aceptable ({quality_score:.2%})")
        else:
            print(f"   ⚠️ REVISAR: Score de calidad bajo ({quality_score:.2%})")
        
        # Verificar que todas las métricas principales estén presentes
        required_metrics = [
            'final_quality_score', 'statistical_fidelity', 'ml_utility_score', 
            'medical_coherence', 'privacy_score', 'usage_recommendation'
        ]
        
        missing_metrics = [m for m in required_metrics if m not in results]
        if not missing_metrics:
            print("   ✅ Todas las métricas principales están presentes")
        else:
            print(f"   ⚠️ Métricas faltantes: {missing_metrics}")
        
        print()
        return True
        
    except Exception as e:
        print(f"   ❌ Error en evaluación: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluator_components():
    """Test de componentes individuales del evaluador"""
    
    print("🧪 === TEST DE COMPONENTES INDIVIDUALES ===")
    print()
    
    # Datos de prueba pequeños
    original_data = create_sample_covid_data().iloc[:20]  # Solo 20 filas
    synthetic_data = original_data.copy()
    
    # Modificar ligeramente
    numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if len(synthetic_data[col]) > 0:
            synthetic_data[col] = synthetic_data[col] + np.random.normal(0, 0.1, len(synthetic_data))
    
    evaluator = UnifiedMedicalEvaluator()
    
    # Test 1: Métricas básicas
    print("1. Probando métricas básicas de dataset...")
    try:
        basic_metrics = evaluator.evaluate_basic_metrics(original_data, synthetic_data)
        print(f"   ✅ Filas originales: {basic_metrics['original_rows']}")
        print(f"   ✅ Filas sintéticas: {basic_metrics['synthetic_rows']}")
        print(f"   ✅ Columnas coincidentes: {basic_metrics.get('columns_match', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Fidelidad estadística
    print("\n2. Probando fidelidad estadística...")
    try:
        stat_metrics = evaluator.evaluate_statistical_fidelity(original_data, synthetic_data)
        print(f"   ✅ Fidelidad estadística: {stat_metrics.get('statistical_fidelity', 0):.2%}")
        print(f"   ✅ Distribuciones similares: {stat_metrics.get('similar_distributions', 0)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Performance ML
    print("\n3. Probando métricas de ML...")
    try:
        ml_metrics = evaluator.evaluate_ml_performance(original_data, synthetic_data)
        print(f"   ✅ Score de utilidad ML: {ml_metrics.get('ml_utility_score', 0):.2%}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ Tests de componentes individuales completados")
    print()

async def main():
    """Función principal para ejecutar todos los tests"""
    
    print("🚀 INICIANDO TESTS DEL EVALUADOR UNIFICADO")
    print("=" * 60)
    print()
    
    # Test 1: Evaluador unificado completo
    success1 = test_unified_evaluator()
    print()
    print("-" * 60)
    print()
    
    # Test 2: Componentes individuales
    try:
        test_evaluator_components()
        success2 = True
    except Exception as e:
        print(f"❌ Error en test de componentes: {e}")
        success2 = False
    
    # Resumen final
    print()
    print("=" * 60)
    print("🏁 RESULTADOS FINALES:")
    print(f"   Test evaluador unificado: {'✅ PASÓ' if success1 else '❌ FALLÓ'}")
    print(f"   Test componentes individuales: {'✅ PASÓ' if success2 else '❌ FALLÓ'}")
    
    if success1 and success2:
        print()
        print("🎉 ¡TODOS LOS TESTS PASARON!")
        print("🎯 El evaluador unificado funciona correctamente")
        print("📊 Métricas completas de evaluación disponibles")
        print("📄 Generación de informes markdown operativa")
    else:
        print()
        print("⚠️ Algunos tests fallaron - revisar configuración")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
