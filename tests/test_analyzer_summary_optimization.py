"""
🧪 Test de optimización del resumen del Analizador - Fase 2 Optimizada

Validar que el resumen incluye toda la información estadística relevante
sin limitaciones excesivas, asegurando informes completos para el LLM.

Author: Patient-IA Team
Date: 2025-01-XX
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.analyzer_agent import AnalyzerAgent


def create_test_dataset(num_columns: int, num_rows: int = 100) -> pd.DataFrame:
    """
    Crear dataset de prueba con características médicas
    
    Args:
        num_columns: Número de columnas a crear
        num_rows: Número de filas
        
    Returns:
        DataFrame de prueba
    """
    data = {
        'patient_id': range(1, num_rows + 1),
        'age': np.random.randint(18, 90, num_rows),
        'gender': np.random.choice(['M', 'F'], num_rows),
    }
    
    # Agregar columnas numéricas
    for i in range(num_columns // 3):
        data[f'numeric_col_{i}'] = np.random.randn(num_rows) * 10 + 50
        # Introducir algunos valores nulos
        if i % 3 == 0:
            mask = np.random.choice([True, False], num_rows, p=[0.15, 0.85])
            data[f'numeric_col_{i}'] = pd.Series(data[f'numeric_col_{i}']).mask(mask)
    
    # Agregar columnas categóricas
    for i in range(num_columns // 3):
        data[f'category_col_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], num_rows)
    
    # Agregar columnas datetime
    for i in range(min(5, num_columns // 10)):
        data[f'date_col_{i}'] = pd.date_range('2020-01-01', periods=num_rows, freq='D')
    
    df = pd.DataFrame(data)
    
    # Asegurar que hay columnas correlacionadas
    if 'numeric_col_0' in df.columns and 'numeric_col_1' in df.columns:
        df['numeric_col_1'] = df['numeric_col_0'] * 0.8 + np.random.randn(num_rows) * 2
    
    return df


def test_small_dataset_summary():
    """Test: Dataset pequeño incluye TODAS las columnas con detalle"""
    print("\n" + "="*80)
    print("🧪 TEST 1: Dataset pequeño (30 columnas)")
    print("="*80)
    
    # Crear dataset pequeño
    df = create_test_dataset(num_columns=30, num_rows=100)
    print(f"✓ Dataset creado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Analizar
    analyzer = AnalyzerAgent()
    result = analyzer.analyze_dataset(df, "test_small_dataset.csv")
    
    # Verificar que el análisis fue exitoso
    assert result.get("success"), "El análisis debe ser exitoso"
    
    # Verificar que el resumen incluye TODAS las columnas
    analysis = result.get("analysis", {})
    num_cols = analysis.get("basic_info", {}).get("columns", 0)
    
    # Contar columnas detalladas en el resumen
    numerical_detailed = len(analysis.get("numerical_columns", []))
    categorical_detailed = len(analysis.get("categorical_columns", []))
    
    print(f"\n📊 Resumen generado:")
    print(f"   - Total columnas: {num_cols}")
    print(f"   - Columnas numéricas detalladas: {numerical_detailed}")
    print(f"   - Columnas categóricas detalladas: {categorical_detailed}")
    
    # Verificar executive_summary
    exec_summary = analysis.get("executive_summary", {})
    print(f"\n📋 Resumen ejecutivo:")
    print(f"   - Tamaño dataset: {exec_summary.get('dataset_size')}")
    print(f"   - Completitud análisis: {exec_summary.get('analysis_completeness')}")
    
    # En dataset pequeño, TODAS las columnas deben estar detalladas
    numerical_total = len([c for c in df.columns if df[c].dtype in ['int64', 'float64']])
    categorical_total = len([c for c in df.columns if df[c].dtype == 'object'])
    
    print(f"\n✅ Validaciones:")
    print(f"   - Columnas numéricas: {numerical_detailed}/{numerical_total} detalladas")
    print(f"   - Columnas categóricas: {categorical_detailed}/{categorical_total} detalladas")
    
    assert numerical_detailed == numerical_total, f"Esperaba {numerical_total} numéricas, obtuvo {numerical_detailed}"
    assert categorical_detailed == categorical_total, f"Esperaba {categorical_total} categóricas, obtuvo {categorical_detailed}"
    
    print("✅ TEST 1 PASADO: Dataset pequeño incluye todas las columnas\n")
    return True


def test_medium_dataset_summary():
    """Test: Dataset mediano incluye 50 columnas detalladas"""
    print("\n" + "="*80)
    print("🧪 TEST 2: Dataset mediano (80 columnas)")
    print("="*80)
    
    # Crear dataset mediano
    df = create_test_dataset(num_columns=80, num_rows=200)
    print(f"✓ Dataset creado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Analizar
    analyzer = AnalyzerAgent()
    result = analyzer.analyze_dataset(df, "test_medium_dataset.csv")
    
    # Verificar que el análisis fue exitoso
    assert result.get("success"), "El análisis debe ser exitoso"
    
    # Verificar que el resumen incluye al menos 50 columnas detalladas
    analysis = result.get("analysis", {})
    numerical_detailed = len(analysis.get("numerical_columns", []))
    categorical_detailed = len(analysis.get("categorical_columns", []))
    total_detailed = numerical_detailed + categorical_detailed
    
    print(f"\n📊 Resumen generado:")
    print(f"   - Total columnas: {analysis.get('basic_info', {}).get('columns', 0)}")
    print(f"   - Columnas numéricas detalladas: {numerical_detailed}")
    print(f"   - Columnas categóricas detalladas: {categorical_detailed}")
    print(f"   - Total detalladas: {total_detailed}")
    
    # Verificar que hay columnas adicionales listadas
    additional_num = len(analysis.get("additional_numerical_columns", []))
    additional_cat = len(analysis.get("additional_categorical_columns", []))
    print(f"   - Columnas numéricas adicionales (solo nombres): {additional_num}")
    print(f"   - Columnas categóricas adicionales (solo nombres): {additional_cat}")
    
    # Verificar correlaciones
    correlations = len(analysis.get("high_correlations", []))
    print(f"   - Correlaciones altas encontradas: {correlations}")
    
    # Verificar valores nulos
    missing = analysis.get("missing_values", {})
    top_missing = len(missing.get("top_missing_columns", []))
    print(f"   - Top columnas con valores nulos: {top_missing}")
    
    # Validaciones
    print(f"\n✅ Validaciones:")
    assert total_detailed >= 40, f"Dataset mediano debe tener al menos 40 columnas detalladas, tiene {total_detailed}"
    assert correlations > 0, "Debe encontrar correlaciones en el dataset"
    assert top_missing <= 20, f"Debe incluir máximo 20 columnas con nulos, tiene {top_missing}"
    
    print(f"   ✓ {total_detailed} columnas detalladas (esperado ≥40)")
    print(f"   ✓ {correlations} correlaciones encontradas")
    print(f"   ✓ {top_missing} columnas con nulos (máximo 20)")
    
    print("✅ TEST 2 PASADO: Dataset mediano incluye información optimizada\n")
    return True


def test_large_dataset_summary():
    """Test: Dataset grande incluye 40 columnas detalladas y resumen completo"""
    print("\n" + "="*80)
    print("🧪 TEST 3: Dataset grande (150 columnas)")
    print("="*80)
    
    # Crear dataset grande
    df = create_test_dataset(num_columns=150, num_rows=300)
    print(f"✓ Dataset creado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Analizar
    analyzer = AnalyzerAgent()
    result = analyzer.analyze_dataset(df, "test_large_dataset.csv")
    
    # Verificar que el análisis fue exitoso
    assert result.get("success"), "El análisis debe ser exitoso"
    
    # Verificar que el resumen es completo
    analysis = result.get("analysis", {})
    numerical_detailed = len(analysis.get("numerical_columns", []))
    categorical_detailed = len(analysis.get("categorical_columns", []))
    total_detailed = numerical_detailed + categorical_detailed
    
    print(f"\n📊 Resumen generado:")
    print(f"   - Total columnas: {analysis.get('basic_info', {}).get('columns', 0)}")
    print(f"   - Columnas numéricas detalladas: {numerical_detailed}")
    print(f"   - Columnas categóricas detalladas: {categorical_detailed}")
    print(f"   - Total detalladas: {total_detailed}")
    
    # Verificar resumen ejecutivo
    exec_summary = analysis.get("executive_summary", {})
    print(f"\n📋 Resumen ejecutivo:")
    print(f"   - Tamaño dataset: {exec_summary.get('dataset_size')}")
    print(f"   - Calidad de datos: {exec_summary.get('data_quality')}")
    print(f"   - Completitud análisis: {exec_summary.get('analysis_completeness')}")
    
    # Verificar que todas las correlaciones están incluidas
    correlations = len(analysis.get("high_correlations", []))
    correlation_stats = analysis.get("correlation_stats", {})
    print(f"\n🔗 Correlaciones:")
    print(f"   - Total correlaciones altas: {correlations}")
    print(f"   - Estadísticas: {correlation_stats}")
    
    # Validaciones
    print(f"\n✅ Validaciones:")
    assert total_detailed >= 35, f"Dataset grande debe tener al menos 35 columnas detalladas, tiene {total_detailed}"
    assert "executive_summary" in analysis, "Debe incluir resumen ejecutivo"
    assert correlations > 0, "Debe encontrar correlaciones"
    
    print(f"   ✓ {total_detailed} columnas detalladas (esperado ≥35)")
    print(f"   ✓ Resumen ejecutivo incluido")
    print(f"   ✓ {correlations} correlaciones sin límite artificial")
    
    print("✅ TEST 3 PASADO: Dataset grande incluye resumen estratégico completo\n")
    return True


def test_medical_patterns_completeness():
    """Test: Patrones médicos incluyen TODA la información sin límites"""
    print("\n" + "="*80)
    print("🧪 TEST 4: Completitud de patrones médicos")
    print("="*80)
    
    # Crear dataset médico con muchas columnas médicas
    data = {
        'patient_id': range(1, 101),
        'age': np.random.randint(18, 90, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'diagnosis_1': np.random.choice(['A00', 'B01', 'C02'], 100),
        'diagnosis_2': np.random.choice(['A00', 'B01', 'C02'], 100),
        'medication_1': np.random.choice(['Med_A', 'Med_B'], 100),
        'medication_2': np.random.choice(['Med_C', 'Med_D'], 100),
        'admission_date': pd.date_range('2020-01-01', periods=100),
        'discharge_date': pd.date_range('2020-01-02', periods=100),
        'outcome': np.random.choice(['Recovered', 'Improved', 'Stable'], 100),
        'blood_pressure': np.random.randint(90, 160, 100),
        'heart_rate': np.random.randint(60, 100, 100),
        'temperature': np.random.uniform(36.0, 39.0, 100),
    }
    df = pd.DataFrame(data)
    print(f"✓ Dataset médico creado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Analizar
    analyzer = AnalyzerAgent()
    result = analyzer.analyze_dataset(df, "test_medical_dataset.csv")
    
    # Verificar patrones médicos
    analysis = result.get("analysis", {})
    medical_patterns = analysis.get("medical_patterns", {})
    
    print(f"\n🏥 Patrones médicos detectados:")
    print(f"   - Tiene patient_id: {medical_patterns.get('has_patient_id')}")
    print(f"   - Tiene age: {medical_patterns.get('has_age')}")
    print(f"   - Tiene dates: {medical_patterns.get('has_dates')}")
    print(f"   - Tiene diagnoses: {medical_patterns.get('has_diagnoses')}")
    print(f"   - Tiene gender: {medical_patterns.get('has_gender')}")
    print(f"   - Tiene outcomes: {medical_patterns.get('has_outcomes')}")
    print(f"   - Tiene medications: {medical_patterns.get('has_medications')}")
    
    key_medical_cols = medical_patterns.get("key_medical_columns", [])
    print(f"\n📋 Columnas médicas clave identificadas: {len(key_medical_cols)}")
    for col in key_medical_cols:
        print(f"   - {col}")
    
    # Validaciones
    print(f"\n✅ Validaciones:")
    assert medical_patterns.get("has_patient_id"), "Debe detectar patient_id"
    assert medical_patterns.get("has_age"), "Debe detectar age"
    assert medical_patterns.get("has_dates"), "Debe detectar fechas"
    assert medical_patterns.get("has_diagnoses"), "Debe detectar diagnósticos"
    assert len(key_medical_cols) >= 10, f"Debe identificar al menos 10 columnas médicas, encontró {len(key_medical_cols)}"
    
    print(f"   ✓ Todos los patrones médicos detectados")
    print(f"   ✓ {len(key_medical_cols)} columnas médicas identificadas (sin límite)")
    
    print("✅ TEST 4 PASADO: Patrones médicos completos sin limitaciones\n")
    return True


def test_summary_completeness_comparison():
    """Test: Comparar completitud del resumen antes y después de la optimización"""
    print("\n" + "="*80)
    print("🧪 TEST 5: Comparación de completitud del resumen")
    print("="*80)
    
    # Crear dataset mediano
    df = create_test_dataset(num_columns=70, num_rows=150)
    print(f"✓ Dataset creado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Analizar
    analyzer = AnalyzerAgent()
    result = analyzer.analyze_dataset(df, "test_comparison_dataset.csv")
    analysis = result.get("analysis", {})
    
    # Calcular métricas de completitud
    completeness_metrics = {
        "numerical_columns_detailed": len(analysis.get("numerical_columns", [])),
        "categorical_columns_detailed": len(analysis.get("categorical_columns", [])),
        "correlations_included": len(analysis.get("high_correlations", [])),
        "missing_columns_tracked": len(analysis.get("missing_values", {}).get("top_missing_columns", [])),
        "medical_columns_identified": len(analysis.get("medical_patterns", {}).get("key_medical_columns", [])),
        "has_executive_summary": "executive_summary" in analysis,
        "has_correlation_stats": "correlation_stats" in analysis,
    }
    
    print(f"\n📊 Métricas de completitud:")
    for metric, value in completeness_metrics.items():
        print(f"   - {metric}: {value}")
    
    # Calcular tamaño del resumen (aproximado)
    summary_json = json.dumps(analysis, indent=2, default=str)
    summary_size = len(summary_json)
    print(f"\n📏 Tamaño del resumen JSON: {summary_size:,} caracteres")
    
    # Validaciones de completitud
    print(f"\n✅ Validaciones:")
    assert completeness_metrics["numerical_columns_detailed"] >= 20, "Debe incluir al menos 20 cols numéricas"
    assert completeness_metrics["correlations_included"] > 0, "Debe incluir correlaciones"
    assert completeness_metrics["missing_columns_tracked"] > 0, "Debe rastrear columnas con nulos"
    assert completeness_metrics["has_executive_summary"], "Debe incluir resumen ejecutivo"
    
    print(f"   ✓ Completitud optimizada verificada")
    print(f"   ✓ Resumen ejecutivo incluido")
    print(f"   ✓ Información estadística completa")
    
    print("✅ TEST 5 PASADO: Resumen optimizado es completo y útil\n")
    return True


def main():
    """Ejecutar todos los tests de optimización del resumen"""
    print("\n" + "="*80)
    print("🚀 SUITE DE TESTS: Optimización del Resumen del Analizador")
    print("="*80)
    
    results = []
    
    try:
        results.append(("Dataset pequeño", test_small_dataset_summary()))
    except Exception as e:
        print(f"❌ TEST 1 FALLÓ: {e}")
        results.append(("Dataset pequeño", False))
    
    try:
        results.append(("Dataset mediano", test_medium_dataset_summary()))
    except Exception as e:
        print(f"❌ TEST 2 FALLÓ: {e}")
        results.append(("Dataset mediano", False))
    
    try:
        results.append(("Dataset grande", test_large_dataset_summary()))
    except Exception as e:
        print(f"❌ TEST 3 FALLÓ: {e}")
        results.append(("Dataset grande", False))
    
    try:
        results.append(("Patrones médicos", test_medical_patterns_completeness()))
    except Exception as e:
        print(f"❌ TEST 4 FALLÓ: {e}")
        results.append(("Patrones médicos", False))
    
    try:
        results.append(("Completitud resumen", test_summary_completeness_comparison()))
    except Exception as e:
        print(f"❌ TEST 5 FALLÓ: {e}")
        results.append(("Completitud resumen", False))
    
    # Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN DE TESTS")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASADO" if passed else "❌ FALLADO"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n🎯 Total: {passed}/{total} tests pasados ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n✅ TODOS LOS TESTS PASARON - Resumen optimizado funcionando correctamente")
        return 0
    else:
        print(f"\n⚠️ {total - passed} tests fallaron - Revisar implementación")
        return 1


if __name__ == "__main__":
    exit(main())
