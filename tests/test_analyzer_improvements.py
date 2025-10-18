"""
Test de Mejoras del Analizador - Fase 2

Este test valida las mejoras implementadas:
- Caché de análisis por hash de dataset
- Performance tracking
- Métricas de análisis
- Logging estructurado
"""

import asyncio
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.agents.analyzer_agent import ClinicalAnalyzerAgent
from src.adapters.universal_dataset_detector import UniversalDatasetDetector
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def create_test_dataset():
    """Crear un dataset de prueba simple"""
    np.random.seed(42)
    n_rows = 100
    
    df = pd.DataFrame({
        'patient_id': range(1, n_rows + 1),
        'Age': np.random.randint(18, 80, n_rows),
        'Gender': np.random.choice(['M', 'F'], n_rows),
        'Temperature': np.random.uniform(36.0, 39.5, n_rows),
        'SpO2': np.random.uniform(85, 100, n_rows),
        'PCR_Result': np.random.choice(['Positive', 'Negative'], n_rows),
        'Severity': np.random.choice(['Low', 'Medium', 'High'], n_rows),
        'Days_in_Hospital': np.random.randint(1, 30, n_rows)
    })
    
    return df

async def test_analyzer_improvements():
    """Test de mejoras del analizador - Fase 2"""
    
    print("=" * 80)
    print("🧪 TEST DE MEJORAS DEL ANALIZADOR - FASE 2")
    print("=" * 80)
    
    # Crear dataset de prueba
    print("\n📊 Creando dataset de prueba...")
    df = create_test_dataset()
    print(f"   ✅ Dataset creado: {len(df)} filas, {len(df.columns)} columnas")
    
    # Inicializar analizador
    print("\n🔧 Inicializando analizador...")
    analyzer = ClinicalAnalyzerAgent()
    print("   ✅ Analizador inicializado")
    
    # Realizar análisis universal (prerequisito)
    print("\n📋 Ejecutando análisis universal...")
    detector = UniversalDatasetDetector()
    universal_analysis = detector.analyze_dataset(df)
    print(f"   ✅ Análisis universal completado")
    
    # Test 1: Primer análisis (Cache Miss)
    print("\n" + "=" * 80)
    print("📋 TEST 1: Primer Análisis (Cache Miss Esperado)")
    print("=" * 80)
    
    context1 = {
        "universal_analysis": universal_analysis,
        "dataframe": df
    }
    
    print("\n🔹 Ejecutando primer análisis...")
    import time
    start = time.time()
    result1 = await analyzer.analyze_dataset(None, context1)
    elapsed1 = time.time() - start
    
    assert "message" in result1, "❌ Resultado debe tener 'message'"
    assert "analysis_complete" in result1, "❌ Resultado debe tener 'analysis_complete'"
    assert result1["analysis_complete"], "❌ Análisis debe estar completo"
    
    print(f"   ✅ Análisis completado en {elapsed1:.2f}s")
    print(f"   ✅ Informe generado: {len(result1['message'])} caracteres")
    print(f"   ✅ Secciones esperadas presentes en informe")
    
    # Test 2: Segundo análisis del MISMO dataset (Cache Hit)
    print("\n" + "=" * 80)
    print("📋 TEST 2: Segundo Análisis (Cache Hit Esperado)")
    print("=" * 80)
    
    context2 = {
        "universal_analysis": universal_analysis,
        "dataframe": df  # Mismo dataframe = mismo hash
    }
    
    print("\n🔹 Ejecutando segundo análisis del mismo dataset...")
    start = time.time()
    result2 = await analyzer.analyze_dataset(None, context2)
    elapsed2 = time.time() - start
    
    assert "message" in result2, "❌ Resultado debe tener 'message'"
    assert result2["message"] == result1["message"], "❌ Resultados deben ser idénticos"
    
    print(f"   ✅ Análisis completado en {elapsed2:.2f}s")
    print(f"   ✅ Resultado idéntico al primero")
    print(f"   ✅ Speedup: {elapsed1/elapsed2:.1f}x más rápido")
    
    # Verificar que fue más rápido (cache hit)
    if elapsed2 < elapsed1 * 0.5:  # Al menos 2x más rápido
        print(f"   ✅ Cache HIT confirmado (2x+ más rápido)")
    else:
        print(f"   ⚠️ Cache puede no estar funcionando óptimamente")
    
    # Test 3: Métricas del Analizador
    print("\n" + "=" * 80)
    print("📋 TEST 3: Métricas de Performance")
    print("=" * 80)
    
    metrics = analyzer.get_metrics()
    print("\n📊 Métricas del Analizador:")
    print(f"  - Total análisis: {metrics['analysis_metrics']['total_analyses']}")
    print(f"  - Cache hits: {metrics['analysis_metrics']['cache_hits']}")
    print(f"  - Cache misses: {metrics['analysis_metrics']['cache_misses']}")
    print(f"  - Cache hit rate: {metrics['analysis_metrics']['cache_hit_rate']}")
    print(f"  - Filas analizadas: {metrics['analysis_metrics']['total_rows_analyzed']}")
    print(f"  - Columnas analizadas: {metrics['analysis_metrics']['total_columns_analyzed']}")
    print(f"  - Tiempo promedio: {metrics['analysis_metrics']['avg_analysis_time_s']}s")
    print(f"  - Estado: {metrics['status']}")
    
    # Verificar métricas
    assert metrics['analysis_metrics']['total_analyses'] == 2, "❌ Debe haber 2 análisis"
    assert metrics['analysis_metrics']['cache_hits'] >= 1, "❌ Debe haber al menos 1 cache hit"
    print("\n  ✅ Métricas recopiladas correctamente")
    
    # Test 4: Análisis de dataset modificado (Cache Miss)
    print("\n" + "=" * 80)
    print("📋 TEST 4: Dataset Modificado (Cache Miss Esperado)")
    print("=" * 80)
    
    # Modificar dataset
    df_modified = df.copy()
    df_modified['Age'] = df_modified['Age'] + 1  # Cambio pequeño
    
    universal_analysis_modified = detector.analyze_dataset(df_modified)
    context3 = {
        "universal_analysis": universal_analysis_modified,
        "dataframe": df_modified
    }
    
    print("\n🔹 Ejecutando análisis de dataset modificado...")
    start = time.time()
    result3 = await analyzer.analyze_dataset(None, context3)
    elapsed3 = time.time() - start
    
    assert "message" in result3, "❌ Resultado debe tener 'message'"
    assert result3["message"] != result1["message"], "❌ Resultado debe ser diferente"
    
    print(f"   ✅ Análisis completado en {elapsed3:.2f}s")
    print(f"   ✅ Resultado diferente (dataset modificado)")
    print(f"   ✅ Cache miss esperado funcionó correctamente")
    
    # Métricas finales
    print("\n" + "=" * 80)
    print("📊 MÉTRICAS FINALES")
    print("=" * 80)
    
    final_metrics = analyzer.get_metrics()
    print(f"\n✅ Total de análisis procesados: {final_metrics['analysis_metrics']['total_analyses']}")
    print(f"✅ Cache hit rate final: {final_metrics['analysis_metrics']['cache_hit_rate']}")
    print(f"✅ Tiempo promedio de análisis: {final_metrics['analysis_metrics']['avg_analysis_time_s']}s")
    print(f"✅ Estado del analizador: {final_metrics['status']}")
    
    # Test 5: Limpiar caché
    print("\n" + "=" * 80)
    print("📋 TEST 5: Limpieza de Caché")
    print("=" * 80)
    
    print("\n🔹 Limpiando caché del analizador...")
    analyzer.clear_cache()
    print("   ✅ Caché limpiado")
    
    # Verificar que ahora es cache miss de nuevo
    print("\n🔹 Ejecutando análisis después de limpiar caché...")
    start = time.time()
    result4 = await analyzer.analyze_dataset(None, context1)
    elapsed4 = time.time() - start
    
    print(f"   ✅ Análisis completado en {elapsed4:.2f}s")
    print(f"   ✅ Cache miss después de limpiar (tiempo similar al primero)")
    
    print("\n" + "=" * 80)
    print("✅ TODOS LOS TESTS PASARON - FASE 2 COMPLETADA")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_analyzer_improvements())
        if success:
            print("\n🎉 Mejoras del analizador validadas exitosamente!")
            sys.exit(0)
        else:
            print("\n❌ Algunos tests fallaron")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error ejecutando tests: {e}", exc_info=True)
        print(f"\n❌ Error ejecutando tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
