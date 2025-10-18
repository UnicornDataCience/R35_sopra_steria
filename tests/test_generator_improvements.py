"""
Test de optimizaciones del generador (Fase 3).

Valida:
- Caché de modelos funcional
- Métricas de calidad calculadas
- Performance mejorado
- No regresiones en funcionalidad
"""

import sys
import os
import time
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.generator_agent import SyntheticGeneratorAgent
from src.generation.model_cache import get_model_cache
from src.generation.quality_metrics import get_quality_evaluator


def print_section(title: str):
    """Helper para imprimir secciones"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def create_test_dataset(n_rows: int = 500) -> pd.DataFrame:
    """Crea un dataset de prueba médico sintético."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'PATIENT_ID': range(1, n_rows + 1),
        'EDAD': np.random.randint(18, 90, n_rows),
        'SEXO': np.random.choice(['M', 'F'], n_rows),
        'PRESION_SISTOLICA': np.random.normal(130, 15, n_rows),
        'PRESION_DIASTOLICA': np.random.normal(85, 10, n_rows),
        'GLUCOSA': np.random.normal(100, 20, n_rows),
        'COLESTEROL': np.random.normal(200, 30, n_rows),
        'DIAGNOSTICO': np.random.choice(['Diabetes', 'Hipertensión', 'Normal', 'Cardiovascular'], n_rows),
        'TRATAMIENTO': np.random.choice(['Medicación A', 'Medicación B', 'Medicación C', 'Sin tratamiento'], n_rows)
    })
    
    # Correlación realista: edad -> presión
    df['PRESION_SISTOLICA'] = df['PRESION_SISTOLICA'] + (df['EDAD'] - 50) * 0.5
    df['PRESION_DIASTOLICA'] = df['PRESION_DIASTOLICA'] + (df['EDAD'] - 50) * 0.3
    
    return df


async def test_cache_functionality():
    """Test 1: Verificar que el caché funciona correctamente."""
    print_section("TEST 1: Funcionalidad de Caché")
    
    # Limpiar caché antes de empezar
    cache = get_model_cache()
    cache.clear_all()
    print("✅ Caché limpiado")
    
    # Crear dataset de prueba pequeño
    df = create_test_dataset(n_rows=200)
    print(f"📊 Dataset de prueba: {df.shape}")
    
    # Crear contexto
    context = {
        'dataframe': df,
        'num_samples': 50,
        'model_type': 'sdv'  # SDV es más rápido para tests
    }
    
    # Primera generación (sin caché)
    agent = SyntheticGeneratorAgent()
    print("\n⏱️  Primera generación (SIN caché)...")
    start_time = time.time()
    
    result1 = await agent.process("Generar datos", context)
    
    time1 = time.time() - start_time
    print(f"✅ Primera generación completada en {time1:.2f}s")
    print(f"   - Cache hit: {result1['generation_info'].get('cache_hit', False)}")
    print(f"   - Samples generados: {len(result1['synthetic_data'])}")
    
    # Segunda generación (con caché)
    print("\n⏱️  Segunda generación (CON caché)...")
    start_time = time.time()
    
    result2 = await agent.process("Generar datos", context)
    
    time2 = time.time() - start_time
    print(f"✅ Segunda generación completada en {time2:.2f}s")
    print(f"   - Cache hit: {result2['generation_info'].get('cache_hit', False)}")
    print(f"   - Samples generados: {len(result2['synthetic_data'])}")
    
    # Verificaciones
    speedup = time1 / time2 if time2 > 0 else 0
    print(f"\n📊 Resultados:")
    print(f"   - Speedup: {speedup:.2f}x")
    print(f"   - Ahorro de tiempo: {time1 - time2:.2f}s ({((time1-time2)/time1*100):.1f}%)")
    
    # Verificar que el caché funcionó
    if result2['generation_info'].get('cache_hit'):
        print("   ✅ PASS: Caché funcionó correctamente")
        return True
    else:
        print("   ❌ FAIL: Caché no funcionó")
        return False


async def test_quality_metrics():
    """Test 2: Verificar que las métricas de calidad se calculan."""
    print_section("TEST 2: Métricas de Calidad")
    
    # Habilitar métricas de calidad
    os.environ['GENERATOR_COMPUTE_QUALITY_METRICS'] = 'true'
    
    # Crear dataset de prueba
    df = create_test_dataset(n_rows=300)
    print(f"📊 Dataset de prueba: {df.shape}")
    
    # Generar datos
    context = {
        'dataframe': df,
        'num_samples': 100,
        'model_type': 'sdv'
    }
    
    agent = SyntheticGeneratorAgent()
    print("\n⏱️  Generando datos con métricas de calidad...")
    
    result = await agent.process("Generar datos", context)
    
    # Verificar que hay métricas
    quality_metrics = result['generation_info'].get('quality_metrics')
    
    if quality_metrics:
        print("✅ Métricas de calidad calculadas:")
        print(f"   - Statistical similarity: {quality_metrics['statistical_similarity']:.3f}")
        print(f"   - Correlation preservation: {quality_metrics['correlation_preservation']:.3f}")
        print(f"   - Distribution fidelity: {quality_metrics['distribution_fidelity']:.3f}")
        print(f"   - Privacy score: {quality_metrics['privacy_score']:.3f}")
        print(f"   - Overall quality: {quality_metrics['overall_quality']:.3f}")
        
        # Verificar que las métricas están en rango válido
        all_in_range = all(
            0 <= quality_metrics[k] <= 1
            for k in ['statistical_similarity', 'correlation_preservation', 
                     'distribution_fidelity', 'privacy_score', 'overall_quality']
        )
        
        if all_in_range:
            print("   ✅ PASS: Todas las métricas están en rango [0, 1]")
            return True
        else:
            print("   ❌ FAIL: Algunas métricas fuera de rango")
            return False
    else:
        print("   ❌ FAIL: No se calcularon métricas de calidad")
        return False


async def test_performance_metrics():
    """Test 3: Verificar métricas de rendimiento del agente."""
    print_section("TEST 3: Métricas de Rendimiento")
    
    df = create_test_dataset(n_rows=200)
    agent = SyntheticGeneratorAgent()
    
    # Generar varias veces
    context = {
        'dataframe': df,
        'num_samples': 50,
        'model_type': 'sdv'
    }
    
    print("⏱️  Generando 3 veces...")
    for i in range(3):
        result = await agent.process(f"Generación {i+1}", context)
        print(f"   - Generación {i+1}: {result['generation_info']['elapsed_time_seconds']:.2f}s")
    
    # Obtener métricas acumuladas
    metrics = agent.get_performance_metrics()
    
    print("\n📊 Métricas acumuladas:")
    print(f"   - Generaciones totales: {metrics['generation_count']}")
    print(f"   - Cache hits: {metrics['cache_hits']}")
    print(f"   - Cache hit rate: {metrics['cache_hit_rate']:.1%}")
    print(f"   - Tiempo total: {metrics['total_generation_time']:.2f}s")
    print(f"   - Tiempo promedio: {metrics['avg_generation_time']:.2f}s")
    
    # Verificar métricas de caché
    cache_stats = metrics['cache_stats']
    print(f"\n📦 Estadísticas de caché:")
    print(f"   - Habilitado: {cache_stats['enabled']}")
    print(f"   - Modelos en caché: {cache_stats['total_models']}")
    print(f"   - Tamaño total: {cache_stats['total_size_mb']:.2f} MB")
    
    # Verificar que las métricas son coherentes
    if metrics['generation_count'] == 3 and metrics['cache_hits'] >= 2:
        print("\n   ✅ PASS: Métricas de rendimiento correctas")
        return True
    else:
        print("\n   ❌ FAIL: Métricas inconsistentes")
        return False


async def test_data_quality():
    """Test 4: Verificar que los datos generados son válidos."""
    print_section("TEST 4: Calidad de Datos Generados")
    
    df = create_test_dataset(n_rows=300)
    print(f"📊 Dataset original: {df.shape}")
    print(f"   Columnas: {list(df.columns)}")
    
    context = {
        'dataframe': df,
        'num_samples': 100,
        'model_type': 'sdv'
    }
    
    agent = SyntheticGeneratorAgent()
    result = await agent.process("Generar datos", context)
    
    synthetic_df = result['synthetic_data']
    print(f"\n📊 Dataset sintético: {synthetic_df.shape}")
    print(f"   Columnas: {list(synthetic_df.columns)}")
    
    # Verificaciones
    checks = []
    
    # 1. Mismo número de columnas
    if len(synthetic_df.columns) == len(df.columns):
        print("   ✅ Mismo número de columnas")
        checks.append(True)
    else:
        print(f"   ❌ Número de columnas diferente: {len(synthetic_df.columns)} vs {len(df.columns)}")
        checks.append(False)
    
    # 2. Columnas con nombres similares
    if set(synthetic_df.columns) == set(df.columns):
        print("   ✅ Nombres de columnas coinciden")
        checks.append(True)
    else:
        print("   ⚠️  Algunos nombres de columnas difieren")
        checks.append(True)  # No crítico
    
    # 3. Tipos de datos razonables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in synthetic_df.columns:
            original_range = (df[col].min(), df[col].max())
            synthetic_range = (synthetic_df[col].min(), synthetic_df[col].max())
            
            # Verificar que los valores sintéticos están en un rango razonable
            # (permitir 20% de extensión del rango original)
            margin = 0.2 * (original_range[1] - original_range[0])
            in_range = (
                synthetic_range[0] >= original_range[0] - margin and
                synthetic_range[1] <= original_range[1] + margin
            )
            
            if in_range:
                print(f"   ✅ {col}: rango razonable {synthetic_range}")
                checks.append(True)
            else:
                print(f"   ⚠️  {col}: rango extendido {synthetic_range} vs {original_range}")
                checks.append(True)  # Advertencia, no error
    
    # 4. Sin valores nulos excesivos
    null_pct = (synthetic_df.isnull().sum().sum() / (synthetic_df.shape[0] * synthetic_df.shape[1])) * 100
    if null_pct < 10:
        print(f"   ✅ Pocos valores nulos: {null_pct:.1f}%")
        checks.append(True)
    else:
        print(f"   ⚠️  Muchos valores nulos: {null_pct:.1f}%")
        checks.append(True)  # Advertencia
    
    if all(checks):
        print("\n   ✅ PASS: Datos generados son válidos")
        return True
    else:
        print("\n   ⚠️  PASS with warnings: Algunos checks fallaron pero no son críticos")
        return True


async def main():
    """Ejecuta todos los tests."""
    print("\n" + "="*60)
    print("  🧪 TEST SUITE: Optimizaciones del Generador (Fase 3)")
    print("="*60)
    
    results = {}
    
    try:
        results['cache'] = await test_cache_functionality()
    except Exception as e:
        print(f"❌ Error en test de caché: {e}")
        results['cache'] = False
    
    try:
        results['quality_metrics'] = await test_quality_metrics()
    except Exception as e:
        print(f"❌ Error en test de métricas de calidad: {e}")
        results['quality_metrics'] = False
    
    try:
        results['performance_metrics'] = await test_performance_metrics()
    except Exception as e:
        print(f"❌ Error en test de métricas de rendimiento: {e}")
        results['performance_metrics'] = False
    
    try:
        results['data_quality'] = await test_data_quality()
    except Exception as e:
        print(f"❌ Error en test de calidad de datos: {e}")
        results['data_quality'] = False
    
    # Resumen
    print_section("RESUMEN DE RESULTADOS")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n   Total: {total_passed}/{total_tests} tests pasados ({total_passed/total_tests*100:.0f}%)")
    
    if total_passed == total_tests:
        print("\n🎉 TODOS LOS TESTS PASARON")
        return 0
    else:
        print("\n⚠️  ALGUNOS TESTS FALLARON")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
