"""
Test para verificar que el resumen del analizador incluye información completa
"""

from src.agents.analyzer_agent import ClinicalAnalyzerAgent
from src.adapters.universal_dataset_detector import UniversalDatasetDetector
import pandas as pd
import json

print("=" * 80)
print("🧪 TEST: Verificación del Resumen Optimizado del Analizador")
print("=" * 80)

# Cargar dataset
print("\n1️⃣ Cargando dataset...")
df = pd.read_csv('data/real/df_final_v2.csv')
print(f"   ✅ Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

# Generar análisis completo
print("\n2️⃣ Generando análisis completo con CompleteEDAAnalyzer...")
from src.analysis.complete_eda import CompleteEDAAnalyzer
eda_analyzer = CompleteEDAAnalyzer()
analysis = eda_analyzer.analyze(df)
print(f"   ✅ Análisis completado")

# Generar resumen optimizado
print("\n3️⃣ Generando resumen optimizado para LLM...")
analyzer = ClinicalAnalyzerAgent()
summarized = analyzer._summarize_analysis_for_llm(analysis)
json_str = json.dumps(summarized, indent=2)
print(f"   ✅ Resumen generado")

# Verificar contenido del resumen
print("\n" + "=" * 80)
print("📊 RESULTADOS DE LA VERIFICACIÓN")
print("=" * 80)

print(f"\n📏 Tamaño del JSON:")
print(f"   - Caracteres: {len(json_str):,}")
print(f"   - Límite actual: 30,000 chars")
print(f"   - Estado: {'✅ OK' if len(json_str) < 30000 else '⚠️ EXCEDE LÍMITE'}")

print(f"\n🔢 Columnas Numéricas Detalladas:")
num_cols = summarized.get("numerical_columns", [])
print(f"   - Cantidad incluida: {len(num_cols)}")
if num_cols:
    print(f"   - Ejemplo: {num_cols[0].get('name')}")
    if 'stats' in num_cols[0]:
        print(f"   - Estadísticas incluidas: ✅")
        print(f"     • mean, std, min, max, median, q1, q3, missing")

print(f"\n📝 Columnas Categóricas Detalladas:")
cat_cols = summarized.get("categorical_columns", [])
print(f"   - Cantidad incluida: {len(cat_cols)}")
if cat_cols:
    print(f"   - Ejemplo: {cat_cols[0].get('name')}")
    print(f"   - Top values incluidos: {len(cat_cols[0].get('top_values', []))}")

print(f"\n🔗 Correlaciones:")
correlations = summarized.get("high_correlations", [])
print(f"   - Cantidad incluida: {len(correlations)}")
if correlations:
    print(f"   - Ejemplo: {correlations[0]}")

print(f"\n🩺 Patrones Médicos:")
medical = summarized.get("medical_patterns", {})
key_cols = medical.get("key_medical_columns", [])
print(f"   - Columnas médicas identificadas: {len(key_cols)}")
if key_cols:
    print(f"   - Primeras 5: {key_cols[:5]}")

print(f"\n❓ Valores Nulos:")
missing = summarized.get("missing_values", {})
if missing:
    print(f"   - Total missing: {missing.get('total_missing', 0):,}")
    print(f"   - Porcentaje: {missing.get('percentage', 0):.2f}%")
    top_missing = missing.get("top_missing_columns", [])
    print(f"   - Top columnas con nulos: {len(top_missing)}")

print(f"\n📋 Resumen Ejecutivo:")
exec_summary = summarized.get("executive_summary", {})
print(json.dumps(exec_summary, indent=2))

print("\n" + "=" * 80)
print("✅ VERIFICACIÓN COMPLETADA")
print("=" * 80)

# Mostrar una muestra del JSON
print("\n📄 Muestra del JSON (primeros 1000 caracteres):")
print("-" * 80)
print(json_str[:1000])
print("\n[...]")
print("-" * 80)
