"""
Test para verificar qué está devolviendo el UniversalDatasetDetector
"""

from src.adapters.universal_dataset_detector import UniversalDatasetDetector
import pandas as pd
import json

print("=" * 80)
print("🔍 DEBUG: Análisis del UniversalDatasetDetector")
print("=" * 80)

# Cargar dataset
print("\n1️⃣ Cargando dataset...")
df = pd.read_csv('data/real/df_final_v2.csv')
print(f"   ✅ Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

# Generar análisis
print("\n2️⃣ Generando análisis...")
detector = UniversalDatasetDetector()
analysis = detector.analyze_dataset(df)

# Mostrar las keys del análisis
print("\n📋 Keys del análisis:")
for key in analysis.keys():
    print(f"   - {key}")

# Mostrar detalles de cada sección
print("\n" + "=" * 80)
print("📊 CONTENIDO DEL ANÁLISIS")
print("=" * 80)

# Basic info
if "basic_info" in analysis:
    print("\n✅ basic_info:")
    print(json.dumps(analysis["basic_info"], indent=2))

# Columns
if "columns" in analysis:
    print(f"\n✅ columns: {len(analysis['columns'])} columnas")
    if analysis['columns']:
        print(f"   Ejemplo primera columna:")
        print(json.dumps(analysis["columns"][0], indent=2, default=str))
else:
    print("\n❌ columns: NO PRESENTE")

# Correlations
if "correlations" in analysis:
    print(f"\n✅ correlations:")
    print(json.dumps(analysis["correlations"], indent=2))
else:
    print("\n❌ correlations: NO PRESENTE")

# Missing values
if "missing_values" in analysis:
    print(f"\n✅ missing_values:")
    print(json.dumps(analysis["missing_values"], indent=2))
else:
    print("\n❌ missing_values: NO PRESENTE")

# Medical patterns
if "medical_patterns" in analysis:
    print(f"\n✅ medical_patterns:")
    print(json.dumps(analysis["medical_patterns"], indent=2))
else:
    print("\n❌ medical_patterns: NO PRESENTE")

print("\n" + "=" * 80)
print("✅ DEBUG COMPLETADO")
print("=" * 80)
