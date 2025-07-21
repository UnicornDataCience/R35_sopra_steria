"""
Test de diagnóstico para validación tabular - Identifica el error de tipo en comparaciones
"""
import pandas as pd
import sys
import os

# Añadir la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.tabular_medical_validator import TabularMedicalValidator

def test_tabular_validation_debug():
    """Test diagnóstico para encontrar el error de validación tabular"""
    print("🔍 DIAGNÓSTICO DE VALIDACIÓN TABULAR")
    print("=" * 50)
    
    # Test 1: Cargar datos reales
    data_files = [
        'data/real/diabetes.csv',
        'data/real/cardiology_fict_data.csv',
        'data/real/df_final_v2.csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"\n📂 Testando archivo: {file_path}")
            try:
                df = pd.read_csv(file_path)
                print(f"✅ Cargado: {len(df)} filas, {len(df.columns)} columnas")
                print(f"📋 Columnas: {list(df.columns)}")
                
                # Mostrar tipos de datos
                print(f"🔢 Tipos de datos:")
                for col in df.columns[:5]:  # Solo mostrar primeras 5 columnas
                    sample_values = df[col].dropna().head(3).tolist()
                    print(f"  {col}: {df[col].dtype} | Ejemplos: {sample_values}")
                
                # Intentar validación paso a paso
                validator = TabularMedicalValidator()
                
                print(f"\n🧪 Ejecutando validación...")
                
                # Test específico: revisar conversiones seguras
                print(f"🔧 Test de conversiones seguras:")
                for col in df.columns[:3]:
                    try:
                        numeric_result = validator._safe_numeric_conversion(df[col])
                        string_result = validator._safe_string_conversion(df[col])
                        print(f"  {col}: numeric OK, string OK")
                    except Exception as e:
                        print(f"  {col}: ❌ Error en conversión: {e}")
                
                # Intentar validación completa
                results = validator.validate_data_quality(df)
                print(f"✅ Validación exitosa!")
                print(f"📊 Puntuación general: {results['overall_quality_score']:.3f}")
                print(f"🔍 Issues encontrados: {len(results['issues'])}")
                for issue in results['issues'][:3]:
                    print(f"  - {issue}")
                
            except Exception as e:
                print(f"❌ Error en {file_path}: {e}")
                import traceback
                print(f"🔍 Traceback completo:")
                traceback.print_exc()
                
                # Análisis específico del error
                if ">=" in str(e) and "str" in str(e) and "int" in str(e):
                    print(f"\n🎯 ERROR IDENTIFICADO: Comparación de string vs int")
                    print(f"   - El error sugiere que hay valores string siendo comparados con números")
                    print(f"   - Revisar las funciones _safe_numeric_conversion")
                    
                    # Mostrar datos problemáticos
                    try:
                        df = pd.read_csv(file_path)
                        for col in df.columns:
                            if col.lower() in validator.clinical_ranges:
                                range_config = validator.clinical_ranges[col.lower()]
                                if range_config['type'] == 'numeric':
                                    print(f"   📋 Columna {col}: {df[col].dtype}")
                                    unique_values = df[col].dropna().unique()[:5]
                                    print(f"      Valores únicos: {unique_values}")
                                    print(f"      Tipos: {[type(v).__name__ for v in unique_values]}")
                    except:
                        pass
                
            print("-" * 50)
        else:
            print(f"⚠️  Archivo no encontrado: {file_path}")
    
    # Test 2: Crear datos de prueba controlados
    print(f"\n🧪 TEST CON DATOS CONTROLADOS")
    print("=" * 50)
    
    # Datos mixtos (strings y números) para reproducir el error
    test_data = {
        'age': [25, '30', 45, 'unknown', 65],
        'sex': [0, 1, '1', 'M', 'F'],
        'diabetes': [0, 1, '0', '1', 'yes'],
        'hypertension': [1, 0, '1', '0', 'no']
    }
    
    df_test = pd.DataFrame(test_data)
    print(f"📊 Datos de prueba creados:")
    print(df_test)
    print(f"🔢 Tipos de datos:")
    print(df_test.dtypes)
    
    try:
        validator = TabularMedicalValidator()
        results = validator.validate_data_quality(df_test)
        print(f"✅ Test con datos mixtos: OK")
        print(f"📊 Puntuación: {results['overall_quality_score']:.3f}")
    except Exception as e:
        print(f"❌ Error con datos de prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tabular_validation_debug()
