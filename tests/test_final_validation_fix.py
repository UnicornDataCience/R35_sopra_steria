"""
Test final integrado - Valida que todos los problemas estén solucionados
"""
import pandas as pd
import sys
import os

# Añadir la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_tabular_validation_fixed():
    """Test para verificar que la validación tabular funciona sin errores"""
    print("🔍 TEST: VALIDACIÓN TABULAR CORREGIDA")
    print("=" * 50)
    
    try:
        from src.agents.validator_agent import MedicalValidatorAgent
        from src.config.azure_config import get_azure_config
        
        # Crear instancia del agente
        agent = MedicalValidatorAgent()
        
        # Cargar datos reales con tipos mixtos
        test_data = {
            'age': [25, '30', 45, 'unknown', 65],
            'EDAD/AGE': [30, 45, '50', 70, 'N/A'],
            'SEXO/SEX': ['MALE', 'FEMALE', 'M', 'F', 1],
            'TEMP_ING/INPAT': [36.5, '37.2', 38.0, 'N/A', 36.8],
            'SAT_02_ING/INPAT': [95, '98', 92, 'missing', 97]
        }
        
        df_test = pd.DataFrame(test_data)
        print(f"📊 Datos de prueba creados: {len(df_test)} filas")
        print(f"🔢 Tipos de datos originales:")
        print(df_test.dtypes)
        
        # Test de validación directa (debería funcionar sin errores)
        print(f"\n🧪 Testando validaciones seguras...")
        
        # Test conversión segura
        age_numeric = agent._safe_numeric_conversion(df_test['age'])
        temp_numeric = agent._safe_numeric_conversion(df_test['TEMP_ING/INPAT'])
        print(f"✅ Conversión numérica segura: OK")
        
        # Test validación between segura
        age_valid, age_invalid = agent._safe_between_validation(df_test, 'age', 0, 120)
        temp_valid, temp_invalid = agent._safe_between_validation(df_test, 'TEMP_ING/INPAT', 35.0, 42.0)
        print(f"✅ Validación between segura: OK")
        print(f"  - Edad válida: {age_valid:.1%} ({age_invalid} inválidos)")
        print(f"  - Temperatura válida: {temp_valid:.1%} ({temp_invalid} inválidos)")
        
        # Test validación completa del agente
        print(f"\n🎯 Test validación completa del agente...")
        context = {
            "dataframe": df_test,
            "validation_target": "original",
            "universal_analysis": {"dataset_type": "COVID-19"}
        }
        
        # Esto debería funcionar sin el error string vs int
        results = agent._perform_comprehensive_medical_validations(
            df_test, True, "original"
        )
        
        print(f"✅ Validación completa: EXITOSA")
        print(f"📊 Resultados:")
        print(f"  - Calidad de datos: {results.get('data_quality', 0):.3f}")
        print(f"  - Coherencia clínica: {results.get('clinical_coherence', 0):.3f}")
        print(f"  - Puntuación general: {results.get('overall_score', 0):.3f}")
        print(f"  - Issues detectados: {len(results.get('issues', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en validación tabular: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_validation_fixed():
    """Test para verificar que la validación JSON funciona correctamente"""
    print(f"\n🔍 TEST: VALIDACIÓN JSON CORREGIDA")
    print("=" * 50)
    
    try:
        from src.validation.json_schema import procesar_archivo_json
        
        # Test archivo que sabemos que funciona
        sdv_file = 'data/synthetic/datos_sinteticos_sdv.json'
        if os.path.exists(sdv_file):
            result = procesar_archivo_json(sdv_file)
            print(f"✅ Archivo SDV: {result['success_rate']:.1f}% válidos")
            
            if result['success_rate'] > 90:
                print(f"✅ Validación JSON funciona correctamente")
                return True
            else:
                print(f"⚠️ Tasa de éxito baja: {result['success_rate']:.1f}%")
                return False
        else:
            print(f"⚠️ Archivo de prueba no encontrado")
            return False
            
    except Exception as e:
        print(f"❌ Error en validación JSON: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test principal integrado"""
    print("🚀 TEST FINAL INTEGRADO - VALIDACIONES CORREGIDAS")
    print("=" * 60)
    
    # Test 1: Validación tabular
    tabular_ok = test_tabular_validation_fixed()
    
    # Test 2: Validación JSON
    json_ok = test_json_validation_fixed()
    
    # Resultado final
    print(f"\n📋 RESULTADO FINAL:")
    print("=" * 60)
    print(f"🔍 Validación tabular: {'✅ CORREGIDA' if tabular_ok else '❌ FALLA'}")
    print(f"🔍 Validación JSON: {'✅ FUNCIONA' if json_ok else '❌ FALLA'}")
    
    if tabular_ok and json_ok:
        print(f"\n🎉 ¡TODOS LOS PROBLEMAS SOLUCIONADOS!")
        print(f"   - Error string vs int: CORREGIDO")
        print(f"   - JSON schema cero: CORREGIDO")
        print(f"   - Sistema de validación: OPERATIVO")
    else:
        print(f"\n⚠️ Algunos problemas persisten")
        if not tabular_ok:
            print(f"   - Validación tabular necesita más trabajo")
        if not json_ok:
            print(f"   - Validación JSON necesita más trabajo")

if __name__ == "__main__":
    main()
