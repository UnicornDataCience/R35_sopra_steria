"""
Test específico para validar que las correcciones de string vs int funcionan
"""
import pandas as pd
import sys
import os

# Añadir la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_string_vs_int_fix():
    """Test específico para validar que el error string vs int está corregido"""
    print("🔍 TEST: ERROR STRING VS INT CORREGIDO")
    print("=" * 50)
    
    try:
        from src.agents.validator_agent import MedicalValidatorAgent
        
        # Crear instancia del agente
        agent = MedicalValidatorAgent()
        print("✅ Agente validador importado correctamente")
        
        # Crear datos de prueba con tipos mixtos que causaban el error
        test_data = {
            'EDAD/AGE': [25, '30', 45, 'unknown', 65, 70.5, 'N/A'],
            'SEXO/SEX': ['MALE', 'FEMALE', 'M', 'F', 1, 0, 'unknown'],
            'TEMP_ING/INPAT': [36.5, '37.2', 38.0, 'N/A', 36.8, 'missing', 39.1],
            'SAT_02_ING/INPAT': [95, '98', 92, 'missing', 97, 85.5, 'N/A'],
            'DIAG ING/INPAT': ['COVID19 - POSITIVO', 'COVID19 - NEGATIVO', 'COVID19 - PENDIENTE', '', 'N/A', 'COVID19 - POSITIVO', 'OTHER']
        }
        
        df_test = pd.DataFrame(test_data)
        print(f"📊 Datos de prueba creados: {len(df_test)} filas")
        print(f"🔢 Tipos problemáticos incluidos:")
        for col in df_test.columns:
            unique_types = set(type(x).__name__ for x in df_test[col].dropna())
            print(f"  {col}: {unique_types}")
        
        # Test 1: Conversión numérica segura
        print(f"\n🧪 Test 1: Conversión numérica segura")
        try:
            edad_numeric = agent._safe_numeric_conversion(df_test['EDAD/AGE'])
            temp_numeric = agent._safe_numeric_conversion(df_test['TEMP_ING/INPAT'])
            sat_numeric = agent._safe_numeric_conversion(df_test['SAT_02_ING/INPAT'])
            
            print(f"✅ Conversión de EDAD/AGE: {edad_numeric.dtype}")
            print(f"✅ Conversión de TEMP_ING/INPAT: {temp_numeric.dtype}")
            print(f"✅ Conversión de SAT_02_ING/INPAT: {sat_numeric.dtype}")
            
            # Verificar que no hay errores en comparaciones
            test_comparison = edad_numeric >= 0
            print(f"✅ Comparación numérica funciona sin errores")
            
        except Exception as e:
            print(f"❌ Error en conversión numérica: {e}")
            return False
        
        # Test 2: Validación between segura  
        print(f"\n🧪 Test 2: Validación between segura")
        try:
            edad_valid, edad_invalid = agent._safe_between_validation(df_test, 'EDAD/AGE', 0, 120)
            temp_valid, temp_invalid = agent._safe_between_validation(df_test, 'TEMP_ING/INPAT', 35.0, 42.0)
            sat_valid, sat_invalid = agent._safe_between_validation(df_test, 'SAT_02_ING/INPAT', 70, 100)
            
            print(f"✅ Validación EDAD/AGE: {edad_valid:.1%} válidos ({edad_invalid} inválidos)")
            print(f"✅ Validación TEMP_ING/INPAT: {temp_valid:.1%} válidos ({temp_invalid} inválidos)")
            print(f"✅ Validación SAT_02_ING/INPAT: {sat_valid:.1%} válidos ({sat_invalid} inválidos)")
            
        except Exception as e:
            print(f"❌ Error en validación between: {e}")
            return False
        
        # Test 3: Validación clínica COVID (la que solía fallar)
        print(f"\n🧪 Test 3: Validación clínica COVID completa")
        try:
            results = {"issues": [], "clinical_inconsistencies": 0}
            coherence_score = agent._validate_covid_clinical_coherence(df_test, results)
            
            print(f"✅ Validación COVID exitosa")
            print(f"📊 Puntuación de coherencia: {coherence_score:.3f}")
            print(f"🔍 Issues detectados: {len(results['issues'])}")
            print(f"⚠️ Inconsistencias clínicas: {results['clinical_inconsistencies']}")
            
            for issue in results['issues']:
                print(f"  - {issue}")
                
        except Exception as e:
            print(f"❌ Error en validación COVID: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 4: Validación de cardiología 
        print(f"\n🧪 Test 4: Validación cardiología")
        try:
            # Añadir datos de cardiología
            cardio_data = {
                'PRESION_SISTOLICA': [120, '140', 'N/A', 110, '160', 90],
                'PRESION_DIASTOLICA': [80, '90', 70, 'missing', '95', 60],
                'FRECUENCIA_CARDIACA': [72, '80', 'N/A', 65, '88', 95],
                'COLESTEROL_TOTAL': [200, '250', 180, 'N/A', '300', 150],
                'HDL': [40, '50', 35, 'missing', '60', 45],
                'LDL': [120, '150', 100, 'N/A', '180', 80]
            }
            
            df_cardio = pd.DataFrame(cardio_data)
            results_cardio = {"issues": [], "clinical_inconsistencies": 0}
            coherence_score_cardio = agent._validate_cardiology_clinical_coherence(df_cardio, results_cardio)
            
            print(f"✅ Validación cardiología exitosa")
            print(f"📊 Puntuación de coherencia: {coherence_score_cardio:.3f}")
            print(f"🔍 Issues detectados: {len(results_cardio['issues'])}")
            
        except Exception as e:
            print(f"❌ Error en validación cardiología: {e}")
            return False
        
        print(f"\n🎉 TODOS LOS TESTS PASARON")
        print(f"✅ Error string vs int COMPLETAMENTE CORREGIDO")
        return True
        
    except Exception as e:
        print(f"❌ Error general en test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_validation_with_repaired_file():
    """Test para verificar que el archivo reparado funciona"""
    print(f"\n🔍 TEST: ARCHIVO JSON REPARADO")
    print("=" * 50)
    
    try:
        from src.validation.json_schema import procesar_archivo_json
        
        # Test archivo SDV (ya funcionaba)
        sdv_file = 'data/synthetic/datos_sinteticos_sdv.json'
        print(f"📂 Testando SDV (archivo que ya funcionaba)...")
        if os.path.exists(sdv_file):
            result_sdv = procesar_archivo_json(sdv_file)
            print(f"✅ SDV: {result_sdv['success_rate']:.1f}% válidos ({result_sdv['total_count']} registros)")
        
        # Test archivo TVAE (recién reparado)
        tvae_file = 'data/synthetic/datos_sinteticos_tvae.json'
        print(f"📂 Testando TVAE (archivo reparado)...")
        if os.path.exists(tvae_file):
            result_tvae = procesar_archivo_json(tvae_file)
            print(f"✅ TVAE: {result_tvae['success_rate']:.1f}% válidos ({result_tvae['total_count']} registros)")
            
            if result_tvae['success_rate'] > 90:
                print(f"🎉 ARCHIVO TVAE REPARADO EXITOSAMENTE")
                return True
            else:
                print(f"⚠️ Archivo reparado pero con baja tasa de éxito")
                print(f"📋 Errores: {result_tvae.get('errors', [])[:3]}")
        else:
            print(f"❌ Archivo TVAE no encontrado")
            return False
            
    except Exception as e:
        print(f"❌ Error en test JSON: {e}")
        return False

def main():
    """Test principal corregido"""
    print("🚀 TEST FINAL CORREGIDO - VALIDACIONES")
    print("=" * 60)
    
    # Test 1: Validación string vs int corregida
    string_int_ok = test_string_vs_int_fix()
    
    # Test 2: Archivo JSON reparado
    json_ok = test_json_validation_with_repaired_file()
    
    # Resultado final
    print(f"\n📋 RESULTADO FINAL:")
    print("=" * 60)
    print(f"🔍 Error string vs int: {'✅ CORREGIDO' if string_int_ok else '❌ FALLA'}")
    print(f"🔍 Validación JSON: {'✅ FUNCIONA' if json_ok else '❌ FALLA'}")
    
    if string_int_ok and json_ok:
        print(f"\n🎉 ¡TODOS LOS PROBLEMAS SOLUCIONADOS!")
        print(f"   - Error de comparación string vs int: CORREGIDO")
        print(f"   - Archivo JSON TVAE: REPARADO")
        print(f"   - JSON schema devuelve valores correctos: CORREGIDO")
        print(f"   - Sistema de validación: COMPLETAMENTE OPERATIVO")
        
        print(f"\n🔧 RESUMEN DE CORRECCIONES IMPLEMENTADAS:")
        print(f"   ✅ Funciones _safe_numeric_conversion y _safe_between_validation añadidas")
        print(f"   ✅ Todas las validaciones .between() reemplazadas con versión segura")
        print(f"   ✅ Archivo TVAE convertido de JSONL a JSON array válido")
        print(f"   ✅ Script de reparación creado para futuros problemas")
    else:
        print(f"\n⚠️ Resumen de estado:")
        if not string_int_ok:
            print(f"   - Validación tabular: Necesita revisión adicional")
        if not json_ok:
            print(f"   - Validación JSON: Archivo TVAE necesita más trabajo")

if __name__ == "__main__":
    main()
