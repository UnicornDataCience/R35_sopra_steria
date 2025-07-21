"""
Test de diagnóstico para validación JSON - Investiga por qué siempre devuelve 0
"""
import json
import sys
import os

# Añadir la raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validation.json_schema import validate_medical_data, procesar_archivo_json, detect_domain_from_data

def test_json_validation_debug():
    """Test diagnóstico para encontrar por qué JSON schema devuelve 0"""
    print("🔍 DIAGNÓSTICO DE VALIDACIÓN JSON")
    print("=" * 50)
    
    # Test 1: Cargar y analizar archivos sintéticos
    synthetic_files = [
        'data/synthetic/datos_sinteticos_sdv.json',
        'data/synthetic/datos_sinteticos_tvae.json',
        'data/synthetic/datos_sinteticos_ctgan.json'
    ]
    
    for file_path in synthetic_files:
        if os.path.exists(file_path):
            print(f"\n📂 Analizando archivo: {file_path}")
            try:
                # Cargar datos manualmente
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"✅ Datos cargados: {len(data) if isinstance(data, list) else 1} registros")
                
                # Analizar estructura
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
                    print(f"📋 Campos del primer registro: {list(sample.keys())}")
                    print(f"🔍 Valores de ejemplo:")
                    for key, value in list(sample.items())[:5]:
                        print(f"  {key}: {value} (tipo: {type(value).__name__})")
                    
                    # Detectar dominio
                    detected_domain = detect_domain_from_data(data)
                    print(f"🎯 Dominio detectado: {detected_domain}")
                    
                    # Test validación manual (registro por registro)
                    print(f"\n🧪 Test manual de validación:")
                    
                    # Validar primeros 3 registros manualmente
                    valid_count = 0
                    invalid_count = 0
                    for i, record in enumerate(data[:3]):
                        try:
                            result = validate_medical_data(record, detected_domain)
                            if result['success_rate'] > 0:
                                valid_count += 1
                                print(f"  Registro {i+1}: ✅ Válido ({result['success_rate']:.1f}%)")
                            else:
                                invalid_count += 1
                                print(f"  Registro {i+1}: ❌ Inválido")
                                if result['errors']:
                                    print(f"    Errores: {result['errors'][:2]}")
                        except Exception as e:
                            invalid_count += 1
                            print(f"  Registro {i+1}: ❌ Error: {e}")
                    
                    print(f"📊 Resultado manual: {valid_count} válidos, {invalid_count} inválidos")
                    
                    # Test con función procesar_archivo_json
                    print(f"\n🔄 Test con función procesar_archivo_json:")
                    result = procesar_archivo_json(file_path)
                    print(f"📊 Resultado oficial:")
                    print(f"  - Total: {result['total_count']}")
                    print(f"  - Válidos: {result['valid_count']}")
                    print(f"  - Inválidos: {result['invalid_count']}")
                    print(f"  - Tasa de éxito: {result['success_rate']:.1f}%")
                    print(f"  - Dominio: {result['domain']}")
                    
                    if result['success_rate'] == 0 and result['total_count'] > 0:
                        print(f"🎯 PROBLEMA IDENTIFICADO: Success rate = 0 pero hay {result['total_count']} registros")
                        print(f"📋 Errores reportados: {result['errors'][:3]}")
                        
                        # Análisis del esquema usado
                        if 'schema_used' in result:
                            schema = result['schema_used']
                            print(f"🔍 Esquema usado:")
                            print(f"  - Campos requeridos: {schema.get('required', [])}")
                            print(f"  - Propiedades: {list(schema.get('properties', {}).keys())[:5]}")
                    
                elif isinstance(data, dict):
                    print(f"📋 Datos es un diccionario con claves: {list(data.keys())}")
                else:
                    print(f"⚠️  Formato de datos inesperado: {type(data)}")
                    
            except Exception as e:
                print(f"❌ Error procesando {file_path}: {e}")
                import traceback
                traceback.print_exc()
                
        else:
            print(f"⚠️  Archivo no encontrado: {file_path}")
    
    # Test 2: Crear datos de prueba conocidos
    print(f"\n🧪 TEST CON DATOS CONTROLADOS")
    print("=" * 50)
    
    # Crear datos de prueba que deberían ser válidos
    test_data_valid = [
        {
            "PATIENT ID": "TEST_001",
            "EDAD/AGE": 45,
            "SEXO/SEX": "MALE",
            "DIAG ING/INPAT": "COVID19 - POSITIVO",
            "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME": "PARACETAMOL",
            "UCI_DIAS/ICU_DAYS": 3,
            "TEMP_ING/INPAT": 37.2,
            "SAT_02_ING/INPAT": 95,
            "RESULTADO/VAL_RESULT": 15.5,
            "MOTIVO_ALTA/DESTINY_DISCHARGE_ING": "Domicilio"
        },
        {
            "PATIENT ID": "TEST_002",
            "EDAD/AGE": 32,
            "SEXO/SEX": "FEMALE",
            "DIAG ING/INPAT": "COVID19 - NEGATIVO",
            "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME": "IBUPROFENO",
            "UCI_DIAS/ICU_DAYS": 0,
            "TEMP_ING/INPAT": 36.5,
            "SAT_02_ING/INPAT": 98,
            "RESULTADO/VAL_RESULT": 5.2,
            "MOTIVO_ALTA/DESTINY_DISCHARGE_ING": "Domicilio"
        }
    ]
    
    print(f"📊 Datos de prueba creados (2 registros válidos)")
    
    try:
        # Test detección de dominio
        domain = detect_domain_from_data(test_data_valid)
        print(f"🎯 Dominio detectado: {domain}")
        
        # Test validación
        result = validate_medical_data(test_data_valid, domain)
        print(f"📊 Resultado validación:")
        print(f"  - Total: {result['total_count']}")
        print(f"  - Válidos: {result['valid_count']}")
        print(f"  - Tasa de éxito: {result['success_rate']:.1f}%")
        
        if result['success_rate'] != 100:
            print(f"🎯 PROBLEMA: Datos conocidos como válidos no pasan validación")
            print(f"📋 Errores: {result['errors']}")
        else:
            print(f"✅ Datos de prueba validados correctamente")
            
    except Exception as e:
        print(f"❌ Error con datos de prueba: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Datos inválidos (para verificar que el validador detecta errores)
    print(f"\n🧪 TEST CON DATOS INVÁLIDOS")
    test_data_invalid = [
        {
            "PATIENT ID": "TEST_BAD",
            "EDAD/AGE": 150,  # Edad imposible
            "SEXO/SEX": "UNKNOWN",  # Sexo inválido
            "TEMP_ING/INPAT": 50,  # Temperatura imposible
            "SAT_02_ING/INPAT": 150  # Saturación imposible
        }
    ]
    
    try:
        result = validate_medical_data(test_data_invalid)
        print(f"📊 Resultado con datos inválidos:")
        print(f"  - Tasa de éxito: {result['success_rate']:.1f}%")
        print(f"  - Errores detectados: {len(result['errors'])}")
        if result['success_rate'] > 0:
            print(f"🎯 PROBLEMA: Datos inválidos no fueron detectados como tales")
        else:
            print(f"✅ Datos inválidos correctamente rechazados")
    except Exception as e:
        print(f"❌ Error con datos inválidos: {e}")

if __name__ == "__main__":
    test_json_validation_debug()
