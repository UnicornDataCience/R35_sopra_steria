#!/usr/bin/env python3
"""
Test del nuevo sistema genérico de validación JSON Schema
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from src.validation.json_schema import (
    validate_medical_data, 
    detect_domain_from_data, 
    get_schema_for_domain,
    validate_json
)

def test_generic_json_schema():
    """Test del sistema genérico de validación JSON Schema"""
    
    print("🧪 Testing Generic JSON Schema System")
    print("=" * 50)
    
    # 1. Test detección automática de dominio COVID
    print("\n1️⃣ Test detección de dominio COVID")
    covid_data = [
        {
            "PATIENT ID": "P001",
            "EDAD/AGE": 45,
            "SEXO/SEX": "M",
            "DIAG ING/INPAT": "COVID-19",
            "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME": "Dexamethasone",
            "UCI_DIAS/ICU_DAYS": 5,
            "TEMP_ING/INPAT": 38.5,
            "SAT_02_ING/INPAT": 92
        }
    ]
    
    detected_domain = detect_domain_from_data(covid_data)
    print(f"   🎯 Dominio detectado: {detected_domain}")
    assert detected_domain == "covid", f"Expected 'covid', got '{detected_domain}'"
    print("   ✅ Detección COVID correcta")
    
    # 2. Test detección automática de dominio Cardiología
    print("\n2️⃣ Test detección de dominio Cardiología")
    cardiology_data = [
        {
            "PATIENT ID": "P002",
            "EDAD/AGE": 60,
            "SEXO/SEX": "F",
            "DIAGNOSTICO": "Hipertensión",
            "MEDICAMENTO": "Enalapril",
            "PRESION_SISTOLICA": 140,
            "PRESION_DIASTOLICA": 90,
            "FRECUENCIA_CARDIACA": 75,
            "COLESTEROL_TOTAL": 220
        }
    ]
    
    detected_domain = detect_domain_from_data(cardiology_data)
    print(f"   🎯 Dominio detectado: {detected_domain}")
    assert detected_domain == "cardiology", f"Expected 'cardiology', got '{detected_domain}'"
    print("   ✅ Detección cardiología correcta")
    
    # 3. Test detección genérica
    print("\n3️⃣ Test detección genérica")
    generic_data = [
        {
            "PATIENT ID": "P003",
            "EDAD/AGE": 30,
            "SEXO/SEX": "M",
            "DIAGNOSTICO": "Diabetes",
            "MEDICAMENTO": "Insulina"
        }
    ]
    
    detected_domain = detect_domain_from_data(generic_data)
    print(f"   🎯 Dominio detectado: {detected_domain}")
    assert detected_domain == "generic", f"Expected 'generic', got '{detected_domain}'"
    print("   ✅ Detección genérica correcta")
    
    # 4. Test validación COVID con esquema específico
    print("\n4️⃣ Test validación COVID con esquema específico")
    covid_results = validate_medical_data(covid_data, "covid")
    print(f"   📊 Resultados: {covid_results['valid_count']}/{covid_results['total_count']} válidos")
    print(f"   📈 Tasa de éxito: {covid_results['success_rate']:.1f}%")
    assert covid_results['success_rate'] == 100.0, "COVID validation should be 100%"
    print("   ✅ Validación COVID exitosa")
    
    # 5. Test validación cardiología con esquema específico
    print("\n5️⃣ Test validación cardiología con esquema específico")
    cardiology_results = validate_medical_data(cardiology_data, "cardiology")
    print(f"   📊 Resultados: {cardiology_results['valid_count']}/{cardiology_results['total_count']} válidos")
    print(f"   📈 Tasa de éxito: {cardiology_results['success_rate']:.1f}%")
    assert cardiology_results['success_rate'] == 100.0, "Cardiology validation should be 100%"
    print("   ✅ Validación cardiología exitosa")
    
    # 6. Test auto-detección y validación
    print("\n6️⃣ Test auto-detección y validación")
    mixed_data = covid_data + cardiology_data + generic_data
    
    # Validar cada tipo por separado con auto-detección
    for i, (data, expected_domain) in enumerate([
        (covid_data, "covid"),
        (cardiology_data, "cardiology"), 
        (generic_data, "generic")
    ], 1):
        results = validate_medical_data(data)  # Sin especificar dominio
        detected = results['domain']
        print(f"   🔍 Datos {i}: Detectado '{detected}', esperado '{expected_domain}'")
        assert detected == expected_domain, f"Expected '{expected_domain}', got '{detected}'"
    
    print("   ✅ Auto-detección funcionando correctamente")
    
    # 7. Test de esquemas
    print("\n7️⃣ Test obtención de esquemas")
    covid_schema = get_schema_for_domain("covid")
    cardiology_schema = get_schema_for_domain("cardiology")
    generic_schema = get_schema_for_domain("generic")
    
    print(f"   📋 Esquema COVID tiene {len(covid_schema['properties'])} propiedades")
    print(f"   📋 Esquema cardiología tiene {len(cardiology_schema['properties'])} propiedades")
    print(f"   📋 Esquema genérico tiene {len(generic_schema['properties'])} propiedades")
    
    # Verificar campos específicos
    assert "DIAG ING/INPAT" in covid_schema['properties'], "COVID schema missing COVID-specific field"
    assert "PRESION_SISTOLICA" in cardiology_schema['properties'], "Cardiology schema missing cardiology-specific field"
    
    print("   ✅ Esquemas cargados correctamente")
    
    # 8. Test datos inválidos
    print("\n8️⃣ Test datos inválidos")
    invalid_data = [
        {
            "PATIENT ID": "P004",
            "EDAD/AGE": 150,  # Edad inválida
            "SEXO/SEX": "X",  # Sexo inválido
            "DIAGNOSTICO": "Test"
        }
    ]
    
    invalid_results = validate_medical_data(invalid_data, "generic")
    print(f"   📊 Resultados: {invalid_results['valid_count']}/{invalid_results['total_count']} válidos")
    print(f"   📈 Tasa de éxito: {invalid_results['success_rate']:.1f}%")
    print(f"   ⚠️ Errores encontrados: {len(invalid_results['errors'])}")
    
    # Debería tener errores
    assert invalid_results['success_rate'] < 100.0, "Invalid data should not pass validation"
    assert len(invalid_results['errors']) > 0, "Should have validation errors"
    
    print("   ✅ Validación de datos inválidos funcionando")
    
    print("\n🎉 TODOS LOS TESTS PASARON EXITOSAMENTE")
    print("✅ El sistema genérico de validación JSON Schema está funcionando correctamente")
    
    return True

if __name__ == "__main__":
    try:
        test_generic_json_schema()
        print("\n🏆 SUCCESS: Generic JSON Schema system working correctly!")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
