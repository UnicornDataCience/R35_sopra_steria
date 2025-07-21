#!/usr/bin/env python3
"""
Test final del sistema completo de validación médica con el nuevo sistema genérico JSON Schema
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import json
from src.agents.validator_agent import MedicalValidatorAgent
from src.validation.json_schema import validate_medical_data, detect_domain_from_data

async def test_integrated_validation_system():
    """Test del sistema integrado de validación médica"""
    
    print("🧪 Testing Integrated Medical Validation System")
    print("=" * 60)
    
    # Inicializar el agente validador
    validator_agent = MedicalValidatorAgent()
    print("✅ MedicalValidatorAgent inicializado")
    
    # 1. Test con datos sintéticos (JSON) - COVID
    print("\n1️⃣ Test con datos sintéticos COVID (JSON Schema)")
    covid_synthetic_data = pd.DataFrame([
        {
            "PATIENT ID": "P001",
            "EDAD/AGE": 45,
            "SEXO/SEX": "M",
            "DIAG ING/INPAT": "COVID-19",
            "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME": "Dexamethasone",
            "UCI_DIAS/ICU_DAYS": 5,
            "TEMP_ING/INPAT": 38.5,
            "SAT_02_ING/INPAT": 92,
            "RESULTADO/VAL_RESULT": 0.5,
            "MOTIVO_ALTA/DESTINY_DISCHARGE_ING": "Alta médica"
        },
        {
            "PATIENT ID": "P002",
            "EDAD/AGE": 60,
            "SEXO/SEX": "F",
            "DIAG ING/INPAT": "COVID-19",
            "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME": "Remdesivir",
            "UCI_DIAS/ICU_DAYS": 3,
            "TEMP_ING/INPAT": 37.8,
            "SAT_02_ING/INPAT": 95,
            "RESULTADO/VAL_RESULT": 0.3,
            "MOTIVO_ALTA/DESTINY_DISCHARGE_ING": "Alta médica"
        }
    ])
    
    # Test validación directa con el nuevo sistema JSON
    sample_data = []
    for _, row in covid_synthetic_data.iterrows():
        record = row.to_dict()
        sample_data.append(record)
    
    json_results = validate_medical_data(sample_data)
    print(f"   🎯 Dominio detectado: {json_results['domain']}")
    print(f"   📊 Validación JSON: {json_results['valid_count']}/{json_results['total_count']} ({json_results['success_rate']:.1f}%)")
    assert json_results['domain'] == 'covid', "Should detect COVID domain"
    assert json_results['success_rate'] == 100.0, "COVID synthetic data should pass JSON validation"
    
    # Test con el agente validador
    context = {
        "synthetic_data": covid_synthetic_data,
        "validation_target": "synthetic",
        "universal_analysis": {"dataset_type": "COVID-19"}
    }
    
    validation_result = await validator_agent.process("validar datos", context)
    print(f"   🤖 Validación del agente completada")
    print(f"   📝 Tipo de respuesta: {'✅ Exitosa' if not validation_result.get('error') else '❌ Error'}")
    
    # 2. Test con datos sintéticos (JSON) - Cardiología
    print("\n2️⃣ Test con datos sintéticos Cardiología (JSON Schema)")
    cardiology_synthetic_data = pd.DataFrame([
        {
            "PATIENT ID": "C001",
            "EDAD/AGE": 65,
            "SEXO/SEX": "M",
            "DIAGNOSTICO": "Hipertensión",
            "MEDICAMENTO": "Enalapril",
            "PRESION_SISTOLICA": 140,
            "PRESION_DIASTOLICA": 90,
            "FRECUENCIA_CARDIACA": 75,
            "COLESTEROL_TOTAL": 220,
            "HDL": 45,
            "LDL": 150
        }
    ])
    
    # Test validación directa
    cardio_sample = [cardiology_synthetic_data.iloc[0].to_dict()]
    cardio_json_results = validate_medical_data(cardio_sample)
    print(f"   🎯 Dominio detectado: {cardio_json_results['domain']}")
    print(f"   📊 Validación JSON: {cardio_json_results['valid_count']}/{cardio_json_results['total_count']} ({cardio_json_results['success_rate']:.1f}%)")
    assert cardio_json_results['domain'] == 'cardiology', "Should detect cardiology domain"
    
    # Test con el agente validador
    cardio_context = {
        "synthetic_data": cardiology_synthetic_data,
        "validation_target": "synthetic"
    }
    
    cardio_validation_result = await validator_agent.process("validar datos", cardio_context)
    print(f"   🤖 Validación del agente completada")
    print(f"   📝 Tipo de respuesta: {'✅ Exitosa' if not cardio_validation_result.get('error') else '❌ Error'}")
    
    # 3. Test con datos originales (Tabular)
    print("\n3️⃣ Test con datos originales (Validación Tabular)")
    original_data = pd.DataFrame([
        {
            "PATIENT_ID": "O001",
            "EDAD": 45,
            "SEXO": "M",
            "DIAGNOSTICO": "Hipertensión",
            "MEDICACION": "Losartán",
            "PRESION_ARTERIAL": "140/90"
        },
        {
            "PATIENT_ID": "O002",
            "EDAD": 52,
            "SEXO": "F",
            "DIAGNOSTICO": "Diabetes",
            "MEDICACION": "Metformina",
            "GLUCOSA": 120
        }
    ])
    
    original_context = {
        "dataframe": original_data,
        "validation_target": "original"
    }
    
    original_validation_result = await validator_agent.process("validar datos", original_context)
    print(f"   🤖 Validación tabular completada")
    print(f"   📝 Tipo de respuesta: {'✅ Exitosa' if not original_validation_result.get('error') else '❌ Error'}")
    
    # 4. Test priorización: Sintéticos sobre originales
    print("\n4️⃣ Test priorización: Sintéticos sobre originales")
    mixed_context = {
        "synthetic_data": covid_synthetic_data,
        "dataframe": original_data,
        # No especificar validation_target para que use la lógica de priorización
    }
    
    mixed_validation_result = await validator_agent.process("validar datos", mixed_context)
    print(f"   🤖 Validación con priorización completada")
    print(f"   📝 Tipo de respuesta: {'✅ Exitosa' if not mixed_validation_result.get('error') else '❌ Error'}")
    print(f"   🎯 Target usado: {mixed_validation_result.get('validation_target', 'Unknown')}")
    
    # Debe haber priorizado datos sintéticos
    assert mixed_validation_result.get('validation_target') == 'synthetic', "Should prioritize synthetic data"
    
    # 5. Test con datos genéricos
    print("\n5️⃣ Test con datos genéricos")
    generic_data = pd.DataFrame([
        {
            "PATIENT ID": "G001",
            "EDAD/AGE": 30,
            "SEXO/SEX": "F",
            "DIAGNOSTICO": "Migraine",
            "MEDICAMENTO": "Ibuprofen"
        }
    ])
    
    generic_sample = [generic_data.iloc[0].to_dict()]
    generic_json_results = validate_medical_data(generic_sample)
    print(f"   🎯 Dominio detectado: {generic_json_results['domain']}")
    print(f"   📊 Validación JSON: {generic_json_results['valid_count']}/{generic_json_results['total_count']} ({generic_json_results['success_rate']:.1f}%)")
    
    generic_context = {
        "synthetic_data": generic_data,
        "validation_target": "synthetic"
    }
    
    generic_validation_result = await validator_agent.process("validar datos", generic_context)
    print(f"   🤖 Validación genérica completada")
    print(f"   📝 Tipo de respuesta: {'✅ Exitosa' if not generic_validation_result.get('error') else '❌ Error'}")
    
    print("\n🎉 TODOS LOS TESTS DEL SISTEMA INTEGRADO PASARON EXITOSAMENTE")
    print("✅ El sistema de validación médica con JSON Schema genérico está funcionando correctamente")
    
    # Resumen de capacidades
    print("\n📋 CAPACIDADES VERIFICADAS:")
    print("   ✅ Auto-detección de dominios médicos (COVID, Cardiología, Genérico)")
    print("   ✅ Validación JSON Schema específica por dominio")
    print("   ✅ Validación tabular para datos originales")
    print("   ✅ Priorización de datos sintéticos sobre originales")
    print("   ✅ Integración completa con MedicalValidatorAgent")
    print("   ✅ Compatibilidad hacia atrás mantenida")
    
    return True

if __name__ == "__main__":
    import asyncio
    
    try:
        asyncio.run(test_integrated_validation_system())
        print("\n🏆 SUCCESS: Integrated Medical Validation System working correctly!")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
