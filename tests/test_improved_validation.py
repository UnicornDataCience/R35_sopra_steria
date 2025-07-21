#!/usr/bin/env python3
"""
Test script para verificar las mejoras en el sistema de validación.
Prueba fuzzy matching, esquemas flexibles y detección de dominio mejorada.
"""

import json
import sys
import os
import pandas as pd

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_json_fuzzy_matching():
    """Test del fuzzy matching de medicamentos en JSON validation"""
    print("🧪 TEST: FUZZY MATCHING JSON MEDICAMENTOS")
    print("=" * 50)
    
    try:
        from validation.json_schema import fuzzy_match_medication, KNOWN_MEDICATIONS
        
        # Test medications from synthetic data
        test_medications = [
            "PARACETAMOL SOL 1 G/100 ML",
            "BISOPROLOL COMP 5 MG", 
            "ACFOL COMP 5 MG",
            "SANDIMMUN NEORAL CAP 50 MG",
            "DOLQUINE COMP 200 MG"
        ]
        
        print(f"📋 Medicamentos conocidos en sistema: {len(KNOWN_MEDICATIONS)}")
        print(f"🔍 Probando {len(test_medications)} medicamentos sintéticos...")
        
        matches = 0
        for med in test_medications:
            match = fuzzy_match_medication(med, KNOWN_MEDICATIONS, threshold=0.7)
            if match:
                matches += 1
                print(f"✅ '{med}' → '{match}'")
            else:
                print(f"❌ '{med}' → No match found")
        
        recognition_rate = matches / len(test_medications)
        print(f"\n📊 Tasa de reconocimiento: {recognition_rate:.1%} ({matches}/{len(test_medications)})")
        
        if recognition_rate >= 0.6:
            print("🎉 MEJORA: El fuzzy matching funciona!")
        else:
            print("⚠️ Fuzzy matching necesita más ajustes")
            
        return recognition_rate >= 0.6
        
    except Exception as e:
        print(f"❌ Error en test fuzzy matching: {e}")
        return False

def test_flexible_schemas():
    """Test de los esquemas JSON más flexibles"""
    print("\n🧪 TEST: ESQUEMAS JSON FLEXIBLES")
    print("=" * 50)
    
    try:
        from validation.json_schema import validate_medical_data
        
        # Datos de prueba con valores que antes eran rechazados
        test_data = [
            {
                "PATIENT ID": "TEST001",
                "EDAD/AGE": "65",  # String en lugar de number
                "SEXO/SEX": "1",   # String en lugar de categorical
                "TEMP_ING/INPAT": "38.5",  # String numeric
                "SAT_02_ING/INPAT": 105,   # Fuera del rango anterior (100)
                "DIAG ING/INPAT": "COVID19 - POSITIVO"
            },
            {
                "PATIENT ID": "TEST002", 
                "EDAD/AGE": 140,  # Fuera del rango anterior (120)
                "SEXO/SEX": "MALE",
                "TEMP_ING/INPAT": 45.5,  # Fuera del rango anterior (45.0)
                "SAT_02_ING/INPAT": "45",  # Valor bajo, antes era 50 min
                "RESULTADO/VAL_RESULT": -5  # Valor negativo, antes min era 0
            }
        ]
        
        results = validate_medical_data(test_data)
        success_rate = results['success_rate']
        
        print(f"📊 Datos de prueba: {len(test_data)} registros")
        print(f"✅ Registros válidos: {results['valid_count']}")
        print(f"❌ Registros inválidos: {results['invalid_count']}")
        print(f"📈 Tasa de éxito: {success_rate:.1f}%")
        print(f"🎯 Dominio detectado: {results['domain']}")
        
        if results['errors']:
            print("\n⚠️ Errores encontrados:")
            for error in results['errors'][:3]:
                print(f"   - {error}")
        
        if success_rate >= 80:
            print("🎉 MEJORA: Esquemas más flexibles funcionan!")
        else:
            print("⚠️ Esquemas necesitan más flexibilidad")
            
        return success_rate >= 80
        
    except Exception as e:
        print(f"❌ Error en test esquemas flexibles: {e}")
        return False

def test_domain_detection():
    """Test de la detección de dominio mejorada"""
    print("\n🧪 TEST: DETECCIÓN DE DOMINIO MEJORADA")
    print("=" * 50)
    
    try:
        from validation.json_schema import detect_domain_from_data
        
        # Test COVID data
        covid_data = [
            {
                "PATIENT ID": "COVID001",
                "DIAG ING/INPAT": "COVID19 - POSITIVO",
                "UCI_DIAS/ICU_DAYS": 5,
                "SAT_02_ING/INPAT": 85,
                "TEMP_ING/INPAT": 38.2
            }
        ]
        
        # Test Cardiology data
        cardiology_data = [
            {
                "PATIENT ID": "CARD001",
                "PRESION_SISTOLICA": 140,
                "PRESION_DIASTOLICA": 90,
                "COLESTEROL_TOTAL": 220,
                "FRECUENCIA_CARDIACA": 75
            }
        ]
        
        # Test Mixed/Generic data
        generic_data = [
            {
                "PATIENT ID": "GEN001",
                "EDAD/AGE": 45,
                "SEXO/SEX": "M",
                "DIAGNOSTICO": "General checkup"
            }
        ]
        
        covid_domain = detect_domain_from_data(covid_data)
        cardiology_domain = detect_domain_from_data(cardiology_data)
        generic_domain = detect_domain_from_data(generic_data)
        
        print(f"🦠 Datos COVID → Dominio: '{covid_domain}'")
        print(f"❤️ Datos Cardiología → Dominio: '{cardiology_domain}'")
        print(f"🏥 Datos Genéricos → Dominio: '{generic_domain}'")
        
        correct_detections = 0
        if covid_domain == "covid":
            correct_detections += 1
            print("✅ COVID detectado correctamente")
        else:
            print("❌ COVID no detectado correctamente")
            
        if cardiology_domain == "cardiology":
            correct_detections += 1
            print("✅ Cardiología detectada correctamente")
        else:
            print("❌ Cardiología no detectada correctamente")
            
        if generic_domain == "generic":
            correct_detections += 1
            print("✅ Genérico detectado correctamente")
        else:
            print("❌ Genérico no detectado correctamente")
        
        detection_rate = correct_detections / 3
        print(f"\n📊 Tasa de detección correcta: {detection_rate:.1%}")
        
        if detection_rate >= 0.67:
            print("🎉 MEJORA: Detección de dominio funciona!")
        else:
            print("⚠️ Detección de dominio necesita ajustes")
            
        return detection_rate >= 0.67
        
    except Exception as e:
        print(f"❌ Error en test detección de dominio: {e}")
        return False

def test_tabular_fuzzy_matching():
    """Test del fuzzy matching en validación tabular"""
    print("\n🧪 TEST: FUZZY MATCHING TABULAR")
    print("=" * 50)
    
    try:
        from validation.tabular_medical_validator import TabularMedicalValidator
        
        # Crear datos de prueba con medicamentos
        test_data = pd.DataFrame({
            'PATIENT ID': ['TEST001', 'TEST002', 'TEST003'],
            'medication': [
                'PARACETAMOL SOL 1 G/100 ML',
                'BISOPROLOL COMP 5 MG',
                'Unknown Drug XYZ'
            ],
            'age': [45, 60, 35]
        })
        
        validator = TabularMedicalValidator()
        
        # Test fuzzy matching individual
        med_score = validator._validate_medication_field(test_data['medication'])
        print(f"📊 Score de medicamentos: {med_score:.1%}")
        
        # Test validación completa
        quality_results = validator.validate_data_quality(test_data)
        print(f"📈 Calidad general: {quality_results['overall_quality_score']:.1%}")
        
        if med_score >= 0.5:
            print("✅ Fuzzy matching tabular funciona")
        else:
            print("⚠️ Fuzzy matching tabular necesita ajustes")
            
        return med_score >= 0.5
        
    except Exception as e:
        print(f"❌ Error en test fuzzy matching tabular: {e}")
        return False

def main():
    """Test completo del sistema de validación mejorado"""
    print("🔍 TEST COMPLETO: SISTEMA DE VALIDACIÓN MEJORADO")
    print("=" * 60)
    print("Verificando mejoras implementadas:")
    print("  1. Fuzzy matching para medicamentos")
    print("  2. Esquemas JSON más flexibles")
    print("  3. Detección de dominio mejorada")
    print("  4. Validación tabular con fuzzy matching")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Run all tests
    if test_json_fuzzy_matching():
        tests_passed += 1
        
    if test_flexible_schemas():
        tests_passed += 1
        
    if test_domain_detection():
        tests_passed += 1
        
    if test_tabular_fuzzy_matching():
        tests_passed += 1
    
    # Final results
    print("\n" + "=" * 60)
    print("📊 RESULTADOS FINALES")
    print("=" * 60)
    print(f"✅ Tests exitosos: {tests_passed}/{total_tests}")
    print(f"📈 Tasa de éxito: {tests_passed/total_tests:.1%}")
    
    if tests_passed >= 3:
        print("🎉 SISTEMA MEJORADO EXITOSAMENTE!")
        print("   - Validación más flexible y realista")
        print("   - Mejor reconocimiento de medicamentos")
        print("   - Detección de dominio más precisa")
    elif tests_passed >= 2:
        print("👍 MEJORAS PARCIALES IMPLEMENTADAS")
        print("   - Algunas validaciones mejoradas")
        print("   - Necesita ajustes adicionales")
    else:
        print("⚠️ MEJORAS NECESITAN MÁS TRABAJO")
        print("   - Revise la implementación")
        print("   - Verifique configuraciones")
    
    return tests_passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
