#!/usr/bin/env python3
"""
Script de Diagnóstico Completo del Sistema de Validación
Identifica y corrige problemas en la validación médica de datos tabulares y sintéticos
"""

import pandas as pd
import json
import os
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

def get_project_root():
    """Encuentra la raíz del proyecto automáticamente"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, 'data', 'synthetic')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return os.getcwd()

def analyze_json_data():
    """Analiza los datos sintéticos JSON"""
    project_root = get_project_root()
    synthetic_dir = os.path.join(project_root, 'data', 'synthetic')
    
    json_files = {
        'SDV': os.path.join(synthetic_dir, 'datos_sinteticos_sdv.json'),
        'TVAE': os.path.join(synthetic_dir, 'datos_sinteticos_tvae.json'),
        'CTGAN': os.path.join(synthetic_dir, 'datos_sinteticos_ctgan.json')
    }
    
    print("=" * 80)
    print("ANÁLISIS DE DATOS SINTÉTICOS JSON")
    print("=" * 80)
    
    results = {}
    
    for name, filepath in json_files.items():
        print(f"\n📁 Analizando {name}: {os.path.basename(filepath)}")
        
        if not os.path.exists(filepath):
            print(f"❌ Archivo no encontrado: {filepath}")
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                num_records = len(data)
                sample_record = data[0] if data else {}
            else:
                print(f"❌ Formato incorrecto - esperado lista, encontrado {type(data)}")
                continue
                
            print(f"   📊 Registros: {num_records}")
            print(f"   🔑 Campos: {list(sample_record.keys())}")
            
            # Analizar campos específicos
            if sample_record:
                # Diagnóstico
                diag_field = sample_record.get('DIAG ING/INPAT', '')
                print(f"   🩺 Diagnóstico ejemplo: {diag_field}")
                
                # Medicamento
                drug_field = sample_record.get('FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME', '')
                print(f"   💊 Medicamento ejemplo: {drug_field}")
                
                # Saturación
                sat_field = sample_record.get('SAT_02_ING/INPAT', '')
                print(f"   🫁 Saturación ejemplo: {sat_field}")
                
                # Detectar dominio basado en contenido
                domain = detect_domain_from_sample(sample_record)
                print(f"   🎯 Dominio detectado: {domain}")
                
                # Analizar medicamentos únicos
                drugs = set()
                for record in data[:100]:  # Analizar primeros 100
                    drug = record.get('FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME', '')
                    if drug and drug.strip():
                        drugs.add(drug.strip().upper())
                
                print(f"   💊 Medicamentos únicos (muestra): {len(drugs)}")
                if drugs:
                    print(f"      Ejemplos: {list(sorted(drugs))[:5]}")
                
                results[name] = {
                    'records': num_records,
                    'domain': domain,
                    'unique_drugs': len(drugs),
                    'sample_drugs': list(sorted(drugs))[:10],
                    'sample_record': sample_record
                }
                
        except Exception as e:
            print(f"❌ Error procesando {name}: {str(e)}")
    
    return results

def detect_domain_from_sample(record: Dict) -> str:
    """Detecta el dominio médico de un registro de muestra"""
    # Analizar diagnóstico
    diag = str(record.get('DIAG ING/INPAT', '')).upper()
    if 'COVID' in diag:
        return 'COVID-19'
    
    # Analizar campos presentes
    fields = set(key.upper() for key in record.keys())
    
    covid_indicators = {'UCI_DIAS/ICU_DAYS', 'SAT_02_ING/INPAT', 'TEMP_ING/INPAT'}
    cardio_indicators = {'PRESION_SISTOLICA', 'FRECUENCIA_CARDIACA', 'COLESTEROL'}
    
    if covid_indicators.intersection(fields):
        return 'COVID-19'
    elif cardio_indicators.intersection(fields):
        return 'Cardiología'
    else:
        return 'Genérico'

def test_current_validation():
    """Prueba el sistema de validación actual"""
    print("\n" + "=" * 80)
    print("PRUEBA DEL SISTEMA DE VALIDACIÓN ACTUAL")
    print("=" * 80)
    
    try:
        # Importar el validador
        from src.validation.json_schema import validate_medical_data, detect_domain_from_data
        from src.agents.validator_agent import ValidatorAgent
        
        project_root = get_project_root()
        synthetic_dir = os.path.join(project_root, 'data', 'synthetic')
        
        # Crear agente validador
        validator = ValidatorAgent()
        
        # Probar cada archivo
        files_to_test = [
            'datos_sinteticos_sdv.json',
            'datos_sinteticos_tvae.json'
        ]
        
        for filename in files_to_test:
            filepath = os.path.join(synthetic_dir, filename)
            if not os.path.exists(filepath):
                print(f"❌ Archivo no encontrado: {filename}")
                continue
                
            print(f"\n📝 Probando {filename}...")
            
            # Cargar datos
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Detectar dominio
            detected_domain = detect_domain_from_data(data)
            print(f"   🎯 Dominio detectado: {detected_domain}")
            
            # Convertir a DataFrame para validación tabular
            df = pd.DataFrame(data)
            
            # Validación tabular
            print("   📊 Validación tabular...")
            tabular_results = validator.validate_tabular_data(df)
            print(f"      Score general: {tabular_results['overall_score']:.3f}")
            print(f"      Calidad datos: {tabular_results['data_quality']:.3f}")
            print(f"      Coherencia clínica: {tabular_results['clinical_coherence']:.3f}")
            print(f"      Validación farmacológica: {tabular_results['pharmacological_validation']:.3f}")
            
            if tabular_results['issues']:
                print(f"      ⚠️  Issues: {len(tabular_results['issues'])}")
                for issue in tabular_results['issues'][:3]:
                    print(f"         - {issue}")
            
            # Validación JSON
            print("   🔗 Validación JSON...")
            try:
                json_results = validate_medical_data(data)
                print(f"      Registros válidos: {json_results['valid_records']}/{json_results['total_records']}")
                print(f"      Tasa de éxito: {json_results['valid_records']/json_results['total_records']:.1%}")
                
                if json_results['errors']:
                    print(f"      ❌ Errores de validación: {len(json_results['errors'])}")
                    for error in json_results['errors'][:3]:
                        print(f"         - {error}")
                        
            except Exception as e:
                print(f"      ❌ Error en validación JSON: {str(e)}")
                
    except Exception as e:
        print(f"❌ Error en las pruebas de validación: {str(e)}")

def analyze_pharmacological_validation():
    """Analiza específicamente la validación farmacológica"""
    print("\n" + "=" * 80)
    print("ANÁLISIS DE VALIDACIÓN FARMACOLÓGICA")
    print("=" * 80)
    
    try:
        from src.validation.clinical_rules import (
            analgesicos_antiinflamatorios, opioides_potentes, psicofarmacos,
            farmacos_cardiovasculares, farmacos_respiratorios, antibioticos,
            antivirales, corticosteroides, farmacos_digestivos
        )
        
        # Mostrar listas de medicamentos disponibles
        print("\n📋 Medicamentos registrados en el sistema:")
        
        drug_categories = {
            'Analgésicos/Antiinflamatorios': analgesicos_antiinflamatorios,
            'Opioides': opioides_potentes,
            'Psicofármacos': psicofarmacos,
            'Cardiovasculares': farmacos_cardiovasculares,
            'Respiratorios': farmacos_respiratorios,
            'Antibióticos': antibioticos,
            'Antivirales': antivirales,
            'Corticosteroides': corticosteroides,
            'Digestivos': farmacos_digestivos
        }
        
        total_drugs = 0
        for category, drugs in drug_categories.items():
            print(f"\n   {category}: {len(drugs)} medicamentos")
            total_drugs += len(drugs)
            print(f"      Ejemplos: {drugs[:3]}")
        
        print(f"\n📊 Total de medicamentos en sistema: {total_drugs}")
        
        # Probar con medicamentos de los datos sintéticos
        project_root = get_project_root()
        synthetic_dir = os.path.join(project_root, 'data', 'synthetic')
        
        print("\n🧪 Probando medicamentos de datos sintéticos...")
        
        # Cargar datos SDV
        with open(os.path.join(synthetic_dir, 'datos_sinteticos_sdv.json'), 'r') as f:
            sdv_data = json.load(f)
        
        # Extraer medicamentos únicos
        synthetic_drugs = set()
        for record in sdv_data[:50]:
            drug = record.get('FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME', '')
            if drug and drug.strip():
                synthetic_drugs.add(drug.strip().upper())
        
        print(f"   Medicamentos únicos en datos sintéticos: {len(synthetic_drugs)}")
        
        # Verificar cuántos están registrados
        all_registered_drugs = set()
        for drugs in drug_categories.values():
            all_registered_drugs.update([drug.upper() for drug in drugs])
        
        matched_drugs = synthetic_drugs.intersection(all_registered_drugs)
        unmatched_drugs = synthetic_drugs - all_registered_drugs
        
        print(f"   ✅ Medicamentos reconocidos: {len(matched_drugs)}/{len(synthetic_drugs)} ({len(matched_drugs)/len(synthetic_drugs):.1%})")
        print(f"   ❌ Medicamentos NO reconocidos: {len(unmatched_drugs)}")
        
        if unmatched_drugs:
            print(f"      Ejemplos de medicamentos no reconocidos:")
            for drug in list(unmatched_drugs)[:5]:
                print(f"         - {drug}")
                
                # Buscar coincidencias parciales
                for registered_drug in all_registered_drugs:
                    if drug in registered_drug or registered_drug in drug:
                        print(f"           (Similar a: {registered_drug})")
                        break
        
        return {
            'total_registered': len(all_registered_drugs),
            'synthetic_unique': len(synthetic_drugs),
            'matched': len(matched_drugs),
            'unmatched': list(unmatched_drugs),
            'match_rate': len(matched_drugs)/len(synthetic_drugs) if synthetic_drugs else 0
        }
        
    except Exception as e:
        print(f"❌ Error analizando validación farmacológica: {str(e)}")
        return {}

def generate_improvement_recommendations(pharma_analysis: Dict):
    """Genera recomendaciones de mejora basadas en el análisis"""
    print("\n" + "=" * 80)
    print("RECOMENDACIONES DE MEJORA")
    print("=" * 80)
    
    print("\n🎯 PROBLEMAS IDENTIFICADOS:")
    
    # 1. Validación farmacológica
    if pharma_analysis.get('match_rate', 0) < 0.5:
        print(f"\n1. ❌ VALIDACIÓN FARMACOLÓGICA MUY ESTRICTA")
        print(f"   - Solo {pharma_analysis.get('match_rate', 0):.1%} de medicamentos sintéticos son reconocidos")
        print(f"   - {len(pharma_analysis.get('unmatched', []))} medicamentos válidos son rechazados")
        print(f"   - Esto causa scores farmacológicos muy bajos")
    
    print(f"\n2. ⚠️  DETECCIÓN DE DOMINIO")
    print(f"   - Puede confundir COVID-19 con cardiología")
    print(f"   - Basada principalmente en nombres de campos")
    print(f"   - Necesita analizar más el contenido")
    
    print(f"\n3. 📊 ESQUEMAS JSON RÍGIDOS")
    print(f"   - Rangos muy estrictos para valores médicos")
    print(f"   - Campos obligatorios pueden ser opcionales")
    print(f"   - No considera variabilidad de datos sintéticos")
    
    print("\n🛠️  SOLUCIONES PROPUESTAS:")
    
    print(f"\n1. 💊 FLEXIBILIZAR VALIDACIÓN FARMACOLÓGICA:")
    print(f"   ✅ Ampliar listas de medicamentos conocidos")
    print(f"   ✅ Implementar coincidencias parciales/fuzzy")
    print(f"   ✅ Permitir medicamentos genéricos comunes")
    print(f"   ✅ Reducir penalización por medicamentos desconocidos")
    
    print(f"\n2. 🎯 MEJORAR DETECCIÓN DE DOMINIO:")
    print(f"   ✅ Analizar contenido de diagnósticos")
    print(f"   ✅ Combinar análisis de campos y valores")
    print(f"   ✅ Implementar sistema de scoring más robusto")
    
    print(f"\n3. 📋 AJUSTAR ESQUEMAS JSON:")
    print(f"   ✅ Ampliar rangos de valores médicos")
    print(f"   ✅ Hacer más campos opcionales")
    print(f"   ✅ Implementar validación por niveles de severidad")
    
    print(f"\n4. 🧪 VALIDACIÓN ESPECÍFICA PARA DATOS SINTÉTICOS:")
    print(f"   ✅ Detectar automáticamente datos sintéticos")
    print(f"   ✅ Aplicar reglas más flexibles")
    print(f"   ✅ Enfocar en patrones generales vs. exactitud específica")

def main():
    """Función principal del diagnóstico"""
    print("🔍 DIAGNÓSTICO COMPLETO DEL SISTEMA DE VALIDACIÓN MÉDICA")
    print("=" * 80)
    print("Este script analiza y diagnostica problemas en el sistema de validación")
    print("de datos médicos tabulares y sintéticos del proyecto Patientia.")
    print("=" * 80)
    
    # 1. Analizar datos JSON
    json_analysis = analyze_json_data()
    
    # 2. Probar validación actual
    test_current_validation()
    
    # 3. Analizar validación farmacológica
    pharma_analysis = analyze_pharmacological_validation()
    
    # 4. Generar recomendaciones
    generate_improvement_recommendations(pharma_analysis)
    
    print("\n" + "=" * 80)
    print("✅ DIAGNÓSTICO COMPLETADO")
    print("=" * 80)
    print("Para implementar las mejoras, ejecute:")
    print("  python fix_validation_system.py")
    print("=" * 80)
    
    return {
        'json_analysis': json_analysis,
        'pharma_analysis': pharma_analysis
    }

if __name__ == "__main__":
    main()
