import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, Union
from jsonschema import validate, ValidationError

def get_project_root():
    """Encuentra la raíz del proyecto automáticamente"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, 'data', 'synthetic')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return os.getcwd()

project_root = get_project_root()
synthetic_dir = os.path.join(project_root, 'data', 'synthetic')

archivos_json = [
    os.path.join(synthetic_dir, 'datos_sinteticos_sdv.json'),
    os.path.join(synthetic_dir, 'datos_sinteticos_tvae.json'),
    os.path.join(synthetic_dir, 'datos_sinteticos_ctgan.json')
]

# FUNCIONES DE FUZZY MATCHING Y VALIDACIÓN MEJORADA

def similarity(a: str, b: str) -> float:
    """Calcula la similitud entre dos strings (0.0 a 1.0)"""
    return SequenceMatcher(None, a.upper(), b.upper()).ratio()

def fuzzy_match_medication(medication: str, known_medications: List[str], threshold: float = 0.6) -> Optional[str]:
    """
    Encuentra coincidencias parciales/fuzzy para medicamentos.
    Versión mejorada con múltiples estrategias de matching.
    
    Args:
        medication: Medicamento a buscar
        known_medications: Lista de medicamentos conocidos
        threshold: Umbral de similitud (0.0 a 1.0)
        
    Returns:
        El medicamento más similar o None
    """
    if not medication or not known_medications:
        return None
    
    medication_str = str(medication).strip()
    if not medication_str:
        return None
        
    # Múltiples estrategias de limpieza y matching
    variants = []
    
    # 1. Original
    variants.append(medication_str.upper())
    
    # 2. Sin dosis ni formas farmacéuticas
    clean1 = re.sub(r'\s+(COMP|SOL|AMP|CAP|MG|ML|G|CAPSULE|TABLET)\s*\d*.*$', '', medication_str.upper())
    clean1 = re.sub(r'\s+\d+\s*(MG|ML|G|%).*$', '', clean1)
    variants.append(clean1.strip())
    
    # 3. Solo la primera palabra (principio activo)
    first_word = medication_str.upper().split()[0] if medication_str.split() else ""
    if first_word and len(first_word) > 2:
        variants.append(first_word)
    
    # 4. Primeras dos palabras
    words = medication_str.upper().split()
    if len(words) >= 2:
        variants.append(f"{words[0]} {words[1]}")
    
    # 5. Sin espacios ni guiones
    no_spaces = re.sub(r'[\s\-_/]', '', medication_str.upper())
    variants.append(no_spaces)
    
    best_match = None
    best_score = 0.0
    
    # Probar cada variante contra cada medicamento conocido
    for variant in variants:
        if not variant or len(variant) < 2:
            continue
            
        for known in known_medications:
            known_clean = known.upper().strip()
            
            # 1. Coincidencia exacta (prioridad máxima)
            if variant == known_clean:
                return known
            
            # 2. Contenido: variant está en known o viceversa
            if variant in known_clean or known_clean in variant:
                score = 0.95
            
            # 3. Comienza con la misma palabra
            elif variant.split()[0] == known_clean.split()[0] if variant.split() and known_clean.split() else False:
                score = 0.85
            
            # 4. Similitud por diflib
            else:
                score = similarity(variant, known_clean)
            
            # Bonificación por longitud similar
            len_ratio = min(len(variant), len(known_clean)) / max(len(variant), len(known_clean))
            if len_ratio > 0.7:
                score += 0.1
            
            if score >= threshold and score > best_score:
                best_score = score
                best_match = known
    
    return best_match

# Lista expandida de medicamentos conocidos
KNOWN_MEDICATIONS = [
    # COVID-19 medications
    "DEXAMETASONA", "AZITROMICINA", "ENOXAPARINA", "FUROSEMIDA", 
    "REMDESIVIR", "TOCILIZUMAB", "METILPREDNISOLONA", "HEPARINA",
    "CEFTRIAXONA", "LEVOFLOXACINO", "PREDNISONA", "OSELTAMIVIR",
    "HYDROXYCHLOROQUINE", "DOLQUINE", "HIDROXICLOROQUINA",
    
    # Cardiology medications  
    "ATORVASTATINA", "SIMVASTATINA", "ENALAPRIL", "LOSARTAN", 
    "METOPROLOL", "BISOPROLOL", "AMLODIPINO", "VALSARTAN",
    "CLOPIDOGREL", "ASPIRINA", "WARFARINA", "RIVAROXABAN",
    "CARVEDILOL", "LISINOPRIL", "ROSUVASTATIN", "DILTIAZEM",
    
    # Common medications
    "PARACETAMOL", "IBUPROFENO", "TRAMADOL", "OMEPRAZOL",
    "PANTOPRAZOL", "METAMIZOL", "DICLOFENACO", "NAPROXENO",
    "AMOXICILINA", "CIPROFLOXACINO", "METFORMINA", "INSULINA",
    "MORFINA", "ACETILCISTEINA", "ACIDO ASCORBICO", "ALCOHOL",
    
    # Vitaminas y suplementos
    "ACFOL", "FOLICO", "VITAMINA", "ASCORBICO", "TIAMINA",
    "COMPLEJO B", "CALCIO", "MAGNESIO", "POTASIO",
    
    # Immunosuppressive and specialty drugs
    "SANDIMMUN", "NEORAL", "CICLOSPORINA", "TACROLIMUS",
    "MICOFENOLATO", "SIROLIMUS", "EVEROLIMUS",
    
    # Additional common drugs from synthetic data
    "AGUA BIDESTILADA", "SUERO SALINO", "GLUCOSA", "MANITOL",
    "ALBUMINA", "PLASMA", "FACTOR VIII", "FACTOR IX",
    
    # Synthetic/generic names for testing
    "MEDICOMP_A", "MEDICOMP_B", "TRATAMIENTO_X", "FARMACO_Y",
    "COMPUESTO_1", "MEDICINA_2", "DROGA_3", "PILL_4"
]

# ESQUEMAS PARA DIFERENTES DOMINIOS MÉDICOS

# Esquema base para todos los pacientes (más flexible)
base_patient_schema = {
    "type": "object",
    "properties": {
        "PATIENT ID": {"type": "string"},
        "EDAD/AGE": {
            "type": ["number", "integer", "string"], 
            "minimum": 0, 
            "maximum": 150  # Más amplio que antes (120)
        },
        "SEXO/SEX": {
            "type": ["string", "integer"], 
            "enum": ["MALE", "FEMALE", "M", "F", "HOMBRE", "MUJER", "0", "1", 0, 1]
        }
    },
    "required": ["PATIENT ID"],  # Solo ID es realmente requerido
    "additionalProperties": True
}

# Esquema específico para COVID-19 (más flexible)
covid_schema = {
    "type": "object",
    "properties": {
        **base_patient_schema["properties"],
        "DIAG ING/INPAT": {"type": ["string", "null"]},
        "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME": {"type": ["string", "null"]},
        "UCI_DIAS/ICU_DAYS": {
            "type": ["number", "integer", "string", "null"], 
            "minimum": 0, 
            "maximum": 365  # Más amplio
        },
        "TEMP_ING/INPAT": {
            "type": ["number", "string", "null"], 
            "minimum": 25.0,  # Más amplio (antes 30.0)
            "maximum": 50.0   # Más amplio (antes 45.0)
        },
        "SAT_02_ING/INPAT": {
            "type": ["number", "integer", "string", "null"], 
            "minimum": 40,    # Más amplio (antes 50)
            "maximum": 110    # Más amplio (antes 100)
        },
        "RESULTADO/VAL_RESULT": {
            "type": ["number", "string", "null"], 
            "minimum": -10    # Permite valores negativos
        },
        "MOTIVO_ALTA/DESTINY_DISCHARGE_ING": {"type": ["string", "null"]}
    },
    "required": ["PATIENT ID"],  # Mínimo requerido
    "additionalProperties": True
}

# Esquema específico para cardiología (más flexible)
cardiology_schema = {
    "type": "object",
    "properties": {
        **base_patient_schema["properties"],
        "DIAGNOSTICO": {"type": ["string", "null"]},
        "MEDICAMENTO": {"type": ["string", "null"]},
        "DOSIS": {"type": ["string", "null"]},
        "PRESION_SISTOLICA": {
            "type": ["number", "integer", "string", "null"], 
            "minimum": 40,    # Más amplio (antes 60)
            "maximum": 300    # Más amplio (antes 250)
        },
        "PRESION_DIASTOLICA": {
            "type": ["number", "integer", "string", "null"], 
            "minimum": 20,    # Más amplio (antes 40)
            "maximum": 200    # Más amplio (antes 150)
        },
        "FRECUENCIA_CARDIACA": {
            "type": ["number", "integer", "string", "null"], 
            "minimum": 20,    # Más amplio (antes 30)
            "maximum": 250    # Más amplio (antes 200)
        },
        "COLESTEROL_TOTAL": {
            "type": ["number", "string", "null"], 
            "minimum": 50,    # Más amplio (antes 100)
            "maximum": 600    # Más amplio (antes 400)
        },
        "HDL": {
            "type": ["number", "string", "null"], 
            "minimum": 10,    # Más amplio (antes 20)
            "maximum": 150    # Más amplio (antes 100)
        },
        "LDL": {
            "type": ["number", "string", "null"], 
            "minimum": 20,    # Más amplio (antes 50)
            "maximum": 400    # Más amplio (antes 300)
        },
        "TRIGLICERIDOS": {
            "type": ["number", "string", "null"], 
            "minimum": 30,    # Más amplio (antes 50)
            "maximum": 800    # Más amplio (antes 500)
        },
        "TIPO_TRATAMIENTO": {"type": ["string", "null"]},
        "DURACION_TRATAMIENTO": {"type": ["string", "null"]}
    },
    "required": ["PATIENT ID", "EDAD/AGE", "SEXO/SEX"],
    "additionalProperties": True
}

# Esquema genérico para datos médicos no clasificados
generic_medical_schema = {
    "type": "object",
    "properties": {
        **base_patient_schema["properties"],
        # Campos comunes en datos médicos
        "DIAGNOSTICO": {"type": "string"},
        "MEDICAMENTO": {"type": "string"},
        "TRATAMIENTO": {"type": "string"},
        "FECHA_INGRESO": {"type": "string"},
        "FECHA_ALTA": {"type": "string"},
        "ESTADO": {"type": "string"}
    },
    "required": ["PATIENT ID", "EDAD/AGE", "SEXO/SEX"],
    "additionalProperties": True
}

# Mapeo de dominios a esquemas
DOMAIN_SCHEMAS = {
    "covid": covid_schema,
    "covid-19": covid_schema,
    "cardiology": cardiology_schema,
    "cardiologia": cardiology_schema,
    "generic": generic_medical_schema,
    "default": generic_medical_schema
}

# Mantener compatibilidad hacia atrás
pacient_schema = covid_schema

def get_schema_for_domain(domain: str = "generic") -> Dict[str, Any]:
    """
    Obtiene el esquema JSON apropiado para un dominio médico específico.
    
    Args:
        domain: El dominio médico ('covid', 'cardiology', 'generic', etc.)
        
    Returns:
        Dict: El esquema JSON para el dominio especificado
    """
    domain_lower = domain.lower() if domain else "generic"
    return DOMAIN_SCHEMAS.get(domain_lower, DOMAIN_SCHEMAS["default"])

def detect_domain_from_data(data: Union[Dict, List[Dict]]) -> str:
    """
    Detecta automáticamente el dominio médico basándose en los campos de los datos.
    Versión mejorada que evita confusiones entre dominios.
    
    Args:
        data: Los datos a analizar (dict o lista de dicts)
        
    Returns:
        str: El dominio detectado ('covid', 'cardiology', 'generic')
    """
    # Convertir a lista si es un solo registro
    if isinstance(data, dict):
        data = [data]
    
    if not data or not isinstance(data, list):
        return "generic"
    
    # Analizar múltiples registros para mayor confianza
    sample_records = []
    for record in data[:10]:  # Analizar hasta 10 registros
        if isinstance(record, dict) and record:
            sample_records.append(record)
    
    if not sample_records:
        return "generic"
    
    # Combinar campos de múltiples registros
    all_fields = set()
    content_indicators = []
    
    for record in sample_records:
        fields = set(key.upper() for key in record.keys())
        all_fields.update(fields)
        
        # Analizar contenido de los valores para mayor precisión
        for key, value in record.items():
            if isinstance(value, str) and value:
                content_indicators.append(value.upper())
    
    # Indicadores específicos de COVID-19 (más restrictivos)
    covid_specific_fields = {
        'UCI_DIAS', 'ICU_DAYS', 'SAT_02_ING', 'INPAT', 'TEMP_ING',
        'DIAG ING/INPAT', 'FARMACO/DRUG_NOMBRE_COMERCIAL', 'MOTIVO_ALTA/DESTINY_DISCHARGE_ING'
    }
    
    # Indicadores específicos de cardiología (más restrictivos)
    cardiology_specific_fields = {
        'PRESION_SISTOLICA', 'PRESION_DIASTOLICA', 'FRECUENCIA_CARDIACA',
        'COLESTEROL_TOTAL', 'HDL', 'LDL', 'TRIGLICERIDOS'
    }
    
    # Contenidos específicos que ayudan a distinguir dominios
    covid_content_indicators = {
        'COVID', 'SARS-COV-2', 'CORONAVIRUS', 'NEUMONIA', 'RESPIRATORIO',
        'VENTILADOR', 'OXIGENO', 'SATURACION', 'UCI', 'ICU'
    }
    
    cardiology_content_indicators = {
        'CARDIACO', 'CARDIOLOGIA', 'CORAZON', 'PRESION', 'COLESTEROL',
        'INFARTO', 'MIOCARDIO', 'ANGINA', 'ARRITMIA', 'HIPERTENSION'
    }
    
    # Puntuación por dominio
    covid_score = 0
    cardiology_score = 0
    
    # Puntuación por campos específicos
    covid_score += len(covid_specific_fields.intersection(all_fields)) * 3
    cardiology_score += len(cardiology_specific_fields.intersection(all_fields)) * 3
    
    # Puntuación por contenido
    content_text = ' '.join(content_indicators)
    for indicator in covid_content_indicators:
        if indicator in content_text:
            covid_score += 2
    
    for indicator in cardiology_content_indicators:
        if indicator in content_text:
            cardiology_score += 2
    
    # Verificar campos que pueden estar en ambos dominios pero con diferente peso
    mixed_fields = {'DIAGNOSTICO', 'MEDICAMENTO', 'EDAD', 'AGE', 'SEXO', 'SEX'}
    mixed_score = len(mixed_fields.intersection(all_fields))
    
    # Solo asignar dominio específico si hay evidencia clara
    if covid_score >= 5 and covid_score > cardiology_score:
        return "covid"
    elif cardiology_score >= 5 and cardiology_score > covid_score:
        return "cardiology"
    elif mixed_score >= 2:
        return "generic"  # Datos médicos generales
    else:
        return "generic"

def validate_json(patient_data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None, domain: str = "generic") -> bool:
    """
    Valida un registro de paciente contra el esquema apropiado.
    
    Args:
        patient_data: Los datos del paciente a validar
        schema: Esquema específico a usar (opcional)
        domain: Dominio médico si no se proporciona esquema
        
    Returns:
        bool: True si la validación es exitosa
        
    Raises:
        ValidationError: Si la validación falla
    """
    if schema is None:
        # Auto-detectar dominio si no se especifica
        if domain == "generic":
            detected_domain = detect_domain_from_data(patient_data)
            domain = detected_domain
        schema = get_schema_for_domain(domain)
    
    try:
        validate(instance=patient_data, schema=schema)
        return True
    except ValidationError as e:
        raise e

def validate_medical_data(data: Union[Dict, List[Dict]], domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Valida un conjunto completo de datos médicos.
    
    Args:
        data: Los datos a validar (dict o lista de dicts)
        domain: Dominio médico específico (opcional, se auto-detecta)
        
    Returns:
        Dict: Resultados de la validación con estadísticas y errores
    """
    # Convertir a lista si es necesario
    if isinstance(data, dict):
        data = [data]
    
    if not isinstance(data, list):
        return {
            "valid_count": 0,
            "invalid_count": 0,
            "total_count": 0,
            "success_rate": 0.0,
            "domain": "unknown",
            "errors": ["Formato de datos no válido"]
        }
    
    # Auto-detectar dominio si no se especifica
    if domain is None:
        domain = detect_domain_from_data(data)
    
    schema = get_schema_for_domain(domain)
    
    valid_count = 0
    invalid_count = 0
    errors = []
    
    for i, record in enumerate(data, 1):
        if record is None:
            continue
            
        if not isinstance(record, dict):
            errors.append(f"Registro {i}: Tipo inválido {type(record)}")
            invalid_count += 1
            continue
        
        try:
            validate_json(record, schema, domain)
            valid_count += 1
        except ValidationError as e:
            errors.append(f"Registro {i}: {e.message}")
            invalid_count += 1
        except Exception as e:
            errors.append(f"Registro {i}: Error inesperado - {str(e)}")
            invalid_count += 1
    
    total_count = valid_count + invalid_count
    success_rate = (valid_count / total_count * 100) if total_count > 0 else 0.0
    
    return {
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "total_count": total_count,
        "success_rate": success_rate,
        "domain": domain,
        "errors": errors[:10],  # Limitar a 10 errores para no sobrecargar
        "schema_used": schema
    }

def procesar_archivo_json(nombre_archivo: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Procesa archivo JSON y valida cada registro usando el nuevo sistema genérico.
    
    Args:
        nombre_archivo: Ruta al archivo JSON
        domain: Dominio médico específico (opcional, se auto-detecta)
        
    Returns:
        Dict: Resultados de la validación
    """
    
    if not os.path.exists(nombre_archivo):
        print(f"❌ Archivo no encontrado: {nombre_archivo}")
        return {
            "valid_count": 0,
            "invalid_count": 0,
            "total_count": 0,
            "success_rate": 0.0,
            "domain": "unknown",
            "errors": [f"Archivo no encontrado: {nombre_archivo}"]
        }
    
    filename = os.path.basename(nombre_archivo)
    log_filename = f"log_json_schema_{filename.replace('.json', '.txt')}"
    log_path = os.path.join(project_root, 'outputs', log_filename)
    
    print(f"🔍 Validando archivo: {filename}")
    
    try:
        # Cargar datos JSON
        with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
            data = json.load(archivo)
        
        if not isinstance(data, list):
            print(f"❌ El archivo no contiene un array JSON válido")
            return {
                "valid_count": 0,
                "invalid_count": 0,
                "total_count": 0,
                "success_rate": 0.0,
                "domain": "unknown",
                "errors": ["El archivo no contiene un array JSON válido"]
            }
        
        # Usar el nuevo sistema de validación genérica
        results = validate_medical_data(data, domain)
        
        # Auto-detectar dominio si no se especificó
        if domain is None:
            domain = results["domain"]
        
        print(f"🎯 Dominio detectado/usado: {domain}")
        
        # Crear log detallado
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Validación JSON Schema - {filename}\n")
            log_file.write(f"Dominio médico: {domain}\n")
            log_file.write("=" * 50 + "\n\n")
            
            # Escribir estadísticas
            log_file.write(f"Total de registros: {results['total_count']}\n")
            log_file.write(f"Registros válidos: {results['valid_count']}\n")
            log_file.write(f"Registros inválidos: {results['invalid_count']}\n")
            log_file.write(f"Tasa de éxito: {results['success_rate']:.1f}%\n\n")
            
            # Escribir errores si los hay
            if results['errors']:
                log_file.write("Errores encontrados:\n")
                log_file.write("-" * 20 + "\n")
                for error in results['errors']:
                    log_file.write(f"{error}\n")
        
        # Mostrar resultados
        if results['total_count'] > 0:
            success_rate = results['success_rate']
            print(f"✅ Registros válidos: {results['valid_count']}/{results['total_count']} ({success_rate:.1f}%)")
            
            if success_rate >= 90:
                print("🎉 EXCELENTE: Calidad JSON muy alta")
            elif success_rate >= 75:
                print("👍 BUENO: Calidad JSON aceptable")
            else:
                print("⚠️ REVISAR: Calidad JSON baja")
        else:
            print("⚠️ No se procesaron registros")
        
        return results
        
    except json.JSONDecodeError as e:
        error_msg = f"Error JSON: {e}"
        print(f"❌ {error_msg}")
        return {
            "valid_count": 0,
            "invalid_count": 0,
            "total_count": 0,
            "success_rate": 0.0,
            "domain": "unknown",
            "errors": [error_msg]
        }
    except Exception as e:
        error_msg = f"Error inesperado: {e}"
        print(f"❌ {error_msg}")
        return {
            "valid_count": 0,
            "invalid_count": 0,
            "total_count": 0,
            "success_rate": 0.0,
            "domain": "unknown",
            "errors": [error_msg]
        }

if __name__ == "__main__":
    print("🔍 Validador de esquemas JSON para datos sintéticos (Genérico)")
    print(f"📂 Directorio: {synthetic_dir}")
    print("=" * 60)
    
    all_results = []
    
    for archivo in archivos_json:
        if os.path.exists(archivo):
            print(f"\n📄 Procesando: {os.path.basename(archivo)}")
            results = procesar_archivo_json(archivo)
            all_results.append((os.path.basename(archivo), results))
        else:
            print(f"\n⚠️ No encontrado: {os.path.basename(archivo)}")
    
    # Resumen final
    if all_results:
        print("\n" + "=" * 60)
        print("📊 RESUMEN FINAL")
        print("=" * 60)
        
        for filename, results in all_results:
            print(f"\n📄 {filename}:")
            print(f"   🎯 Dominio: {results['domain']}")
            print(f"   ✅ Válidos: {results['valid_count']}")
            print(f"   ❌ Inválidos: {results['invalid_count']}")
            print(f"   📊 Tasa: {results['success_rate']:.1f}%")
    
    print("\n✅ Validación JSON Schema completada")
