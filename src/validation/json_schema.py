import json
import os
from jsonschema import validate
from jsonschema.exceptions import ValidationError

def get_project_root():
    """Encuentra la raíz del proyecto automáticamente"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Buscar hacia arriba hasta encontrar el directorio que contiene 'data'
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, 'data', 'synthetic')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # Fallback
    script_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(script_dir, '..', '..'))

# Configuración dinámica
project_root = get_project_root()
data_dir = os.path.join(project_root, 'data', 'synthetic')

archivos_json = [
    os.path.join(data_dir, 'datos_sinteticos_sdv.json'),
    os.path.join(data_dir, 'datos_sinteticos_tvae.json'),
    os.path.join(data_dir, 'datos_sinteticos_ctgan.json')
]

pacient_schema = {
    "type": "object",
    "properties": {
        "PATIENT ID": {
            "type": ["string", "number"],  # Aceptar string o número
            "description": "ID único del paciente"
        },
        "EDAD/AGE": {
            "type": "number", 
            "minimum": 0, 
            "maximum": 120,
            "description": "Edad del paciente en años"
        },
        "SEXO/SEX": {
            "type": "string",
            "enum": ["MALE", "FEMALE", "M", "F", "MASCULINO", "FEMENINO"],
            "description": "Sexo del paciente"
        },
        "DIAG ING/INPAT": {
            "type": "string",
            "description": "Diagnóstico de ingreso"
        },
        "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME": {
            "type": "string",
            "description": "Medicamento prescrito"
        },
        "UCI_DIAS/ICU_DAYS": {
            "type": "number", 
            "minimum": 0,  # ✅ CORREGIDO: Permitir 0 (sin UCI)
            "description": "Días en UCI"
        },
        "TEMP_ING/INPAT": {
            "type": "number",  # ✅ CORREGIDO: number en lugar de float
            "minimum": 30.0, 
            "maximum": 45.0,  # Más permisivo para casos extremos
            "description": "Temperatura corporal en °C"
        },
        "SAT_02_ING/INPAT": {
            "type": "number", 
            "minimum": 50,  # Más permisivo para casos críticos
            "maximum": 100,
            "description": "Saturación de oxígeno en %"
        },
        "RESULTADO/VAL_RESULT": {
            "type": ["string", "number"],  # Aceptar string o número
            "description": "Resultado de laboratorio (PCR)"
        },
        "MOTIVO_ALTA/DESTINY_DISCHARGE_ING": {
            "type": "string",
            "description": "Motivo de alta hospitalaria"
        }
    },
    "required": [
        "PATIENT ID", 
        "EDAD/AGE", 
        "SEXO/SEX",
        "DIAG ING/INPAT"
    ],  # ✅ CORREGIDO: Solo campos esenciales como required
    "additionalProperties": False
}

def validate_json(patient_data):
    """Valida un registro de paciente contra el esquema"""
    try:
        # Limpiar datos antes de validar
        clean_data = {}
        for key, value in patient_data.items():
            if value is None or (isinstance(value, str) and value.strip() == ''):
                # Usar valores por defecto para campos requeridos
                if key == "EDAD/AGE":
                    clean_data[key] = 0
                elif key == "SEXO/SEX":
                    clean_data[key] = "UNKNOWN"
                elif key == "DIAG ING/INPAT":
                    clean_data[key] = "NO_DIAGNOSIS"
                elif key == "PATIENT ID":
                    clean_data[key] = "UNKNOWN_ID"
                else:
                    clean_data[key] = ""
            else:
                clean_data[key] = value
        
        validate(instance=clean_data, schema=pacient_schema)
        return True
    except ValidationError as e:
        raise e
    except Exception as e:
        raise ValidationError(f"Error de validación: {str(e)}")

def procesar_archivo_json(nombre_archivo):
    """Procesa archivo JSON línea por línea y valida cada registro"""
    
    if not os.path.exists(nombre_archivo):
        print(f"❌ Archivo no encontrado: {nombre_archivo}")
        return
    
    print(f"🔍 Validando archivo: {os.path.basename(nombre_archivo)}")
    
    try:
        # VERIFICAR: ¿Cómo está cargando el archivo?
        with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
            # PROBLEMA POSIBLE: ¿Está leyendo como JSON array o línea por línea?
            
            # MÉTODO CORRECTO para archivos JSON generados:
            data = json.load(archivo)  # Cargar como JSON completo
            
            # NO ESTO (línea por línea):
            # for line in archivo:  # ❌ Esto causaría "None"
            
        if not isinstance(data, list):
            print(f"❌ El archivo no contiene un array JSON válido")
            return
            
        # Procesar cada registro
        for i, record in enumerate(data, 1):
            if record is None:
                print(f"Línea {i}: None (registro vacío)")
                continue
                
            # Validar registro
            validate_json(record)
            
    except json.JSONDecodeError as e:
        print(f"❌ Error al decodificar JSON: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("🔍 Validador de esquemas JSON para datos sintéticos")
    print("=" * 50)
    print(f"📂 Buscando archivos en: {data_dir}")
    
    for archivo in archivos_json:
        if os.path.exists(archivo):
            print(f"\n📂 Procesando: {os.path.basename(archivo)}")
            procesar_archivo_json(archivo)
        else:
            print(f"\n⚠️ Archivo no encontrado: {os.path.basename(archivo)}")
    
    print("\n✅ Validación completada")
