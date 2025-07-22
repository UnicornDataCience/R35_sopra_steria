import json
import os
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

# ESQUEMA CORREGIDO - Más permisivo con los tipos numéricos
pacient_schema = {
    "type": "object",
    "properties": {
        "PATIENT ID": {"type": "string"},
        "EDAD/AGE": {"type": ["number", "integer"], "minimum": 0, "maximum": 120},
        "SEXO/SEX": {"type": "string", "enum": ["MALE", "FEMALE", "M", "F"]},
        "DIAG ING/INPAT": {"type": "string"},
        "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME": {"type": "string"},
        "UCI_DIAS/ICU_DAYS": {"type": ["number", "integer"], "minimum": 0},
        "TEMP_ING/INPAT": {"type": "number", "minimum": 30.0, "maximum": 45.0},
        "SAT_02_ING/INPAT": {"type": ["number", "integer"], "minimum": 50, "maximum": 100},
        "RESULTADO/VAL_RESULT": {"type": "number", "minimum": 0},
        "MOTIVO_ALTA/DESTINY_DISCHARGE_ING": {"type": "string"}
    },
    "required": ["PATIENT ID", "EDAD/AGE", "SEXO/SEX", "DIAG ING/INPAT"],
    "additionalProperties": True  # Más permisivo
}

def validate_json(patient_data, schema=None):
    """Valida un registro de paciente contra el esquema"""
    try:
        schema_to_use = schema or pacient_schema
        validate(instance=patient_data, schema=schema_to_use)
        return True
    except ValidationError as e:
        raise e

def procesar_archivo_json(nombre_archivo):
    """Procesa archivo JSON y valida cada registro"""
    
    if not os.path.exists(nombre_archivo):
        print(f"❌ Archivo no encontrado: {nombre_archivo}")
        return
    
    filename = os.path.basename(nombre_archivo)
    log_filename = f"log_json_schema_{filename.replace('.json', '.txt')}"
    log_path = os.path.join(project_root, 'outputs', log_filename)
    
    print(f"🔍 Validando archivo: {filename}")
    
    valid_count = 0
    invalid_count = 0
    
    try:
        # ✅ CORREGIR: Cargar como JSON array completo
        with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
            data = json.load(archivo)  # Cargar todo el JSON de una vez
        
        if not isinstance(data, list):
            print(f"❌ El archivo no contiene un array JSON válido")
            return
            
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Validación JSON Schema - {filename}\n")
            log_file.write("=" * 50 + "\n\n")
            
            for i, record in enumerate(data, 1):
                if record is None:
                    log_file.write(f"Registro {i}: None (saltado)\n")
                    continue
                
                if not isinstance(record, dict):
                    log_file.write(f"Registro {i}: Tipo inválido {type(record)}\n")
                    invalid_count += 1
                    continue
                
                try:
                    validate_json(record)
                    log_file.write(f"Registro {i}: ✅ VÁLIDO\n")
                    valid_count += 1
                except ValidationError as e:
                    log_file.write(f"Registro {i}: ❌ ERROR - {e.message}\n")
                    invalid_count += 1
    
    except json.JSONDecodeError as e:
        print(f"❌ Error JSON: {e}")
        return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    total_records = valid_count + invalid_count
    if total_records > 0:
        success_rate = valid_count / total_records * 100
        print(f"✅ Registros válidos: {valid_count}/{total_records} ({success_rate:.1f}%)")
        
        if success_rate >= 90:
            print("🎉 EXCELENTE: Calidad JSON muy alta")
        elif success_rate >= 75:
            print("👍 BUENO: Calidad JSON aceptable")
        else:
            print("⚠️ REVISAR: Calidad JSON baja")
    else:
        print("⚠️ No se procesaron registros")

if __name__ == "__main__":
    print("🔍 Validador de esquemas JSON para datos sintéticos")
    print(f"📂 Directorio: {synthetic_dir}")
    print("=" * 60)
    
    for archivo in archivos_json:
        if os.path.exists(archivo):
            print(f"\n📄 Procesando: {os.path.basename(archivo)}")
            procesar_archivo_json(archivo)
        else:
            print(f"\n⚠️ No encontrado: {os.path.basename(archivo)}")
    
    print("\n✅ Validación JSON Schema completada")
