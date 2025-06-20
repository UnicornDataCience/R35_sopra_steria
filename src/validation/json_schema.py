import json
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import os

script_dir = os.getcwd()
CSV_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic', 'datos_sinteticos_sdv.json'))
CSV_PATH_2 = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic', 'datos_sinteticos_tvae.json'))


pacient_schema = {
    "type": "object",
    "properties": {
        "PATIENT ID": {"type": "string", "pattern": "^SYN"},
        "EDAD/AGE": {"type": "number", "minimum": 1},
        "SEXO/SEX": {"type": "string"},
        "DIAG ING/INPAT": {"type": "string"},
        "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME": {"type": "string"},
        "UCI_DIAS/ICU_DAYS": {"type": "number", "minimum": 1},
        "TEMP_ING/INPAT": {"type": "float", "minimum": 30.0, "maximum": 42.0},
        "SAT_02_ING/INPAT": {"type": "number", "minimum": 1, "maximum": 100},
        "RESULTADO/VAL_RESULT": {"type": "string"},
        "MOTIVO_ALTA/DESTINY_DISCHARGE_ING": {"type": "string"}
    },
    "required": ["PATIENT ID",
                "EDAD/AGE", 
                "SEXO/SEX",
                "DIAG ING/INPAT",
                "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME",
                "UCI_DIAS/ICU_DAYS",
                "TEMP_ING/INPAT",
                "SAT_02_ING/INPAT",
                "RESULTADO/VAL_RESULT",
                "MOTIVO_ALTA/DESTINY_DISCHARGE_ING"]
}

def validate_json(pacient_data):
    ''' 
    La función validate_json recibe un archivo JSON y valida su estructura
    según el esquema definido en pacient_schema. Si el archivo es válido,
    retorna True. Si no es válido, lanza una excepción de validación.
    '''
    try:
        validate(instance=pacient_data, schema=pacient_data)
        print(f"✔️  El dato con ID '{pacient_data.get('PATIENT ID')}' es VÁLIDO.")
    except ValidationError as e:
        print(f"❌ El dato con ID '{pacient_data.get('PATIENT ID')}' es INVÁLIDO.")
        print(f"   Error: {e.message}")

def procesar_archivo_jsonl(nombre_archivo):
    print(f"Procesando el archivo: {nombre_archivo}\n")
    try:
        with open(nombre_archivo, 'r', encoding='utf-8') as f:
            for numero_linea, linea in enumerate(f, 1):
                # Ignorar líneas vacías
                if not linea.strip():
                    continue
                
                try:
                    # Decodificar el JSON de la línea actual
                    datos_paciente = json.loads(linea)
                    # Validar el objeto JSON decodificado
                    validate_json(datos_paciente)
                except json.JSONDecodeError as e:
                    # Este error ocurrirá si una línea no es un JSON válido
                    print(f"❌ Error al decodificar JSON en la línea {numero_linea}: {e}")
                
                print("-" * 20)

    except FileNotFoundError:
        print(f"Error: El archivo '{nombre_archivo}' no fue encontrado.")

# --- Ejecutar el proceso ---
# Asegúrate de que el archivo 'datos_pacientes.jsonl' existe en la misma carpeta
procesar_archivo_jsonl(CSV_PATH)
procesar_archivo_jsonl(CSV_PATH_2)