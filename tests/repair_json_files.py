"""
Script para reparar archivos JSON corruptos (formato JSONL a JSON array)
"""
import json
import os

def repair_jsonl_to_json_array(input_file, output_file=None):
    """Convierte archivo JSONL (JSON Lines) a JSON array válido"""
    if output_file is None:
        output_file = input_file
    
    print(f"🔧 Reparando archivo: {input_file}")
    
    try:
        # Leer líneas como objetos JSON individuales
        records = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Ignorar líneas vacías
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Error en línea {line_num}: {e}")
        
        print(f"✅ Se procesaron {len(records)} registros")
        
        # Escribir como array JSON válido
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Archivo reparado: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error reparando archivo: {e}")
        return False

def repair_all_synthetic_files():
    """Repara todos los archivos sintéticos con problemas"""
    base_dir = "data/synthetic"
    
    files_to_repair = [
        "datos_sinteticos_tvae.json",
        "datos_sinteticos_ctgan.json"  # Por si también tiene problemas
    ]
    
    for filename in files_to_repair:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            # Crear backup
            backup_path = filepath + ".backup"
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(filepath, backup_path)
                print(f"📁 Backup creado: {backup_path}")
            
            # Reparar archivo
            repair_jsonl_to_json_array(filepath)
        else:
            print(f"⚠️ Archivo no encontrado: {filepath}")

if __name__ == "__main__":
    repair_all_synthetic_files()
