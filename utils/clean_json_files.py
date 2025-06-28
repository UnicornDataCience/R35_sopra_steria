import json
import pandas as pd
import os

def clean_json_file(file_path):
    """Limpia un archivo JSON eliminando registros None"""
    
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return
    
    print(f"🧹 Limpiando: {file_path}")
    
    try:
        # Cargar datos
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filtrar registros válidos
        clean_data = []
        none_count = 0
        
        for record in data:
            if record is None:
                none_count += 1
                continue
            if isinstance(record, dict) and record:
                clean_data.append(record)
        
        # Guardar archivo limpio
        backup_path = file_path.replace('.json', '_backup.json')
        os.rename(file_path, backup_path)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Limpiado: {len(clean_data)} registros válidos")
        print(f"🗑️ Eliminados: {none_count} registros None")
        print(f"💾 Backup: {backup_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Limpiar archivos JSON sintéticos
    synthetic_dir = os.path.abspath(os.path.join('..', 'data', 'synthetic'))
    
    json_files = [
        'datos_sinteticos_sdv.json',
        'datos_sinteticos_tvae.json',
        'datos_sinteticos_ctgan.json'
    ]
    
    for json_file in json_files:
        file_path = os.path.join(synthetic_dir, json_file)
        clean_json_file(file_path)