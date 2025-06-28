import pandas as pd
import json
import os
from pathlib import Path

def fix_json_generation(df: pd.DataFrame, output_path: str) -> bool:
    """Genera JSON limpio sin valores None"""
    
    try:
        # 1. Limpiar DataFrame
        df_clean = df.copy()
        
        # 2. Manejar valores NaN/None por tipo de columna
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':  # Strings
                df_clean[col] = df_clean[col].fillna('').astype(str)
            elif df_clean[col].dtype in ['int64', 'float64']:  # Números
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # 3. Eliminar filas completamente vacías
        df_clean = df_clean.dropna(how='all')
        
        # 4. Convertir a lista de diccionarios
        records = df_clean.to_dict('records')
        
        # 5. Limpiar cada registro
        clean_records = []
        for record in records:
            clean_record = {}
            for key, value in record.items():
                # Convertir None a valores apropiados
                if pd.isna(value) or value is None:
                    if key in ['EDAD/AGE', 'UCI_DIAS/ICU_DAYS']:
                        clean_record[key] = 0
                    elif key in ['TEMP_ING/INPAT', 'SAT_02_ING/INPAT']:
                        clean_record[key] = 37.0 if 'TEMP' in key else 95.0
                    else:
                        clean_record[key] = ""
                else:
                    clean_record[key] = value
            
            # Solo añadir si el registro tiene contenido válido
            if any(str(v).strip() for v in clean_record.values()):
                clean_records.append(clean_record)
        
        # 6. Guardar JSON válido
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_records, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON limpio guardado: {output_path} ({len(clean_records)} registros)")
        return True
        
    except Exception as e:
        print(f"❌ Error generando JSON: {e}")
        return False

def fix_all_synthetic_jsons():
    """Regenera todos los JSONs sintéticos limpios"""
    
    base_dir = Path(__file__).parent.parent
    synthetic_dir = base_dir / 'data' / 'synthetic'
    
    # Archivos a procesar
    files_to_fix = [
        ('datos_sinteticos_sdv.csv', 'datos_sinteticos_sdv.json'),
        ('datos_sinteticos_tvae.csv', 'datos_sinteticos_tvae.json'),
        ('datos_sinteticos_ctgan.csv', 'datos_sinteticos_ctgan.json'),
    ]
    
    for csv_name, json_name in files_to_fix:
        csv_path = synthetic_dir / csv_name
        json_path = synthetic_dir / json_name
        
        if csv_path.exists():
            print(f"🔧 Arreglando {json_name}...")
            
            # Cargar CSV
            df = pd.read_csv(csv_path)
            
            # Backup del JSON original si existe
            if json_path.exists():
                backup_path = json_path.with_suffix('.json.backup')
                json_path.rename(backup_path)
                print(f"📦 Backup creado: {backup_path}")
            
            # Generar JSON limpio
            fix_json_generation(df, str(json_path))
        else:
            print(f"⚠️ No encontrado: {csv_path}")

if __name__ == "__main__":
    fix_all_synthetic_jsons()