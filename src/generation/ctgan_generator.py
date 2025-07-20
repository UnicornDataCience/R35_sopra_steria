import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import tempfile
import os
import sys
import uuid

# CORREGIR: Usar la misma estructura que los otros generadores
script_dir = os.path.dirname(__file__)

class CTGANGenerator:
    """
    Clase para generar datos sintéticos usando SDV y CTGANSynthesizer.
    """
    def __init__(self, sample_size=10):
        self.sample_size = sample_size

    def generate(self, real_df, sample_size=None, is_covid_dataset=False, selected_columns=None):
        n_samples = sample_size if sample_size is not None else self.sample_size
        
        # Si hay columnas seleccionadas, el DataFrame ya debería estar filtrado
        # pero como medida de seguridad, no aplicamos filtrado adicional aquí
        
        # Solo aplicar el filtrado legacy si NO hay columnas seleccionadas Y es COVID
        if selected_columns is None and is_covid_dataset:
            columnas_covid = ['PATIENT ID', 'EDAD/AGE', 'SEXO/SEX', 'DIAG ING/INPAT',
                              'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME', 'UCI_DIAS/ICU_DAYS',
                              'TEMP_ING/INPAT', 'SAT_02_ING/INPAT', 'RESULTADO/VAL_RESULT',
                              'MOTIVO_ALTA/DESTINY_DISCHARGE_ING']
            
            # Filtrar solo las columnas que existen en el DataFrame
            existing_covid_cols = [col for col in columnas_covid if col in real_df.columns]
            if len(existing_covid_cols) < len(columnas_covid):
                print(f"⚠️ Advertencia: Faltan algunas columnas COVID-19 esperadas. Se usarán: {existing_covid_cols}")
            
            real_df = real_df[existing_covid_cols].copy()
            print(f"DEBUG: CTGAN - DataFrame filtrado a {len(real_df.columns)} columnas para COVID-19.")

            # Filtrar solo pacientes COVID-19 si la columna de diagnóstico existe
            if 'DIAG ING/INPAT' in real_df.columns:
                real_df = real_df[real_df['DIAG ING/INPAT'].astype(str).str.contains('COVID19', na=False, case=False)].copy()
                print(f"DEBUG: CTGAN - DataFrame filtrado por contenido COVID-19: {len(real_df)} filas.")
        elif selected_columns:
            print(f"🎯 CTGAN - Usando {len(real_df.columns)} columnas seleccionadas por MedicalColumnSelector")

        # Rellenar nulos y corregir tipos (aplicar a las columnas que queden)
        # Asegurarse de que las columnas existan antes de intentar rellenar/convertir
        if 'MOTIVO_ALTA/DESTINY_DISCHARGE_ING' in real_df.columns:
            real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].fillna('Domicilio')
            real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].replace(0, 'Domicilio')
        
        # Convertir columnas numéricas - IGUAL que los otros generadores
        numeric_cols = {
            'EDAD/AGE': int,
            'PATIENT ID': int,
            'UCI_DIAS/ICU_DAYS': int,
            'TEMP_ING/INPAT': float,
            'SAT_02_ING/INPAT': float
        }
        for col, dtype in numeric_cols.items():
            if col in real_df.columns:
                real_df[col] = pd.to_numeric(real_df[col], errors='coerce').fillna(0).astype(dtype)
        
        # Reemplazar 0s por valores medios - IGUAL que los otros generadores
        for col in ['UCI_DIAS/ICU_DAYS', 'EDAD/AGE', 'TEMP_ING/INPAT', 'PATIENT ID', 'SAT_02_ING/INPAT']:
            if col in real_df.columns and real_df[col].dtype in ['int64', 'float64']:
                mean_val = real_df[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0 # Fallback si la media es NaN
                real_df[col] = real_df[col].replace(0, round(mean_val) if col != 'TEMP_ING/INPAT' else mean_val)

        # Si el DataFrame está vacío después del filtrado, no se puede generar
        if real_df.empty:
            print("❌ CTGAN - DataFrame vacío después del filtrado. No se puede generar datos sintéticos.")
            return pd.DataFrame() # Devolver DataFrame vacío

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=real_df)
        
        # Sistema dinámico de tipos de columnas basado en nombres comunes
        # Tipos numéricos típicos
        numeric_patterns = ['edad', 'age', 'dias', 'days', 'temp', 'sat', 'id']
        # Tipos categóricos típicos  
        categorical_patterns = ['sexo', 'sex', 'diag', 'farmaco', 'drug', 'resultado', 'result', 'motivo', 'destiny']
        
        # Definir tipos de columnas automáticamente
        for col in real_df.columns:
            col_lower = col.lower()
            
            # Determinar tipo basado en patrones
            if any(pattern in col_lower for pattern in numeric_patterns):
                if 'id' in col_lower:
                    metadata.update_column(column_name=col, sdtype='id', regex_format='SYN-[0-9]{4}')
                else:
                    metadata.update_column(column_name=col, sdtype='numerical')
            elif any(pattern in col_lower for pattern in categorical_patterns):
                metadata.update_column(column_name=col, sdtype='categorical')
            else:
                # Para columnas no reconocidas, dejar que SDV detecte automáticamente
                # pero podemos forzar categórico si es objeto
                if real_df[col].dtype == 'object':
                    metadata.update_column(column_name=col, sdtype='categorical')
                else:
                    metadata.update_column(column_name=col, sdtype='numerical')
        
        metadata.validate()
        
        # Usar nombre único para metadata temporal y eliminar tras uso
        tmp_json = f"metadata_ctgan_{uuid.uuid4().hex}.json"
        try:
            metadata.save_to_json(tmp_json)
            synth = CTGANSynthesizer(metadata)
            synth.fit(real_df)
            result = synth.sample(n_samples)
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
        
        return result

if __name__ == "__main__":
    # Para pruebas, cargar un DataFrame de ejemplo
    script_dir = os.path.dirname(__file__)
    archivo_csv_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'real', 'df_final_v2.csv'))
    real_df_test = pd.read_csv(archivo_csv_path, sep=',', low_memory=False, encoding="utf-8")

    sample_size = 500
    
    # Usar la clase CTGANGenerator
    generator = CTGANGenerator(sample_size=sample_size)
    datos_sinteticos = generator.generate(real_df_test, sample_size, is_covid_dataset=True) # Probar con COVID
    print(f"✅ Generados {len(datos_sinteticos)} registros CTGAN")
    print(datos_sinteticos.head())
    
    # CORREGIR: Asegurar que la carpeta existe y usar rutas consistentes
    synthetic_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic'))
    os.makedirs(synthetic_dir, exist_ok=True)
    
    # Guardar archivos con nombres consistentes
    csv_path = os.path.join(synthetic_dir, 'datos_sinteticos_ctgan.csv')
    json_path = os.path.join(synthetic_dir, 'datos_sinteticos_ctgan.json')
    
    # NUEVO: Limpiar DataFrame antes de guardar (evitar líneas None)
    datos_sinteticos = datos_sinteticos.dropna(how='all')
    datos_sinteticos = datos_sinteticos[~datos_sinteticos.isnull().all(axis=1)]
    
    # Guardar archivos
    datos_sinteticos.to_csv(csv_path, index=False)
    
    # NUEVO: Guardar JSON limpio
    def save_clean_json(df, json_path):
        """Guarda DataFrame como JSON limpio sin valores None"""
        # Importar la función de limpieza
        import sys
        import os
        utils_path = os.path.join(os.path.dirname(__file__), '..', '..', 'utils')
        sys.path.insert(0, utils_path)
        
        # from fix_json_generators import fix_json_generation  # Comentado hasta que esté disponible
        # return fix_json_generation(df, json_path)
        
        # Fallback: guardar JSON simple
        df.to_json(json_path, orient='records', indent=2)
        return True

    # REEMPLAZAR la línea:
    # datos_sinteticos.to_json(json_path, orient='records', lines=True)

    # POR:
    success = save_clean_json(datos_sinteticos, json_path)
    if success:
        print(f"✅ JSON limpio: {json_path}")
    else:
        print(f"❌ Error generando JSON: {json_path}")
    
    print(f"✅ Archivos guardados:")
    print(f"   📄 CSV: {csv_path}")
    print(f"   📄 JSON: {json_path}")
    
    # Guardar metadatos consistentes
    metadata_path = os.path.join(synthetic_dir, 'metadata_ctgan.json')
    metadata = SingleTableMetadata()
    metadata = metadata.detect_from_dataframe(
        data=datos_sinteticos,
        table_name='my_table_ctgan',  # CORREGIR: Nombre específico
        infer_sdtypes=False
    )
    
    # Configurar tipos de columnas
    column_types = {
        "EDAD/AGE": "numerical",
        "SEXO/SEX": "categorical",
        "DIAG ING/INPAT": "categorical",
        "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME": "categorical",
        "UCI_DIAS/ICU_DAYS": "numerical",
        "TEMP_ING/INPAT": "numerical",
        "SAT_02_ING/INPAT": "numerical",
        "RESULTADO/VAL_RESULT": "categorical",
        "MOTIVO_ALTA/DESTINY_DISCHARGE_ING": "categorical"
    }
    
    for col, sdtype in column_types.items():
        if col in datos_sinteticos.columns:
            metadata.update_column(column_name=col, sdtype=sdtype)
    
    metadata.update_column(
        column_name='PATIENT ID',
        sdtype='id',
        regex_format='SYN-[0-9]{4}')
    
    metadata.save_to_json(metadata_path)
    print(f"   📋 Metadata: {metadata_path}")