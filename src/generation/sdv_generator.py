import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer

from sdv.metadata import SingleTableMetadata
import os
import sys
import tempfile
import uuid

script_dir = os.path.dirname(__file__)

class SDVGenerator:
    """
    Clase para generar datos sintéticos usando SDV y GaussianCopulaSynthesizer.
    """
    def __init__(self, sample_size=1000):
        self.sample_size = sample_size

    def generate(self, real_df, sample_size=None, is_covid_dataset=False, selected_columns=None):
        n_samples = sample_size if sample_size is not None else self.sample_size
        
        # Si hay columnas seleccionadas, el DataFrame ya debería estar filtrado
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
            print(f"DEBUG: SDV - DataFrame filtrado a {len(real_df.columns)} columnas para COVID-19.")

            # Filtrar solo pacientes COVID-19 si la columna de diagnóstico existe
            if 'DIAG ING/INPAT' in real_df.columns:
                real_df = real_df[real_df['DIAG ING/INPAT'].astype(str).str.contains('COVID19', na=False, case=False)].copy()
                print(f"DEBUG: SDV - DataFrame filtrado por contenido COVID-19: {len(real_df)} filas.")
        elif selected_columns:
            print(f"🎯 SDV - Usando {len(real_df.columns)} columnas seleccionadas por MedicalColumnSelector")

        # Procesamiento dinámico de datos
        real_df = real_df.fillna(0)
        
        # Convertir columnas numéricas automáticamente basado en patrones
        numeric_patterns = ['edad', 'age', 'dias', 'days', 'temp', 'sat', 'id']
        for col in real_df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in numeric_patterns):
                if 'temp' in col_lower:
                    real_df[col] = pd.to_numeric(real_df[col], errors='coerce').fillna(0).astype(float)
                else:
                    real_df[col] = pd.to_numeric(real_df[col], errors='coerce').fillna(0).astype(int)
        
        # Reemplazar 0s por valores medios en todas las columnas numéricas
        for col in real_df.columns:
            if real_df[col].dtype in ['int64', 'float64']:
                mean_val = real_df[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0 # Fallback si la media es NaN
                real_df[col] = real_df[col].replace(0, round(mean_val) if real_df[col].dtype == 'int64' else mean_val)

        # Manejar columnas categóricas con patrones de descarga/destino
        discharge_patterns = ['motivo', 'destiny', 'alta']
        for col in real_df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in discharge_patterns):
                real_df[col] = real_df[col].fillna('Domicilio')
                real_df[col] = real_df[col].replace(0, 'Domicilio')

        # Si el DataFrame está vacío después del filtrado, no se puede generar
        if real_df.empty:
            print("❌ SDV - DataFrame vacío después del filtrado. No se puede generar datos sintéticos.")
            return pd.DataFrame() # Devolver DataFrame vacío
            
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=real_df)
        
        # Sistema dinámico de tipos de columnas
        for col in real_df.columns:
            col_lower = col.lower()
            
            # Determinar tipo basado en patrones
            if any(pattern in col_lower for pattern in numeric_patterns):
                if 'id' in col_lower:
                    metadata.update_column(column_name=col, sdtype='id', regex_format='SYN-[0-9]{4}')
                else:
                    metadata.update_column(column_name=col, sdtype='numerical')
            else:
                # Para columnas no numéricas, categórico
                metadata.update_column(column_name=col, sdtype='categorical')
        metadata.validate()
        # Usar nombre único para metadata temporal y eliminar tras uso
        tmp_json = f"metadata_sdv_{uuid.uuid4().hex}.json"
        try:
            metadata.save_to_json(tmp_json)
            synth = GaussianCopulaSynthesizer(metadata)
            synth.fit(real_df)
            synthetic = synth.sample(n_samples)
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
        return synthetic

if __name__ == "__main__":
    # Para pruebas, cargar un DataFrame de ejemplo
    script_dir = os.path.dirname(__file__)
    archivo_csv_covid = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'real', 'df_final_v2.csv'))
    real_df_covid_test = pd.read_csv(archivo_csv_covid, sep=',', low_memory=False, encoding="utf-8")

    archivo_csv_diabetes = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'real', 'diabetes.csv'))
    real_df_diabetes_test = pd.read_csv(archivo_csv_diabetes, sep=',', low_memory=False, encoding="utf-8")

    sample_size = 500
    
    # Probar con dataset COVID
    print("--- Probando SDV con dataset COVID-19 ---")
    generator_covid = SDVGenerator(sample_size=sample_size)
    datos_sinteticos_covid = generator_covid.generate(real_df_covid_test, sample_size, is_covid_dataset=True)
    print(f"✅ Generados {len(datos_sinteticos_covid)} registros SDV (COVID-19)")
    print(datos_sinteticos_covid.head())

    # Probar con dataset Diabetes
    print("--- Probando SDV con dataset Diabetes ---")
    generator_diabetes = SDVGenerator(sample_size=sample_size)
    datos_sinteticos_diabetes = generator_diabetes.generate(real_df_diabetes_test, sample_size, is_covid_dataset=False)
    print(f"✅ Generados {len(datos_sinteticos_diabetes)} registros SDV (Diabetes)")
    print(datos_sinteticos_diabetes.head())
    
    synthetic_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic'))
    os.makedirs(synthetic_dir, exist_ok=True)
    
    csv_path_covid = os.path.join(synthetic_dir, 'datos_sinteticos_sdv_covid.csv')
    json_path_covid = os.path.join(synthetic_dir, 'datos_sinteticos_sdv_covid.json')
    datos_sinteticos_covid.to_csv(csv_path_covid, index=False)
    datos_sinteticos_covid.to_json(json_path_covid, orient='records', lines=True)
    print(f"✅ Archivos COVID guardados: {csv_path_covid}, {json_path_covid}")

    csv_path_diabetes = os.path.join(synthetic_dir, 'datos_sinteticos_sdv_diabetes.csv')
    json_path_diabetes = os.path.join(synthetic_dir, 'datos_sinteticos_sdv_diabetes.json')
    datos_sinteticos_diabetes.to_csv(csv_path_diabetes, index=False)
    datos_sinteticos_diabetes.to_json(json_path_diabetes, orient='records', lines=True)
    print(f"✅ Archivos Diabetes guardados: {csv_path_diabetes}, {json_path_diabetes}")
    
    metadata_path = os.path.join(synthetic_dir, 'metadata_sdv.json')
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(
        data=datos_sinteticos_covid, # Usar el dataset COVID para la metadata principal
        table_name='my_table_sdv',
        infer_sdtypes=False
    )
    
    column_types_covid = {
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

    for col, sdtype in column_types_covid.items():
        if col in datos_sinteticos_covid.columns:
            metadata.update_column(column_name=col, sdtype=sdtype)

    if 'PATIENT ID' in datos_sinteticos_covid.columns:
        metadata.update_column(
            column_name='PATIENT ID',
            sdtype='id',
            regex_format='SYN-[0-9]{4}')

    metadata.save_to_json(metadata_path)
    print(f"   📋 Metadata: {metadata_path}")