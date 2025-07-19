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

    def generate(self, real_df, sample_size=None, is_covid_dataset=False):
        n_samples = sample_size if sample_size is not None else self.sample_size
        # El DataFrame ya viene como parámetro, no necesitamos leerlo desde archivo
        # real_df = pd.read_csv(
        #     real_df_path,
        #     sep=',',
        #     low_memory=False,
        #     encoding="utf-8"
        # )
        columnas = ['PATIENT ID', 'EDAD/AGE', 'SEXO/SEX', 'DIAG ING/INPAT',
                    'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME', 'UCI_DIAS/ICU_DAYS',
                    'TEMP_ING/INPAT', 'SAT_02_ING/INPAT', 'RESULTADO/VAL_RESULT',
                    'MOTIVO_ALTA/DESTINY_DISCHARGE_ING']
        # Filtrar solo las columnas que existen en el DataFrame
        existing_cols = [col for col in columnas if col in real_df.columns]
        if existing_cols:
            real_df = real_df[existing_cols]
        real_df = real_df[real_df['DIAG ING/INPAT'].str.contains('COVID19', na=False, case=False)]
        real_df.fillna(0, inplace=True)
        real_df['EDAD/AGE'] = pd.to_numeric(real_df['EDAD/AGE'], errors='coerce').fillna(0).astype(int)
        real_df['PATIENT ID'] = pd.to_numeric(real_df['PATIENT ID'], errors='coerce').fillna(0).astype(int)
        real_df['UCI_DIAS/ICU_DAYS'] = pd.to_numeric(real_df['UCI_DIAS/ICU_DAYS'], errors='coerce').fillna(0).astype(int)
        real_df['TEMP_ING/INPAT'] = pd.to_numeric(real_df['TEMP_ING/INPAT'], errors='coerce').fillna(0).astype(float)
        real_df['SAT_02_ING/INPAT'] = pd.to_numeric(real_df['SAT_02_ING/INPAT'], errors='coerce').fillna(0).astype(int)
        real_df['UCI_DIAS/ICU_DAYS'] = real_df['UCI_DIAS/ICU_DAYS'].replace(0, round(real_df['UCI_DIAS/ICU_DAYS'].mean()))
        real_df['EDAD/AGE'] = real_df['EDAD/AGE'].replace(0, round((real_df['EDAD/AGE'].mean())))
        real_df['TEMP_ING/INPAT'] = real_df['TEMP_ING/INPAT'].replace(0, round(real_df['TEMP_ING/INPAT'].mean()))
        real_df['PATIENT ID'] = real_df['PATIENT ID'].replace(0, round(real_df['PATIENT ID'].mean()))
        real_df['SAT_02_ING/INPAT'] = real_df['SAT_02_ING/INPAT'].replace(0, round(real_df['SAT_02_ING/INPAT'].mean()))
        real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].fillna('Domicilio')
        real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].replace(0, 'Domicilio')
        metadata = SingleTableMetadata()
        metadata = metadata.detect_from_dataframe(
            data=real_df,
            table_name='my_table_sdv',
            infer_sdtypes=False
        )
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
            metadata.update_column(
                column_name=col,
                sdtype=sdtype)
        metadata.update_column(
            column_name='PATIENT ID',
            sdtype='id',
            regex_format='SYN-[0-9]{4}')
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