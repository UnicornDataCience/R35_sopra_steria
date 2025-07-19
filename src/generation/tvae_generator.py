import pandas as pd
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
import os
import sys
import tempfile
import uuid

script_dir = os.path.dirname(__file__)

def generate_synthetic_data_tvae(real_df, sample_size=10):
    ''' 
    La función generate_synthetic_data toma un DataFrame de pandas de datos reales
    para generar una muestra de cohortes sintéticas utilizando el sintetizador GaussianCopulaSynthesizer
    de la librería SDV.
    parámetros:
    - real_df: DataFrame de pandas que contiene los datos reales.
    - sample_size: número de muestras a generar (por defecto 1000).
    return:
    - synth.sample(sample_size): una muestra que contiene los datos sintéticos generados
    '''
    real_df = pd.read_csv(
    real_df, 
    sep =',', 
    low_memory=False, 
    encoding="utf-8"
    )
    
    # Una vez verifiques los nombres, selecciona las columnas correctas:
    columnas = ['PATIENT ID', 'EDAD/AGE', 'SEXO/SEX', 'DIAG ING/INPAT',
                'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME', 'UCI_DIAS/ICU_DAYS',
                'TEMP_ING/INPAT', 'SAT_02_ING/INPAT', 'RESULTADO/VAL_RESULT',
                'MOTIVO_ALTA/DESTINY_DISCHARGE_ING']
    
    real_df = real_df[columnas]
    real_df = real_df[real_df['DIAG ING/INPAT'].str.contains('COVID19', na=False, case=False)]
    # Rellenar los valores nulos con 0
    real_df.fillna(0, inplace=True)
    # Convertir las columnas numéricas a tipo float
    # Convertir las columnas a numéricas, forzando errores a NaN y luego rellenando con 0 antes de convertir a int/float
    real_df['EDAD/AGE'] = pd.to_numeric(real_df['EDAD/AGE'], errors='coerce').fillna(0).astype(int)
    real_df['PATIENT ID'] = pd.to_numeric(real_df['PATIENT ID'], errors='coerce').fillna(0).astype(int)
    real_df['UCI_DIAS/ICU_DAYS'] = pd.to_numeric(real_df['UCI_DIAS/ICU_DAYS'], errors='coerce').fillna(0).astype(int)
    real_df['TEMP_ING/INPAT'] = pd.to_numeric(real_df['TEMP_ING/INPAT'], errors='coerce').fillna(0).astype(float)
    real_df['SAT_02_ING/INPAT'] = pd.to_numeric(real_df['SAT_02_ING/INPAT'], errors='coerce').fillna(0).astype(int)
    # imputar los valores 0 por la media de la columna
    real_df['UCI_DIAS/ICU_DAYS'] = real_df['UCI_DIAS/ICU_DAYS'].replace(0, round(real_df['UCI_DIAS/ICU_DAYS'].mean()))
    real_df['EDAD/AGE'] = real_df['EDAD/AGE'].replace(0, round((real_df['EDAD/AGE'].mean())))
    real_df['TEMP_ING/INPAT'] = real_df['TEMP_ING/INPAT'].replace(0, real_df['TEMP_ING/INPAT'].mean())
    real_df['PATIENT ID'] = real_df['PATIENT ID'].replace(0, round(real_df['PATIENT ID'].mean()))
    real_df['SAT_02_ING/INPAT'] = real_df['SAT_02_ING/INPAT'].replace(0, round(real_df['SAT_02_ING/INPAT'].mean()))
    # en MOTIVO_ALTA/DESTINY_DISCHARGE_ING sustituir los valores nulos por 'Domici,io'
    real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].fillna('Domicilio')
    # en MOTIVO_ALTA/DESTINY_DISCHARGE_ING sustituir los valores 0 por 'Domicilio'
    real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].replace(0, 'Domicilio')
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(
        data=real_df,
        table_name = 'my_table_tvae',
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
        # regex_format="SYN_^[1-9][0-9]{0,3}$ 
        regex_format= r'SYN-[0-9]{4}')  # Formato regex para ID de paciente

    metadata.validate()  # Verifica integridad de metadatos
    # Usar nombre único para metadata temporal y eliminar tras uso
    tmp_json = f"metadata_tvae_{uuid.uuid4().hex}.json"
    try:
        metadata.save_to_json(tmp_json)
        synth = TVAESynthesizer(metadata) # Crea un sintetizador de copula gaussiana
        synth.fit(real_df) # Ajusta el sintetizador a los datos reales
        result = synth.sample(sample_size)
    finally:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)
    return result


class TVAEGenerator:
    """
    Clase para generar datos sintéticos usando SDV y TVAESynthesizer.
    """
    def __init__(self, sample_size=10):
        self.sample_size = sample_size

    def generate(self, real_df, sample_size=None, is_covid_dataset=False):
        import uuid
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
        real_df = real_df[columnas]
        real_df = real_df[real_df['DIAG ING/INPAT'].str.contains('COVID19', na=False, case=False)]
        real_df.fillna(0, inplace=True)
        real_df['EDAD/AGE'] = pd.to_numeric(real_df['EDAD/AGE'], errors='coerce').fillna(0).astype(int)
        real_df['PATIENT ID'] = pd.to_numeric(real_df['PATIENT ID'], errors='coerce').fillna(0).astype(int)
        real_df['UCI_DIAS/ICU_DAYS'] = pd.to_numeric(real_df['UCI_DIAS/ICU_DAYS'], errors='coerce').fillna(0).astype(int)
        real_df['TEMP_ING/INPAT'] = pd.to_numeric(real_df['TEMP_ING/INPAT'], errors='coerce').fillna(0).astype(float)
        real_df['SAT_02_ING/INPAT'] = pd.to_numeric(real_df['SAT_02_ING/INPAT'], errors='coerce').fillna(0).astype(int)
        real_df['UCI_DIAS/ICU_DAYS'] = real_df['UCI_DIAS/ICU_DAYS'].replace(0, round(real_df['UCI_DIAS/ICU_DAYS'].mean()))
        real_df['EDAD/AGE'] = real_df['EDAD/AGE'].replace(0, round((real_df['EDAD/AGE'].mean())))
        real_df['TEMP_ING/INPAT'] = real_df['TEMP_ING/INPAT'].replace(0, real_df['TEMP_ING/INPAT'].mean())
        real_df['PATIENT ID'] = real_df['PATIENT ID'].replace(0, round(real_df['PATIENT ID'].mean()))
        real_df['SAT_02_ING/INPAT'] = real_df['SAT_02_ING/INPAT'].replace(0, round(real_df['SAT_02_ING/INPAT'].mean()))
        real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].fillna('Domicilio')
        real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].replace(0, 'Domicilio')
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(
            data=real_df,
            table_name='my_table_tvae',
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
            regex_format='SYN-[1-9]{4}')
        metadata.validate()
        # Usar nombre único para metadata temporal y eliminar tras uso
        tmp_json = f"metadata_tvae_{uuid.uuid4().hex}.json"
        try:
            metadata.save_to_json(tmp_json)
            synth = TVAESynthesizer(metadata)
            synth.fit(real_df)
            result = synth.sample(n_samples)
        finally:
            if os.path.exists(tmp_json):
                os.remove(tmp_json)
        return result

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

def save_clean_json(df, json_path):
    """Guarda DataFrame como JSON limpio sin valores None"""
    # Limpiar DataFrame
    df_clean = df.copy()
    
    # Reemplazar NaN y None con valores apropiados
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna('')
        else:
            df_clean[col] = df_clean[col].fillna(0)
    
    # Eliminar filas completamente vacías
    df_clean = df_clean.dropna(how='all')
    
    # Convertir a JSON limpio
    df_clean.to_json(json_path, orient='records', indent=2, force_ascii=False)
    
    return len(df_clean)

if __name__ == "__main__":
    import os
    # Para pruebas, cargar un DataFrame de ejemplo
    script_dir = os.path.dirname(__file__)
    archivo_csv_covid = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'real', 'df_final_v2.csv'))
    real_df_covid_test = pd.read_csv(archivo_csv_covid, sep=',', low_memory=False, encoding="utf-8")

    archivo_csv_diabetes = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'real', 'diabetes.csv'))
    real_df_diabetes_test = pd.read_csv(archivo_csv_diabetes, sep=',', low_memory=False, encoding="utf-8")

    sample_size = 500
    
    # Probar con dataset COVID
    print("Probando TVAE con dataset COVID-19 ---")
    generator_covid = TVAEGenerator(sample_size=sample_size)
    datos_sinteticos_covid = generator_covid.generate(real_df_covid_test, sample_size, is_covid_dataset=True)
    print(f"✅ Generados {len(datos_sinteticos_covid)} registros TVAE (COVID-19)")
    print(datos_sinteticos_covid.head())

    # Probar con dataset Diabetes
    print("Probando TVAE con dataset Diabetes ---")
    generator_diabetes = TVAEGenerator(sample_size=sample_size)
    datos_sinteticos_diabetes = generator_diabetes.generate(real_df_diabetes_test, sample_size, is_covid_dataset=False)
    print(f"✅ Generados {len(datos_sinteticos_diabetes)} registros TVAE (Diabetes)")
    print(datos_sinteticos_diabetes.head())
    
    synthetic_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic'))
    os.makedirs(synthetic_dir, exist_ok=True)
    
    csv_path_covid = os.path.join(synthetic_dir, 'datos_sinteticos_tvae_covid.csv')
    json_path_covid = os.path.join(synthetic_dir, 'datos_sinteticos_tvae_covid.json')
    datos_sinteticos_covid.to_csv(csv_path_covid, index=False)
    datos_sinteticos_covid.to_json(json_path_covid, orient='records', lines=True)
    print(f"✅ Archivos COVID guardados: {csv_path_covid}, {json_path_covid}")

    csv_path_diabetes = os.path.join(synthetic_dir, 'datos_sinteticos_tvae_diabetes.csv')
    json_path_diabetes = os.path.join(synthetic_dir, 'datos_sinteticos_tvae_diabetes.json')
    datos_sinteticos_diabetes.to_csv(csv_path_diabetes, index=False)
    datos_sinteticos_diabetes.to_json(json_path_diabetes, orient='records', lines=True)
    print(f"✅ Archivos Diabetes guardados: {csv_path_diabetes}, {json_path_diabetes}")
    
    metadata_path = os.path.join(synthetic_dir, 'metadata_tvae.json')
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(
        data=datos_sinteticos_covid, # Usar el dataset COVID para la metadata principal
        table_name='my_table_tvae',
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