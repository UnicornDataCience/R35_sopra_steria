import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata

def generate_synthetic_data_ctgan(real_df, sample_size=10):
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
    real_df['SAT_02_ING/INPAT'] = pd.to_numeric(real_df['SAT_02_ING/INPAT'], errors='coerce').fillna(0).astype(float)
    # imputar los valores 0 por la media de la columna
    real_df['UCI_DIAS/ICU_DAYS'] = real_df['UCI_DIAS/ICU_DAYS'].replace(0, real_df['UCI_DIAS/ICU_DAYS'].mean())
    real_df['EDAD/AGE'] = real_df['EDAD/AGE'].replace(0, real_df['EDAD/AGE'].mean())
    real_df['TEMP_ING/INPAT'] = real_df['TEMP_ING/INPAT'].replace(0, real_df['TEMP_ING/INPAT'].mean())
    real_df['PATIENT ID'] = real_df['PATIENT ID'].replace(0, real_df['PATIENT ID'].mean())
    real_df['SAT_02_ING/INPAT'] = real_df['SAT_02_ING/INPAT'].replace(0, real_df['SAT_02_ING/INPAT'].mean())
    
    metadata = Metadata()
    metadata = metadata.detect_from_dataframe(
        data=real_df,
        table_name = 'my_table_sdv',
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

    metadata.validate()  # Verifica integridad de metadatos
    metadata.save_to_json('metadata_ctgan.json')

    synth = CTGANSynthesizer(metadata) # Crea un sintetizador de copula gaussiana
    synth.fit(real_df) # Ajusta el sintetizador a los datos reales
    return synth.sample(sample_size)


if __name__ == "__main__":
    # Cambia aquí el nombre de tu archivo CSV
    archivo_csv = "C:\\Users\\Lenovo\\Desktop\\MASTER\\2º Cuatrimestre\\TFM\\2º fase\\r35_historia_clinica_sintetica\\R35_sopra_steria\\data\\real\\df_final.csv"
    sample_size = 10
    
    # Llama a la función y guarda el resultado
    datos_sinteticos = generate_synthetic_data_ctgan(archivo_csv, sample_size)
    print(datos_sinteticos)
    # Si quieres guardar el resultado:
    datos_sinteticos.to_csv("C:\\Users\\Lenovo\\Desktop\\MASTER\\2º Cuatrimestre\\TFM\\2º fase\\r35_historia_clinica_sintetica\\R35_sopra_steria\\data\\synthetic\\datos_sinteticos_ctgan.csv", index=False)