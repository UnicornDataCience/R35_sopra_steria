import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import os
import tempfile
import uuid

script_dir = os.path.dirname(__file__)

class SDVGenerator:
    """
    Clase para generar datos sintéticos usando SDV y GaussianCopulaSynthesizer.
    """
    def __init__(self, sample_size=1000):
        self.sample_size = sample_size

    def generate(self, real_df_path, sample_size=None):
        n_samples = sample_size if sample_size is not None else self.sample_size
        real_df = pd.read_csv(
            real_df_path,
            sep=',',
            low_memory=False,
            encoding="utf-8"
        )
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
        real_df['TEMP_ING/INPAT'] = real_df['TEMP_ING/INPAT'].replace(0, round(real_df['TEMP_ING/INPAT'].mean()))
        real_df['PATIENT ID'] = real_df['PATIENT ID'].replace(0, round(real_df['PATIENT ID'].mean()))
        real_df['SAT_02_ING/INPAT'] = real_df['SAT_02_ING/INPAT'].replace(0, round(real_df['SAT_02_ING/INPAT'].mean()))
        real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].fillna('Domicilio')
        real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].replace(0, 'Domicilio')
        metadata = Metadata()
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
            regex_format='SYN-1[0-9]{3}')
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
    import os
    archivo_csv = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'real', 'df_final_v2.csv'))
    sample_size = 1000
    # Usar la clase SDVGenerator
    generator = SDVGenerator(sample_size=sample_size)
    datos_sinteticos = generator.generate(archivo_csv)
    print(datos_sinteticos)
    datos_sinteticos.to_csv(os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic', 'datos_sinteticos_sdv.csv')))
    datos_sinteticos.to_json(os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic', 'datos_sinteticos_sdv.json')), orient='records', lines=True)