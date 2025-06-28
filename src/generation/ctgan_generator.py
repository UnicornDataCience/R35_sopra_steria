import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
import tempfile
import os
import uuid

# CORREGIR: Usar la misma estructura que los otros generadores
script_dir = os.path.dirname(__file__)

class CTGANGenerator:
    """
    Clase para generar datos sintéticos usando SDV y CTGANSynthesizer.
    """
    def __init__(self, sample_size=10):
        self.sample_size = sample_size

    def generate(self, real_df_path, sample_size=None):
        import uuid
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
        
        # Filtrar solo pacientes COVID-19
        real_df = real_df[real_df['DIAG ING/INPAT'].str.contains('COVID19', na=False, case=False)]
        
        # Rellenar nulos y corregir tipos
        real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].fillna('Domicilio')
        real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = real_df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].replace(0, 'Domicilio')
        
        # Convertir columnas numéricas - IGUAL que los otros generadores
        real_df['EDAD/AGE'] = pd.to_numeric(real_df['EDAD/AGE'], errors='coerce').fillna(0).astype(int)
        real_df['PATIENT ID'] = pd.to_numeric(real_df['PATIENT ID'], errors='coerce').fillna(0).astype(int)
        real_df['UCI_DIAS/ICU_DAYS'] = pd.to_numeric(real_df['UCI_DIAS/ICU_DAYS'], errors='coerce').fillna(0).astype(int)
        real_df['TEMP_ING/INPAT'] = pd.to_numeric(real_df['TEMP_ING/INPAT'], errors='coerce').fillna(0).astype(float)
        real_df['SAT_02_ING/INPAT'] = pd.to_numeric(real_df['SAT_02_ING/INPAT'], errors='coerce').fillna(0).astype(float)
        
        # Reemplazar 0s por valores medios - IGUAL que los otros generadores
        real_df['UCI_DIAS/ICU_DAYS'] = real_df['UCI_DIAS/ICU_DAYS'].replace(0, round(real_df['UCI_DIAS/ICU_DAYS'].mean()))
        real_df['EDAD/AGE'] = real_df['EDAD/AGE'].replace(0, round(real_df['EDAD/AGE'].mean()))
        real_df['TEMP_ING/INPAT'] = real_df['TEMP_ING/INPAT'].replace(0, real_df['TEMP_ING/INPAT'].mean())
        real_df['PATIENT ID'] = real_df['PATIENT ID'].replace(0, round(real_df['PATIENT ID'].mean()))
        real_df['SAT_02_ING/INPAT'] = real_df['SAT_02_ING/INPAT'].replace(0, round(real_df['SAT_02_ING/INPAT'].mean()))

        metadata = Metadata()
        metadata = metadata.detect_from_dataframe(
            data=real_df,
            table_name='my_table_ctgan',  # CORREGIR: Usar nombre específico para CTGAN
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
            regex_format='SYN-[0-9]{4}')  # CORREGIR: Consistente con otros generadores
        
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
    # CORREGIR: Usar la misma estructura de rutas que otros generadores
    script_dir = os.path.dirname(__file__)
    archivo_csv = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'real', 'df_final_v2.csv'))  # Usar mismo archivo
    sample_size = 500  # Mismo tamaño que otros
    
    # Usar la clase CTGANGenerator
    generator = CTGANGenerator(sample_size=sample_size)
    datos_sinteticos = generator.generate(archivo_csv)
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
        
        from fix_json_generators import fix_json_generation
        return fix_json_generation(df, json_path)

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
    metadata = Metadata()
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