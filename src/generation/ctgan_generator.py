import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def generate_synthetic_data_ctgan(real_df, sample_size=1000):
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
    metadata = SingleTableMetadata() # Crea un objeto de metadatos para las columnas para la tabla única
    metadata.detect_from_dataframe(data=real_df) # Detecta automáticamente los metadatos de las columnas del DataFrame
    synth = CTGANSynthesizer(metadata) # Crea un sintetizador de copula gaussiana
    synth.fit(real_df) # Ajusta el sintetizador a los datos reales
    return synth.sample(sample_size)
