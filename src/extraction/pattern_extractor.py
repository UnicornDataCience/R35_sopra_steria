import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def extract_patient_vectors(df):
    ''' 
    La función extract_patient_vectors toma un DataFrame de pandas y
    extrae vectores de las variables estructuradas, luego las normaliza y
    reduce la dimensionalidad a 2 dimensiones para su clustering o análisis de patrones.
    parámetros:
    - df: DataFrame de pandas que contiene las variables estructuradas de los pacientes.
    return:
    - pca.fit_transform(X_scaled): un array numpy de 2 dimensiones.
    '''
    features = ['edad', 'epoc', 'hta', 'diabetes', 'sat_02', 'pcr'] # Variables
    X = df[features].fillna(0) # Reemplaza NaN con 0
    X_scaled = StandardScaler().fit_transform(X) # Normaliza los datos
    pca = PCA(n_components=2) # Reduce la dimensionalidad a 2 dimensiones
    return pca.fit_transform(X_scaled)