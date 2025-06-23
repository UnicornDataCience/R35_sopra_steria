import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

script_dir = os.path.dirname(__file__)
CSV_REAL_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'real', 'df_final_v2.csv'))
df = pd.read_csv(CSV_REAL_PATH, sep=',', low_memory=False, encoding="utf-8")

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
        # Crear una copia del DataFrame filtrado
    df = df[df['DIAG ING/INPAT'].str.contains('COVID19', na=False, case=False)].copy()
    # Crear una copia explícita del DataFrame filtrado
    df_covid_dia = df.filter(regex='^DIA_').copy()
    df_covid_dia['COMORBILIDADES/COMORBILITIES'] = df_covid_dia.apply(lambda x: ', '.join(x.dropna().astype(str)), axis=1)
    # unir la columna al dataframe original df_covid
    df = df.join(df_covid_dia['COMORBILIDADES/COMORBILITIES'])
    columnas = ['EDAD/AGE','SAT_02_ING/INPAT', 'RESULTADO/VAL_RESULT', 'COMORBILIDADES/COMORBILITIES']
    df = df[columnas].copy()  # También aquí para evitar futuros problemas
    df['EDAD/AGE'] = pd.to_numeric(df['EDAD/AGE'], errors='coerce').fillna(0).astype(int)
    df['SAT_02_ING/INPAT'] = pd.to_numeric(df['SAT_02_ING/INPAT'], errors='coerce').fillna(0).astype(int)
    df['EDAD/AGE'] = df['EDAD/AGE'].replace(0, round((df['EDAD/AGE'].mean())))
    df['SAT_02_ING/INPAT'] = df['SAT_02_ING/INPAT'].replace(0, round(df['SAT_02_ING/INPAT'].mean()))
    dict_diagnosticos = {
        "HIPER_ART/ART_HYPER": ["G93.2", "I10", "I10.0", "I11.9", "I12.0", "I12.9", "I13.0", "I13.10", "I13.2", "I16.0", "I16.9", "I27.2", "I27.20", "I67.4", "K76.6"],
        "ENF_RESPIRA/RESPI_DISEASE": ["B09.9", "J06.9", "J68.4", "J80", "J95.821", "J96", "J96.0", "J96.00", "J96.01", "J96.02", "J96.1", "J96.10", "J96.2", "J96.20", "J96.21", "J96.22", "J96.9", "J96.90", "J96.91", "J96.92", "J98.8", "F17.200", "F17.201", "F17.203", "F17.204", "F17.205", "F17.206", "F17.207", "F17.208", "F17.209", "F17.210", "F17.211", "F17.212", "F17.213", "F17.214", "F17.215", "F17.216", "F17.217", "F17.218", "F17.219", "O99.335"],
        "DIABETES/DIABETES": ["E09", "E09.649", "E09.65", "E09.9", "E10", "E10.319", "E10.65", "E10.9", "E11", "E11.22", "E11.40", "E11.42", "E11.51", "E11.59", "E11.621", "E11.649", "E11.65", "E11.9", "O24", "O24.410", "Z83.2", "Z86.32"]
        }

    for tipo, codigos in dict_diagnosticos.items():
        df.loc[:, tipo] = df['COMORBILIDADES/COMORBILITIES'].apply(lambda x: 1 if any(codigo in str(x) for codigo in codigos) else 0)

    df.drop(columns=['COMORBILIDADES/COMORBILITIES'], inplace=True)
    # cambiar formato de la columna EDAD/AGE a int
    df['EDAD/AGE'] = df['EDAD/AGE'].astype(int)    
    # features = ['edad', 'epoc', 'hta', 'diabetes', 'sat_02', 'pcr'] # Variables
    df.rename(columns={
        'EDAD/AGE': 'edad',
        'SAT_02_ING/INPAT': 'sat_02',
        'RESULTADO/VAL_RESULT': 'pcr',
        'HIPER_ART/ART_HYPER': 'hta',
        'ENF_RESPIRA/RESPI_DISEASE': 'epoc',
        'DIABETES/DIABETES': 'diabetes'
    }, inplace=True)
    X = df.fillna(0) # Reemplaza NaN con 0
    X_scaled = StandardScaler().fit_transform(X) # Normaliza los datos
    pca = PCA(n_components=2) # Reduce la dimensionalidad a 2 dimensiones
    
    return pca.fit_transform(X_scaled)

# Ejemplo de uso
if __name__ == "__main__":
    patient_vectors = extract_patient_vectors(df)
    print(patient_vectors[:5])  # Muestra los primeros 5 vectores de pacientes
    # Aquí podrías guardar los vectores o realizar más análisis