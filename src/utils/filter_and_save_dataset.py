import pandas as pd
import os

def filter_covid_columns(input_path: str, output_path: str):
    """Filtra las columnas relevantes para COVID-19 y guarda el dataset reducido."""
    covid_columns = ['PATIENT ID', 'EDAD/AGE', 'SAT_02_ING/INPAT', 'RESULTADO/VAL_RESULT',
                     'HIPER_ART/ART_HYPER', 'ENF_RESPIRA/RESPI_DISEASE', 'DIABETES/DIABETES']

    # Leer el dataset original
    df = pd.read_csv(input_path)

    # Filtrar las columnas relevantes
    filtered_df = df[covid_columns].copy()

    # Guardar el dataset filtrado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    print(f"✅ Dataset filtrado guardado en: {output_path}")

if __name__ == "__main__":
    input_file = "data/real/df_final.csv"  # Ruta del dataset original
    output_file = "data/real/filtered_dataset.csv"  # Ruta para guardar el dataset filtrado

    filter_covid_columns(input_file, output_file)