import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from typing import Dict, Optional, Literal
from .ctgan_generator import CTGANGenerator
from .tvae_generator import TVAEGenerator
from .sdv_generator import SDVGenerator

class UnifiedGenerator:
    """
    Clase unificada para generar datos sintéticos usando diferentes modelos de SDV.
    """
    
    def __init__(self):
        self.model = None
        self.model_type = None

    async def generate(
        self, 
        data: pd.DataFrame, 
        num_rows: int, 
        model_type: Literal['ctgan', 'tvae', 'gaussian_copula'] = 'ctgan',
        context: Optional[Dict] = None
    ) -> Optional[pd.DataFrame]:
        """
        Flujo completo para limpiar, entrenar y generar datos sintéticos con el modelo especificado.
        """
        print(f"Iniciando proceso de generacion con el modelo: {model_type.upper()}")
        self.model_type = model_type
        
        try:
            # Si el contexto es COVID-19, el dataset ya debe venir filtrado del agente generador
            # No es necesario volver a cargar el archivo aquí.

            # 1. Limpieza robusta de los datos
            print(f"DEBUG: Columnas antes de la limpieza: {data.columns.tolist()}")
            cleaned_data = self._robust_data_cleaner(data)
            print(f"DEBUG: Columnas después de la limpieza: {cleaned_data.columns.tolist()}")
            print(f"DEBUG: Tipos de datos de cleaned_data antes de metadata: {cleaned_data.dtypes.to_dict()}")

            # 2. Detección y validación de Metadata
            print("Detectando metadata a partir de los datos limpios...")
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(cleaned_data)
            print(f"DEBUG: Metadata detectada para las columnas: {metadata.columns}")
            metadata.validate()
            print("Detectando metadata a partir de los datos limpios...")

            # 3. Selección y entrenamiento del Modelo
            print(f"Entrenando el sintetizador {model_type.upper()}...")
            if model_type == 'ctgan':
                synthesizer = CTGANSynthesizer(metadata)
            elif model_type == 'tvae':
                synthesizer = TVAESynthesizer(metadata)
            elif model_type == 'gaussian_copula':
                synthesizer = GaussianCopulaSynthesizer(metadata)
            else:
                raise ValueError(f"Modelo '{model_type}' no soportado. Use 'ctgan', 'tvae', or 'gaussian_copula'.")

            synthesizer.fit(cleaned_data)
            self.model = synthesizer
            print("Modelo CTGAN entrenado exitosamente.")

            # 4. Generación de Muestras
            print(f"Generando {num_rows} registros sinteticos...")
            synthetic_data = self.model.sample(num_rows=num_rows)
            print("Generacion completada.")
            debug_info = {
                "cleaned_data_dtypes": cleaned_data.dtypes.to_dict(),
                "metadata_columns": metadata.columns,
                "metadata_column_details": metadata.column_details,
                "synthesizer_columns": synthesizer.get_columns(),
                "synthetic_data_columns": synthetic_data.columns.tolist()
            }
            raise Exception(f"DEBUG INFO: {debug_info}")
            return synthetic_data

        except Exception as e:
            print(f"Error crítico en el flujo de generación con {model_type.upper()}: {e}")
            import traceback
            traceback.print_exc()
            raise # Re-lanzar la excepción para que sea visible en el test

    def _robust_data_cleaner(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia un DataFrame de forma robusta, manejando números, fechas y categorías.
        """
        print("Iniciando limpieza profunda de datos...")
        cleaned_df = df.copy()

        for col in cleaned_df.columns:
            print(f"Procesando columna: {col}")
            col_non_null = cleaned_df[col].dropna()
            if col_non_null.empty:
                print(f"  -> Columna '{col}' está vacía o todo son NaNs. Se omitirá.")
                continue

            # Determinar el tipo de dato de la columna
            is_numeric = pd.api.types.is_numeric_dtype(col_non_null)
            is_datetime = pd.api.types.is_datetime64_any_dtype(col_non_null)

            if not is_numeric and not is_datetime:
                series_str = col_non_null.astype(str)
                series_str = series_str.str.replace(',', '.', regex=False)

                numeric_test = pd.to_numeric(series_str, errors='coerce')
                if (numeric_test.notna().sum() / len(col_non_null)) > 0.8:
                    is_numeric = True
                else:
                    datetime_test = pd.to_datetime(series_str, errors='coerce', infer_datetime_format=True)
                    if (datetime_test.notna().sum() / len(col_non_null)) > 0.8:
                        is_datetime = True

            # Aplicar limpieza según el tipo
            if is_numeric:
                print(f"  -> Limpiando columna NUMÉRICA: {col}")
                series_str = cleaned_df[col].astype(str).str.replace(',', '.', regex=False)
                cleaned_df[col] = pd.to_numeric(series_str, errors='coerce')
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)

            elif is_datetime:
                print(f"  -> Limpiando columna de FECHA: {col}")
                cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce', infer_datetime_format=True)
                mode_date = cleaned_df[col].mode()
                if not mode_date.empty:
                    cleaned_df[col].fillna(mode_date[0], inplace=True)

            else: # Categórica
                print(f"  -> Limpiando columna CATEGÓRICA: {col}")
                mode_val = cleaned_df[col].mode()
                if not mode_val.empty:
                    cleaned_df[col].fillna(mode_val[0], inplace=True)
                cleaned_df[col] = cleaned_df[col].astype(str)

            print(f"  -> Columna '{col}' después de limpieza: {cleaned_df[col].head(5).tolist()}")

        print("Limpieza profunda completada.")
        print(f"DEBUG: DataFrame final después de limpieza: {cleaned_df.head(5)}")
        return cleaned_df
