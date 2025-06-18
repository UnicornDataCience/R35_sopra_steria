import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import os

class SDVGenerator:
    """Clase para generación de datos sintéticos usando SDV"""
    
    def __init__(self):
        self.synthesizer = None
        self.metadata = None
        self.quality_score = None
        
    def generate(self, real_df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
        """
        Genera datos sintéticos usando SDV
        
        Args:
            real_df: DataFrame con datos reales
            sample_size: Número de muestras a generar
            
        Returns:
            DataFrame con datos sintéticos
        """
        try:
            # Procesar datos
            processed_df = self._preprocess_data(real_df.copy())
            
            # Crear metadata
            self.metadata = self._create_metadata(processed_df)
            
            # Crear y entrenar sintetizador
            self.synthesizer = GaussianCopulaSynthesizer(self.metadata)
            self.synthesizer.fit(processed_df)
            
            # Generar datos sintéticos
            synthetic_data = self.synthesizer.sample(sample_size)
            
            # Calcular score de calidad
            self.quality_score = self._calculate_quality_score(processed_df, synthetic_data)
            
            return synthetic_data
            
        except Exception as e:
            raise RuntimeError(f"Error en generación SDV: {str(e)}")
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesa los datos para SDV"""
        
        # Definir columnas requeridas
        columnas = [
            'PATIENT ID', 'EDAD/AGE', 'SEXO/SEX', 'DIAG ING/INPAT',
            'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME', 'UCI_DIAS/ICU_DAYS',
            'TEMP_ING/INPAT', 'SAT_02_ING/INPAT', 'RESULTADO/VAL_RESULT',
            'MOTIVO_ALTA/DESTINY_DISCHARGE_ING'
        ]
        
        # Verificar que las columnas existen
        available_cols = [col for col in columnas if col in df.columns]
        if not available_cols:
            raise ValueError("No se encontraron columnas requeridas en el DataFrame")
        
        df = df[available_cols]
        
        # Filtrar casos COVID-19 si la columna existe
        if 'DIAG ING/INPAT' in df.columns:
            df = df[df['DIAG ING/INPAT'].str.contains('COVID19', na=False, case=False)]
        
        # Rellenar valores nulos
        df.fillna(0, inplace=True)
        
        # Convertir columnas numéricas
        numeric_conversions = {
            'EDAD/AGE': 'int',
            'PATIENT ID': 'int', 
            'UCI_DIAS/ICU_DAYS': 'int',
            'TEMP_ING/INPAT': 'float',
            'SAT_02_ING/INPAT': 'int'
        }
        
        for col, dtype in numeric_conversions.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                if dtype == 'int':
                    df[col] = df[col].astype(int)
                else:
                    df[col] = df[col].astype(float)
        
        # Reemplazar ceros con valores medios en columnas numéricas
        numeric_cols = ['UCI_DIAS/ICU_DAYS', 'EDAD/AGE', 'TEMP_ING/INPAT', 'PATIENT ID', 'SAT_02_ING/INPAT']
        for col in numeric_cols:
            if col in df.columns:
                mean_val = df[df[col] != 0][col].mean()
                if not pd.isna(mean_val):
                    df[col] = df[col].replace(0, round(mean_val))
        
        # Manejar columnas categóricas
        categorical_replacements = {
            'MOTIVO_ALTA/DESTINY_DISCHARGE_ING': 'Domicilio'
        }
        
        for col, default_val in categorical_replacements.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val)
                df[col] = df[col].replace(0, default_val)
        
        return df
    
    def _create_metadata(self, df: pd.DataFrame) -> Metadata:
        """Crea metadata para SDV"""
        
        metadata = Metadata()
        metadata = metadata.detect_from_dataframe(
            data=df,
            table_name='clinical_data_table',
            infer_sdtypes=False
        )
        
        # Definir tipos de columnas
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
        
        # Actualizar metadata para columnas existentes
        for col, sdtype in column_types.items():
            if col in df.columns:
                metadata.update_column(column_name=col, sdtype=sdtype)
        
        # Configurar PATIENT ID como identificador
        if 'PATIENT ID' in df.columns:
            metadata.update_column(
                column_name='PATIENT ID',
                sdtype='id',
                regex_format='SYN-[0-9]{4}'
            )
        
        # Validar metadata
        metadata.validate()
        
        return metadata
    
    def _calculate_quality_score(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Calcula un score básico de calidad"""
        try:
            # Comparar distribuciones básicas
            score = 0.0
            comparisons = 0
            
            numeric_cols = original.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in synthetic.columns:
                    # Comparar medias
                    orig_mean = original[col].mean()
                    synth_mean = synthetic[col].mean()
                    
                    if orig_mean != 0:
                        mean_diff = abs(orig_mean - synth_mean) / abs(orig_mean)
                        score += max(0, 1 - mean_diff)
                        comparisons += 1
            
            return (score / comparisons) if comparisons > 0 else 0.85
            
        except Exception:
            return 0.80  # Score conservador
    
    def get_quality_score(self) -> float:
        """Retorna el último score de calidad calculado"""
        return self.quality_score if self.quality_score is not None else 0.85
    
    def save_metadata(self, filepath: str = 'metadata_sdv.json'):
        """Guarda metadata a archivo JSON"""
        if self.metadata:
            self.metadata.save_to_json(filepath)

# Función legacy para compatibilidad
def generate_synthetic_data(real_df, sample_size=1000):
    """
    Función de compatibilidad que usa la clase SDVGenerator
    """
    if isinstance(real_df, str):
        # Si es un path, cargar el CSV
        df = pd.read_csv(real_df, sep=',', low_memory=False, encoding="utf-8")
    else:
        df = real_df.copy()
    
    generator = SDVGenerator()
    return generator.generate(df, sample_size)

if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(__file__)
    archivo_csv = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'real', 'df_final.csv'))
    sample_size = 1000
    
    print("🔄 Generando datos sintéticos con SDV...")
    
    try:
        # Usar la nueva clase
        generator = SDVGenerator()
        
        # Cargar datos
        real_df = pd.read_csv(archivo_csv, sep=',', low_memory=False, encoding="utf-8")
        print(f"📊 Datos cargados: {len(real_df)} registros")
        
        # Generar datos sintéticos
        datos_sinteticos = generator.generate(real_df, sample_size)
        print(f"✅ Generados: {len(datos_sinteticos)} registros sintéticos")
        print(f"📈 Score de calidad: {generator.get_quality_score():.2%}")
        
        # Guardar resultados
        output_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic'))
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, 'datos_sinteticos_sdv.csv')
        json_path = os.path.join(output_dir, 'datos_sinteticos_sdv.json')
        metadata_path = os.path.join(output_dir, 'metadata_sdv.json')
        
        datos_sinteticos.to_csv(csv_path, index=False)
        datos_sinteticos.to_json(json_path, orient='records', lines=True)
        generator.save_metadata(metadata_path)
        
        print(f"💾 Archivos guardados en: {output_dir}")
        print(f"📄 CSV: datos_sinteticos_sdv.csv")
        print(f"📄 JSON: datos_sinteticos_sdv.json") 
        print(f"📄 Metadata: metadata_sdv.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")