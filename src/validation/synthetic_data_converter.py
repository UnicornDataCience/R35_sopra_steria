"""
Conversor para transformar datos sintéticos al formato JSON médico estándar
"""
import pandas as pd
from typing import Dict, Any

class SyntheticDataConverter:
    """Convierte datos sintéticos simples al formato JSON médico estándar"""
    
    def __init__(self):
        # Mapeo de columnas simples a formato JSON médico
        self.column_mapping = {
            'age': 'EDAD/AGE',
            'sex': 'SEXO/SEX', 
            'gender': 'SEXO/SEX',
            'diagnosis': 'DIAG ING/INPAT',
            'temperature': 'TEMP_ING/INPAT',
            'oxygen_saturation': 'SAT_02_ING/INPAT',
            'heart_rate': 'FC/HR_ING/INPAT',
            'systolic_bp': 'TA_MAX_ING/INPAT',
            'diastolic_bp': 'TA_MIN_ING/INPAT',
            'cholesterol': 'RESULTADO/VAL_RESULT',  # Usar como biomarcador
            'medication': 'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME',
            'patient_id': 'PATIENT ID',
            'patient_type': 'UCI_DIAS/ICU_DAYS'  # Mapear tipo de paciente a días UCI
        }
        
        # Mapeo de valores categóricos
        self.value_mapping = {
            'SEXO/SEX': {
                0: 'FEMALE',
                1: 'MALE',
                'F': 'FEMALE',
                'M': 'MALE'
            },
            'DIAG ING/INPAT': {
                'Hypertension': 'COVID19 - NEGATIVO',
                'Myocardial Infarction': 'COVID19 - POSITIVO',
                'Atrial Fibrillation': 'COVID19 - POSITIVO',
                'Heart Failure': 'COVID19 - POSITIVO'
            }
        }
    
    def convert_to_medical_json_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte DataFrame al formato JSON médico estándar"""
        if df.empty:
            return df
        
        # Verificar si ya está en formato médico
        if self.is_already_medical_format(df):
            print(f"[DEBUG] Datos ya están en formato médico. Columnas: {list(df.columns)}")
            # Solo asegurar que las columnas requeridas estén presentes
            required_columns = ['PATIENT ID', 'EDAD/AGE', 'SEXO/SEX', 'DIAG ING/INPAT']
            missing_required = [col for col in required_columns if col not in df.columns]
            
            if missing_required:
                print(f"[DEBUG] Faltan columnas requeridas: {missing_required}")
                converted_df = df.copy()
                # Agregar columnas faltantes con valores por defecto
                defaults = {
                    'PATIENT ID': [f'PAT_{i:03d}' for i in range(1, len(df) + 1)],
                    'EDAD/AGE': 45,
                    'SEXO/SEX': 'MALE',
                    'DIAG ING/INPAT': 'COVID19 - NEGATIVO'
                }
                for col in missing_required:
                    if col == 'PATIENT ID':
                        converted_df[col] = defaults[col]
                    else:
                        converted_df[col] = defaults[col]
                return converted_df
            else:
                print(f"[DEBUG] Todas las columnas requeridas están presentes")
                return df.copy()
            
        # Si no está en formato médico, hacer la conversión completa
        print(f"[DEBUG] Convirtiendo datos simples a formato médico. Columnas originales: {list(df.columns)}")
        converted_df = pd.DataFrame()
        
        # Generar ID de paciente si no existe
        if 'PATIENT ID' not in df.columns and 'patient_id' not in df.columns:
            converted_df['PATIENT ID'] = [f'PAT_{i:03d}' for i in range(1, len(df) + 1)]
        
        # Convertir columnas existentes
        for original_col in df.columns:
            if original_col.lower() in self.column_mapping:
                target_col = self.column_mapping[original_col.lower()]
                
                # Copiar datos con posible mapeo de valores
                if target_col in self.value_mapping:
                    converted_df[target_col] = df[original_col].map(
                        self.value_mapping[target_col]
                    ).fillna(df[original_col])  # Mantener valor original si no hay mapeo
                else:
                    converted_df[target_col] = df[original_col]
            else:
                # Mantener columnas que ya están en formato correcto
                converted_df[original_col] = df[original_col]
        
        # Agregar columnas obligatorias que falten con valores por defecto
        required_columns = {
            'EDAD/AGE': 45,  # Edad por defecto
            'SEXO/SEX': 'MALE',  # Sexo por defecto
            'DIAG ING/INPAT': 'COVID19 - NEGATIVO',  # Diagnóstico por defecto
            'TEMP_ING/INPAT': 36.5,  # Temperatura normal
            'SAT_02_ING/INPAT': 98,  # Saturación normal
            'RESULTADO/VAL_RESULT': 5.0,  # PCR normal
            'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME': 'PARACETAMOL',  # Medicamento por defecto
            'UCI_DIAS/ICU_DAYS': 0,  # Sin UCI por defecto
            'MOTIVO_ALTA/DESTINY_DISCHARGE_ING': 'ALTA DOMICILIO'  # Alta normal
        }
        
        for col, default_value in required_columns.items():
            if col not in converted_df.columns:
                converted_df[col] = default_value
        
        print(f"[DEBUG] Conversión completada. Columnas finales: {list(converted_df.columns)}")
        return converted_df
    
    def is_already_medical_format(self, df: pd.DataFrame) -> bool:
        """Verifica si el DataFrame ya está en formato médico JSON"""
        medical_indicators = ['EDAD/AGE', 'SEXO/SEX', 'DIAG ING/INPAT']
        # Si tiene al menos 2 de estos 3 indicadores principales, es formato médico
        found_indicators = sum(1 for col in medical_indicators if col in df.columns)
        return found_indicators >= 2
