"""
Módulo de Análisis Exploratorio de Datos (EDA) Completo

Este módulo genera estadísticas descriptivas completas, análisis de correlaciones,
valores nulos, y análisis de columnas para datasets médicos.

Autor: Patient-IA
Fecha: Octubre 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CompleteEDAAnalyzer:
    """
    Analizador EDA completo que genera estadísticas descriptivas detalladas,
    correlaciones, análisis de valores nulos, y patrones médicos.
    """
    
    def __init__(self):
        self.logger = logger
        
        # Patrones médicos comunes
        self.medical_patterns = {
            'patient_id': ['patient', 'id', 'paciente', 'record', 'numero'],
            'age': ['edad', 'age', 'anos', 'years'],
            'gender': ['sexo', 'gender', 'sex', 'genero'],
            'diagnosis': ['diagnostico', 'diagnosis', 'diag', 'enfermedad', 'disease'],
            'date': ['fecha', 'date', 'timestamp', 'datetime'],
            'outcome': ['outcome', 'resultado', 'estado', 'status', 'fallecido', 'deceased', 'survived'],
            'medication': ['farmaco', 'medication', 'drug', 'tratamiento', 'treatment']
        }
    
    def analyze(self, df: pd.DataFrame, sample_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Realiza un análisis EDA completo del dataset.
        
        Args:
            df: DataFrame a analizar
            sample_info: Información de muestreo si aplica
            
        Returns:
            Dict con análisis completo incluyendo:
            - basic_info: Información básica del dataset
            - columns: Análisis detallado por columna con estadísticas
            - correlations: Matriz de correlación y correlaciones altas
            - missing_values: Análisis de valores nulos
            - medical_patterns: Patrones médicos detectados
        """
        logger.info(f"🔬 Iniciando análisis EDA completo: {df.shape[0]} filas x {df.shape[1]} columnas")
        
        analysis = {}
        
        # 1. Información básica
        analysis['basic_info'] = self._analyze_basic_info(df)
        
        # 2. Análisis detallado de columnas
        analysis['columns'] = self._analyze_columns(df)
        
        # 3. Análisis de correlaciones
        analysis['correlations'] = self._analyze_correlations(df)
        
        # 4. Análisis de valores nulos
        analysis['missing_values'] = self._analyze_missing_values(df)
        
        # 5. Patrones médicos
        analysis['medical_patterns'] = self._detect_medical_patterns(df)
        
        # 6. Información de muestreo (si aplica)
        if sample_info:
            analysis['sampling_info'] = sample_info
        
        logger.info(f"✅ Análisis EDA completado")
        
        return analysis
    
    def _analyze_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza información básica del dataset"""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB",
            'duplicated_rows': int(df.duplicated().sum())
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analiza cada columna en detalle con estadísticas descriptivas.
        """
        columns_analysis = []
        
        for col in df.columns:
            col_data = df[col]
            col_info = {
                'name': col,
                'missing_count': int(col_data.isnull().sum()),
                'missing_percentage': float(col_data.isnull().sum() / len(df) * 100),
                'unique_count': int(col_data.nunique())
            }
            
            # Detectar tipo de columna
            if pd.api.types.is_numeric_dtype(col_data):
                col_info['type'] = 'numerical'
                col_info['stats'] = self._get_numerical_stats(col_data)
                
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_info['type'] = 'datetime'
                col_info['stats'] = self._get_datetime_stats(col_data)
                
            else:
                col_info['type'] = 'categorical'
                col_info['stats'] = self._get_categorical_stats(col_data)
            
            columns_analysis.append(col_info)
        
        return columns_analysis
    
    def _get_numerical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Obtiene estadísticas descriptivas para columnas numéricas"""
        try:
            desc = series.describe()
            return {
                'count': int(desc['count']),
                'mean': float(desc['mean']),
                'std': float(desc['std']),
                'min': float(desc['min']),
                '25%': float(desc['25%']),
                '50%': float(desc['50%']),
                '75%': float(desc['75%']),
                'max': float(desc['max']),
                'skewness': float(series.skew()) if len(series.dropna()) > 0 else None,
                'kurtosis': float(series.kurt()) if len(series.dropna()) > 0 else None
            }
        except Exception as e:
            logger.warning(f"Error calculando estadísticas numéricas para {series.name}: {e}")
            return {}
    
    def _get_datetime_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Obtiene estadísticas para columnas de fecha/hora"""
        try:
            non_null = series.dropna()
            if len(non_null) == 0:
                return {}
            
            return {
                'min_date': str(non_null.min()),
                'max_date': str(non_null.max()),
                'range_days': (non_null.max() - non_null.min()).days if len(non_null) > 0 else None
            }
        except Exception as e:
            logger.warning(f"Error calculando estadísticas de fecha para {series.name}: {e}")
            return {}
    
    def _get_categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Obtiene estadísticas para columnas categóricas"""
        try:
            value_counts = series.value_counts()
            top_values = value_counts.head(10).to_dict()
            
            return {
                'unique_values': int(series.nunique()),
                'top_values': [{str(k): int(v)} for k, v in top_values.items()],
                'mode': str(series.mode()[0]) if len(series.mode()) > 0 else None
            }
        except Exception as e:
            logger.warning(f"Error calculando estadísticas categóricas para {series.name}: {e}")
            return {}
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza correlaciones entre columnas numéricas.
        """
        try:
            # Seleccionar solo columnas numéricas
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty or len(numeric_df.columns) < 2:
                return {
                    'high_correlations': [],
                    'correlation_matrix': {}
                }
            
            # Calcular matriz de correlación
            corr_matrix = numeric_df.corr()
            
            # Encontrar correlaciones altas (|r| > 0.5, excluyendo diagonal)
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5 and not np.isnan(corr_value):
                        high_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': float(corr_value)
                        })
            
            # Ordenar por valor absoluto de correlación
            high_correlations = sorted(high_correlations, 
                                     key=lambda x: abs(x['correlation']), 
                                     reverse=True)
            
            return {
                'high_correlations': high_correlations,
                'total_numeric_columns': len(numeric_df.columns),
                'correlation_matrix': corr_matrix.to_dict() if len(corr_matrix) < 50 else {}
            }
            
        except Exception as e:
            logger.warning(f"Error calculando correlaciones: {e}")
            return {
                'high_correlations': [],
                'correlation_matrix': {}
            }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza valores nulos en el dataset.
        """
        try:
            missing_counts = df.isnull().sum()
            missing_pct = (missing_counts / len(df) * 100)
            
            # Filtrar solo columnas con valores nulos
            cols_with_missing = {col: int(count) for col, count in missing_counts.items() if count > 0}
            
            return {
                'total_missing': int(missing_counts.sum()),
                'columns_with_missing': cols_with_missing,
                'missing_percentage_by_column': {col: float(pct) for col, pct in missing_pct.items() if pct > 0}
            }
            
        except Exception as e:
            logger.warning(f"Error analizando valores nulos: {e}")
            return {
                'total_missing': 0,
                'columns_with_missing': {}
            }
    
    def _detect_medical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta patrones médicos comunes en el dataset.
        """
        detected = {
            'has_patient_id': False,
            'has_age': False,
            'has_gender': False,
            'has_dates': False,
            'has_diagnoses': False,
            'has_outcomes': False,
            'has_medications': False,
            'key_medical_columns': [],
            'patient_id_column': None,
            'age_columns': [],
            'date_columns': [],
            'diagnosis_columns': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Patient ID
            if any(pattern in col_lower for pattern in self.medical_patterns['patient_id']):
                detected['has_patient_id'] = True
                detected['patient_id_column'] = col
                detected['key_medical_columns'].append(col)
            
            # Age
            if any(pattern in col_lower for pattern in self.medical_patterns['age']):
                detected['has_age'] = True
                detected['age_columns'].append(col)
                detected['key_medical_columns'].append(col)
            
            # Gender
            if any(pattern in col_lower for pattern in self.medical_patterns['gender']):
                detected['has_gender'] = True
                detected['key_medical_columns'].append(col)
            
            # Diagnosis
            if any(pattern in col_lower for pattern in self.medical_patterns['diagnosis']):
                detected['has_diagnoses'] = True
                detected['diagnosis_columns'].append(col)
                detected['key_medical_columns'].append(col)
            
            # Date
            if any(pattern in col_lower for pattern in self.medical_patterns['date']) or \
               pd.api.types.is_datetime64_any_dtype(df[col]):
                detected['has_dates'] = True
                detected['date_columns'].append(col)
            
            # Outcome
            if any(pattern in col_lower for pattern in self.medical_patterns['outcome']):
                detected['has_outcomes'] = True
                detected['key_medical_columns'].append(col)
            
            # Medication
            if any(pattern in col_lower for pattern in self.medical_patterns['medication']):
                detected['has_medications'] = True
                detected['key_medical_columns'].append(col)
        
        return detected


def analyze_dataset_complete(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Función de conveniencia para análisis EDA completo.
    
    Args:
        df: DataFrame a analizar
        
    Returns:
        Dict con análisis completo
    """
    analyzer = CompleteEDAAnalyzer()
    return analyzer.analyze(df)
