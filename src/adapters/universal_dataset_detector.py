"""
Universal Dataset Detector - FASE 2 del Plan de Refactorización
Detecta automáticamente el tipo de dataset médico y sus patrones
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import re
import logging

# Configurar logger
logger = logging.getLogger(__name__)

class DatasetType(Enum):
    """Tipos de datasets médicos soportados"""
    COVID19 = "covid19"
    DIABETES = "diabetes"
    CARDIOVASCULAR = "cardiovascular"
    ONCOLOGY = "oncology"
    GENERAL_MEDICAL = "general_medical"
    LABORATORY = "laboratory"
    PHARMACY = "pharmacy"
    UNKNOWN = "unknown"

class ColumnType(Enum):
    """Tipos de columnas médicas"""
    PATIENT_ID = "patient_id"
    AGE = "age"
    GENDER = "gender"
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"
    LAB_RESULT = "lab_result"
    VITAL_SIGN = "vital_sign"
    DATE = "date"
    CLINICAL_VALUE = "clinical_value"
    TEXT_NOTE = "text_note"
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    UNKNOWN = "unknown"

@dataclass
class ColumnMapping:
    """Mapeo de columnas detectadas"""
    original_name: str
    detected_type: ColumnType
    confidence: float
    data_type: str
    sample_values: List[Any]
    null_percentage: float
    unique_values: int

@dataclass
class DomainPatterns:
    """Patrones específicos del dominio médico detectados"""
    medical_domain: DatasetType
    key_indicators: List[str]
    column_mappings: Dict[str, ColumnMapping]
    data_quality_score: float
    recommendations: List[str]

class UniversalDatasetDetector:
    """Detector universal de datasets médicos"""
    
    def __init__(self):
        self.domain_keywords = {
            DatasetType.COVID19: [
                "covid", "coronavirus", "sars", "pcr", "antigen", "vaccine", "ventilator",
                "icu", "oxygen", "saturation", "fever", "cough", "pneumonia", "quarantine"
            ],
            DatasetType.DIABETES: [
                "diabetes", "glucose", "insulin", "hba1c", "metformin", "blood_sugar",
                "diabetic", "hyperglycemia", "hypoglycemia", "glucometer"
            ],
            DatasetType.CARDIOVASCULAR: [
                "heart", "cardiac", "blood_pressure", "hypertension", "ecg", "ekg",
                "cholesterol", "triglycerides", "stent", "bypass", "arrhythmia"
            ],
            DatasetType.ONCOLOGY: [
                "cancer", "tumor", "oncology", "chemotherapy", "radiation", "biopsy",
                "metastasis", "malignant", "benign", "staging"
            ],
            DatasetType.LABORATORY: [
                "lab", "blood", "urine", "serum", "plasma", "hemoglobin", "creatinine",
                "bilirubin", "albumin", "protein", "electrolytes"
            ],
            DatasetType.PHARMACY: [
                "medication", "drug", "prescription", "dosage", "pill", "tablet",
                "injection", "pharmacy", "pharmaceutical", "adverse_effect"
            ]
        }
        
        self.column_patterns = {
            ColumnType.PATIENT_ID: [
                r'.*patient.*id.*', r'.*id.*patient.*', r'.*patient.*', r'.*id.*',
                r'.*identifier.*', r'.*subject.*', r'.*participant.*'
            ],
            ColumnType.AGE: [
                r'.*age.*', r'.*edad.*', r'.*years.*', r'.*años.*', r'.*birth.*age.*'
            ],
            ColumnType.GENDER: [
                r'.*sex.*', r'.*gender.*', r'.*sexo.*', r'.*género.*', r'.*male.*female.*'
            ],
            ColumnType.DIAGNOSIS: [
                r'.*diagnosis.*', r'.*diagnostic.*', r'.*diag.*', r'.*condition.*',
                r'.*disease.*', r'.*icd.*', r'.*enfermedad.*'
            ],
            ColumnType.MEDICATION: [
                r'.*medication.*', r'.*drug.*', r'.*medicine.*', r'.*farmaco.*',
                r'.*medicamento.*', r'.*treatment.*', r'.*therapy.*'
            ],
            ColumnType.LAB_RESULT: [
                r'.*lab.*', r'.*laboratory.*', r'.*test.*', r'.*result.*',
                r'.*valor.*', r'.*analisis.*', r'.*blood.*', r'.*urine.*'
            ],
            ColumnType.VITAL_SIGN: [
                r'.*vital.*', r'.*pressure.*', r'.*temperature.*', r'.*pulse.*',
                r'.*heart.*rate.*', r'.*bp.*', r'.*temp.*', r'.*sat.*'
            ],
            ColumnType.DATE: [
                r'.*date.*', r'.*fecha.*', r'.*time.*', r'.*timestamp.*',
                r'.*admission.*', r'.*discharge.*', r'.*visit.*'
            ]
        }

    def detect_dataset_type(self, df: pd.DataFrame) -> DatasetType:
        """Detecta el tipo de dataset médico"""
        
        # Obtener todo el texto del dataset para análisis
        text_content = self._extract_text_content(df)
        
        # Calcular scores por dominio
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = self._calculate_keyword_score(text_content, keywords)
            domain_scores[domain] = score
        
        # Encontrar el dominio con mayor score
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        
        # Verificar si el score es suficientemente alto
        if best_domain[1] > 0.1:  # Threshold mínimo
            return best_domain[0]
        else:
            return DatasetType.GENERAL_MEDICAL

    def infer_medical_columns(self, df: pd.DataFrame) -> Dict[str, ColumnMapping]:
        """Infiere el tipo de cada columna médica"""
        
        column_mappings = {}
        
        for col in df.columns:
            mapping = self._analyze_column(df, col)
            column_mappings[col] = mapping
        
        return column_mappings

    def extract_domain_patterns(self, df: pd.DataFrame) -> DomainPatterns:
        """Extrae patrones específicos del dominio"""
        
        # Detectar tipo de dataset
        dataset_type = self.detect_dataset_type(df)
        
        # Inferir mapeo de columnas
        column_mappings = self.infer_medical_columns(df)
        
        # Identificar indicadores clave
        key_indicators = self._identify_key_indicators(df, dataset_type)
        
        # Calcular score de calidad de datos
        quality_score = self._calculate_data_quality(df, column_mappings)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(df, dataset_type, column_mappings)
        
        return DomainPatterns(
            medical_domain=dataset_type,
            key_indicators=key_indicators,
            column_mappings=column_mappings,
            data_quality_score=quality_score,
            recommendations=recommendations
        )

    def _extract_text_content(self, df: pd.DataFrame) -> str:
        """Extrae todo el contenido de texto del DataFrame"""
        text_parts = []
        
        # Nombres de columnas
        text_parts.extend(df.columns.tolist())
        
        # Valores únicos de columnas categóricas/texto
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()[:10]  # Primeros 10 valores únicos
                text_parts.extend([str(val) for val in unique_vals])
        
        return ' '.join(text_parts).lower()

    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calcula score basado en presencia de keywords"""
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        matched_keywords = 0
        for keyword in keywords:
            if keyword.lower() in text:
                matched_keywords += 1
        
        return matched_keywords / len(keywords)

    def _analyze_column(self, df: pd.DataFrame, column: str) -> ColumnMapping:
        """Analiza una columna individual"""
        
        col_data = df[column]
        
        # Detectar tipo de columna
        detected_type = self._detect_column_type(column, col_data)
        
        # Calcular métricas
        null_percentage = col_data.isnull().sum() / len(col_data) * 100
        unique_values = col_data.nunique()
        sample_values = col_data.dropna().head(5).tolist()
        
        # Calcular confianza basada en patrones y contenido
        confidence = self._calculate_column_confidence(column, col_data, detected_type)
        
        return ColumnMapping(
            original_name=column,
            detected_type=detected_type,
            confidence=confidence,
            data_type=str(col_data.dtype),
            sample_values=sample_values,
            null_percentage=null_percentage,
            unique_values=unique_values
        )

    def _detect_column_type(self, column_name: str, col_data: pd.Series) -> ColumnType:
        """Detecta el tipo de columna médica"""
        
        col_name_lower = column_name.lower()
        
        # Buscar patrones en el nombre de la columna
        for col_type, patterns in self.column_patterns.items():
            for pattern in patterns:
                if re.search(pattern, col_name_lower):
                    return col_type
        
        # Análisis basado en contenido si no se encuentra patrón de nombre
        return self._detect_type_by_content(col_data)

    def _detect_type_by_content(self, col_data: pd.Series) -> ColumnType:
        """Detecta tipo basado en el contenido de la columna"""
        
        # Eliminar valores nulos para análisis
        clean_data = col_data.dropna()
        
        if len(clean_data) == 0:
            return ColumnType.UNKNOWN
        
        # Verificar si es numérico PRIMERO
        if pd.api.types.is_numeric_dtype(clean_data):
            # Distinguir entre ID, edad, y valores clínicos
            if clean_data.nunique() == len(clean_data):  # Valores únicos
                return ColumnType.PATIENT_ID
            elif clean_data.min() >= 0 and clean_data.max() <= 120:  # Rango de edad
                return ColumnType.AGE
            else:
                return ColumnType.CLINICAL_VALUE
        
        # Verificar si son fechas DESPUÉS
        if self._is_date_column(clean_data):
            return ColumnType.DATE
        
        # Para datos categóricos/texto
        unique_ratio = clean_data.nunique() / len(clean_data)
        
        if unique_ratio < 0.1:  # Pocos valores únicos
            return ColumnType.CATEGORICAL
        elif unique_ratio > 0.8:  # Muchos valores únicos
            return ColumnType.TEXT_NOTE
        else:
            return ColumnType.CATEGORICAL

    def _is_date_column(self, col_data: pd.Series) -> bool:
        """Verifica si una columna contiene fechas"""
        try:
            # Verificar si ya es tipo datetime
            if pd.api.types.is_datetime64_any_dtype(col_data):
                return True
            
            # Solo intentar parsear si parece contener fechas
            sample_values = col_data.dropna().astype(str).head(5)
            if len(sample_values) == 0:
                return False
            
            # Buscar patrones de fecha comunes
            import re
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                r'\d{1,2}/\d{1,2}/\d{2,4}',  # M/D/YY or MM/DD/YYYY
            ]
            
            for value in sample_values:
                has_date_pattern = any(re.search(pattern, str(value)) for pattern in date_patterns)
                if has_date_pattern:
                    # Solo entonces intentar parsear
                    try:
                        pd.to_datetime(col_data.head(10), errors='raise')
                        return True
                    except:
                        return False
            
            return False
        except:
            return False

    def _calculate_column_confidence(self, column_name: str, col_data: pd.Series, detected_type: ColumnType) -> float:
        """Calcula la confianza en la detección del tipo de columna"""
        
        confidence = 0.5  # Base confidence
        
        # Aumentar confianza si el patrón del nombre coincide
        col_name_lower = column_name.lower()
        if detected_type in self.column_patterns:
            for pattern in self.column_patterns[detected_type]:
                if re.search(pattern, col_name_lower):
                    confidence += 0.3
                    break
        
        # Ajustar por calidad de datos
        null_ratio = col_data.isnull().sum() / len(col_data)
        confidence *= (1 - null_ratio)  # Reducir si hay muchos nulos
        
        return min(confidence, 1.0)

    def _identify_key_indicators(self, df: pd.DataFrame, dataset_type: DatasetType) -> List[str]:
        """Identifica indicadores clave del tipo de dataset"""
        
        indicators = []
        
        # Agregar keywords encontrados
        if dataset_type in self.domain_keywords:
            text_content = self._extract_text_content(df)
            for keyword in self.domain_keywords[dataset_type]:
                if keyword.lower() in text_content:
                    indicators.append(keyword)
        
        # Agregar información sobre estructura de datos
        indicators.append(f"Columns: {len(df.columns)}")
        indicators.append(f"Rows: {len(df)}")
        
        return indicators

    def _calculate_data_quality(self, df: pd.DataFrame, column_mappings: Dict[str, ColumnMapping]) -> float:
        """Calcula un score de calidad general de los datos"""
        
        quality_factors = []
        
        # Factor 1: Porcentaje de valores no nulos
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.count().sum()
        completeness = non_null_cells / total_cells
        quality_factors.append(completeness)
        
        # Factor 2: Porcentaje de columnas identificadas correctamente
        identified_columns = sum(1 for mapping in column_mappings.values() 
                               if mapping.detected_type != ColumnType.UNKNOWN)
        identification_rate = identified_columns / len(column_mappings)
        quality_factors.append(identification_rate)
        
        # Factor 3: Confianza promedio en las detecciones
        avg_confidence = np.mean([mapping.confidence for mapping in column_mappings.values()])
        quality_factors.append(avg_confidence)
        
        return np.mean(quality_factors)

    def _generate_recommendations(self, df: pd.DataFrame, dataset_type: DatasetType, 
                                column_mappings: Dict[str, ColumnMapping]) -> List[str]:
        """Genera recomendaciones para mejorar el dataset"""
        
        recommendations = []
        
        # Recomendación sobre valores faltantes
        high_null_columns = [name for name, mapping in column_mappings.items() 
                           if mapping.null_percentage > 20]
        if high_null_columns:
            recommendations.append(f"Considerar imputación para columnas con muchos nulos: {', '.join(high_null_columns[:3])}")
        
        # Recomendación sobre tipos de datos
        unknown_columns = [name for name, mapping in column_mappings.items() 
                         if mapping.detected_type == ColumnType.UNKNOWN]
        if unknown_columns:
            recommendations.append(f"Revisar columnas no identificadas: {', '.join(unknown_columns[:3])}")
        
        # Recomendación específica por dominio
        if dataset_type == DatasetType.COVID19:
            recommendations.append("Verificar presencia de variables clave COVID-19: síntomas, resultados PCR, fechas de exposición")
        elif dataset_type == DatasetType.DIABETES:
            recommendations.append("Verificar presencia de métricas diabetes: glucosa, HbA1c, medicamentos antidiabéticos")
        
        # Recomendación sobre tamaño del dataset
        if len(df) < 100:
            recommendations.append("Dataset pequeño - considerar aumentar el tamaño de muestra para mejor generación sintética")
        
        return recommendations

    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Método principal para análisis completo del dataset
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Diccionario con resultados del análisis universal
        """
        try:
            # 1. Detectar tipo de dataset
            dataset_type = self.detect_dataset_type(df)
            
            # 2. Inferir columnas médicas
            column_mappings = self.infer_medical_columns(df)
            
            # 3. Extraer patrones de dominio
            domain_patterns = self.extract_domain_patterns(df)
            
            # 4. Generar mapeo de columnas mejorado
            column_inference = {}
            for name, mapping in column_mappings.items():
                column_inference[name] = {
                    'original_name': mapping.original_name,
                    'detected_type': mapping.detected_type.value,
                    'confidence': mapping.confidence,
                    'data_type': mapping.data_type,
                    'sample_values': mapping.sample_values,
                    'null_percentage': mapping.null_percentage,
                    'unique_values': mapping.unique_values
                }
            
            # 5. Crear lista de patrones de dominio
            domain_pattern_list = [
                f"Medical domain: {domain_patterns.medical_domain.value}",
                f"Data quality score: {domain_patterns.data_quality_score:.2f}"
            ]
            domain_pattern_list.extend(domain_patterns.key_indicators)
            domain_pattern_list.extend(domain_patterns.recommendations)
            
            # 6. Resultado completo
            return {
                'dataset_type': dataset_type.value,
                'is_covid_dataset': dataset_type == DatasetType.COVID19,
                'column_inference': column_inference,
                'domain_patterns': domain_pattern_list,
                'data_quality_score': domain_patterns.data_quality_score,
                'key_indicators': domain_patterns.key_indicators,
                'recommendations': domain_patterns.recommendations,
                'medical_domain': dataset_type.value,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en análisis universal: {e}")
            return {
                'dataset_type': 'unknown',
                'is_covid_dataset': False,
                'column_inference': {},
                'domain_patterns': [],
                'data_quality_score': 0.0,
                'key_indicators': [],
                'recommendations': [f"Error en análisis: {str(e)}"],
                'medical_domain': 'unknown',
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'error': True
            }

# Funciones de utilidad para integración con el sistema existente
def analyze_universal_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Función de conveniencia para análisis completo"""
    
    detector = UniversalDatasetDetector()
    patterns = detector.extract_domain_patterns(df)
    
    return {
        'dataset_type': patterns.medical_domain.value,
        'is_covid_dataset': patterns.medical_domain == DatasetType.COVID19,
        'column_mappings': {name: {
            'original_name': mapping.original_name,
            'detected_type': mapping.detected_type.value,
            'confidence': mapping.confidence,
            'data_type': mapping.data_type
        } for name, mapping in patterns.column_mappings.items()},
        'key_indicators': patterns.key_indicators,
        'data_quality_score': patterns.data_quality_score,
        'recommendations': patterns.recommendations,
        'cleaned_dataframe': df  # Para compatibilidad con código existente
    }
