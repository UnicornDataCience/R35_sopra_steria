import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

class DataExtractor:
    """Extractor de patrones clínicos de datasets médicos"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_patterns(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Extrae patrones clínicos relevantes del dataset
        
        Args:
            dataframe: DataFrame con datos clínicos
            
        Returns:
            Dict con patrones detectados, estadísticas y metadata
        """
        try:
            patterns = []
            statistics = {}
            
            # Análisis básico del dataset
            total_records = len(dataframe)
            total_features = len(dataframe.columns)
            
            # Detectar columnas relevantes
            clinical_columns = self._identify_clinical_columns(dataframe)
            
            # Extraer patrones demográficos
            demographic_patterns = self._extract_demographic_patterns(dataframe, clinical_columns)
            patterns.extend(demographic_patterns)
            
            # Extraer patrones de diagnósticos
            diagnosis_patterns = self._extract_diagnosis_patterns(dataframe, clinical_columns)
            patterns.extend(diagnosis_patterns)
            
            # Extraer patrones de medicamentos
            medication_patterns = self._extract_medication_patterns(dataframe, clinical_columns)
            patterns.extend(medication_patterns)
            
            # Extraer patrones temporales
            temporal_patterns = self._extract_temporal_patterns(dataframe, clinical_columns)
            patterns.extend(temporal_patterns)
            
            # Estadísticas generales
            statistics = {
                "total_records": total_records,
                "total_features": total_features,
                "missing_data_percentage": (dataframe.isnull().sum().sum() / (total_records * total_features)) * 100,
                "clinical_columns": clinical_columns,
                "data_types": dict(dataframe.dtypes.astype(str))
            }
            
            return {
                "clinical_patterns": patterns,
                "statistics": statistics,
                "metadata": {
                    "extraction_timestamp": pd.Timestamp.now().isoformat(),
                    "total_patterns_found": len(patterns)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting patterns: {e}")
            return self._fallback_extraction(dataframe)
    
    def _identify_clinical_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identifica columnas clínicas relevantes"""
        clinical_keywords = {
            "demographic": ["age", "edad", "sex", "sexo", "gender", "genero"],
            "diagnosis": ["diagnosis", "diagnostico", "icd", "cie", "condition"],
            "medication": ["medication", "medicamento", "drug", "farmaco", "prescr"],
            "temporal": ["date", "fecha", "time", "tiempo", "visit", "visita"],
            "vital_signs": ["pressure", "presion", "temp", "temperatura", "pulse", "pulso"],
            "lab_results": ["lab", "laboratorio", "test", "resultado", "value", "valor"]
        }
        
        identified = {category: [] for category in clinical_keywords.keys()}
        
        for column in df.columns:
            column_lower = column.lower()
            for category, keywords in clinical_keywords.items():
                if any(keyword in column_lower for keyword in keywords):
                    identified[category].append(column)
        
        return identified
    
    def _extract_demographic_patterns(self, df: pd.DataFrame, clinical_cols: Dict) -> List[str]:
        """Extrae patrones demográficos"""
        patterns = []
        
        # Análisis de edad
        age_cols = clinical_cols.get("demographic", [])
        for col in age_cols:
            if "age" in col.lower() or "edad" in col.lower():
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    mean_age = df[col].mean()
                    patterns.append(f"Edad promedio: {mean_age:.1f} años")
                    
                    if mean_age > 60:
                        patterns.append("Población predominantemente geriátrica (>60 años)")
                    elif mean_age > 40:
                        patterns.append("Población adulta (40-60 años)")
        
        # Análisis de sexo/género
        sex_cols = [col for col in clinical_cols.get("demographic", []) 
                   if any(keyword in col.lower() for keyword in ["sex", "sexo", "gender", "genero"])]
        
        for col in sex_cols:
            if col in df.columns:
                sex_dist = df[col].value_counts()
                if len(sex_dist) > 0:
                    patterns.append(f"Distribución por sexo: {sex_dist.to_dict()}")
        
        return patterns
    
    def _extract_diagnosis_patterns(self, df: pd.DataFrame, clinical_cols: Dict) -> List[str]:
        """Extrae patrones de diagnósticos"""
        patterns = []
        
        diagnosis_cols = clinical_cols.get("diagnosis", [])
        for col in diagnosis_cols:
            if col in df.columns:
                # Top diagnósticos
                top_diagnoses = df[col].value_counts().head(5)
                if len(top_diagnoses) > 0:
                    patterns.append(f"Diagnósticos frecuentes en '{col}': {list(top_diagnoses.index[:3])}")
                
                # Detectar comorbilidades comunes
                if df[col].str.contains('diabetes|diabetic', case=False, na=False).any():
                    diabetes_count = df[col].str.contains('diabetes|diabetic', case=False, na=False).sum()
                    patterns.append(f"Pacientes con diabetes detectados: {diabetes_count} ({diabetes_count/len(df)*100:.1f}%)")
                
                if df[col].str.contains('hypertension|hipertension', case=False, na=False).any():
                    htn_count = df[col].str.contains('hypertension|hipertension', case=False, na=False).sum()
                    patterns.append(f"Pacientes con hipertensión: {htn_count} ({htn_count/len(df)*100:.1f}%)")
        
        return patterns
    
    def _extract_medication_patterns(self, df: pd.DataFrame, clinical_cols: Dict) -> List[str]:
        """Extrae patrones de medicamentos"""
        patterns = []
        
        med_cols = clinical_cols.get("medication", [])
        for col in med_cols:
            if col in df.columns:
                # Medicamentos más frecuentes
                if df[col].dtype == 'object':
                    top_meds = df[col].value_counts().head(3)
                    if len(top_meds) > 0:
                        patterns.append(f"Medicamentos frecuentes: {list(top_meds.index)}")
                
                # Detectar patrones farmacológicos
                common_meds = ['metformin', 'lisinopril', 'atorvastatin', 'aspirin']
                for med in common_meds:
                    if df[col].str.contains(med, case=False, na=False).any():
                        count = df[col].str.contains(med, case=False, na=False).sum()
                        patterns.append(f"Uso de {med}: {count} pacientes")
        
        return patterns
    
    def _extract_temporal_patterns(self, df: pd.DataFrame, clinical_cols: Dict) -> List[str]:
        """Extrae patrones temporales"""
        patterns = []
        
        temporal_cols = clinical_cols.get("temporal", [])
        for col in temporal_cols:
            if col in df.columns:
                try:
                    # Convertir a datetime si es posible
                    df_temp = pd.to_datetime(df[col], errors='coerce')
                    if not df_temp.isnull().all():
                        date_range = df_temp.max() - df_temp.min()
                        patterns.append(f"Rango temporal en '{col}': {date_range.days} días")
                        
                        # Analizar distribución por año
                        years = df_temp.dt.year.value_counts().head(3)
                        if len(years) > 0:
                            patterns.append(f"Años con más registros: {list(years.index)}")
                except:
                    continue
        
        return patterns
    
    def _fallback_extraction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extracción básica en caso de error"""
        return {
            "clinical_patterns": [
                f"Dataset con {len(df)} registros y {len(df.columns)} columnas",
                f"Columnas disponibles: {list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}",
                "Análisis básico completado"
            ],
            "statistics": {
                "total_records": len(df),
                "total_features": len(df.columns),
                "columns": list(df.columns)
            },
            "metadata": {
                "extraction_timestamp": pd.Timestamp.now().isoformat(),
                "fallback_mode": True
            }
        }