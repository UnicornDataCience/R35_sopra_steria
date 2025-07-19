"""
Dynamic Pipeline Configuration - FASE 2 del Plan de Refactorización
Configuración dinámica de pipelines basada en el tipo de dataset detectado

Este módulo genera automáticamente configuraciones específicas para:
- Análisis de datos
- Generación sintética
- Validación médica
- Simulación temporal

Fecha: 2024-01-15
Versión: 1.0.0
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict
import logging

# Configurar logging
logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Tipos de análisis disponibles"""
    DESCRIPTIVE = "descriptive"
    TEMPORAL = "temporal"
    CORRELATIONAL = "correlational"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"

class SynthesisMethod(Enum):
    """Métodos de síntesis disponibles"""
    CTGAN = "ctgan"
    TVAE = "tvae"
    GAUSSIAN_COPULA = "gaussian_copula"
    MARGINAL_DISTRIBUTIONS = "marginal_distributions"
    BAYESIAN_NETWORK = "bayesian_network"

class ValidationLevel(Enum):
    """Niveles de validación médica"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CUSTOM = "custom"

@dataclass
class AnalysisConfig:
    """Configuración de análisis"""
    analysis_types: List[AnalysisType]
    key_columns: List[str]
    grouping_columns: List[str]
    temporal_column: Optional[str]
    target_columns: List[str]
    correlation_threshold: float
    outlier_detection: bool
    clustering_method: str
    custom_parameters: Dict[str, Any]

@dataclass
class SynthesisConfig:
    """Configuración de síntesis"""
    preferred_method: SynthesisMethod
    alternative_methods: List[SynthesisMethod]
    sample_size: int
    privacy_level: float
    constraint_preservation: List[str]
    categorical_columns: List[str]
    continuous_columns: List[str]
    datetime_columns: List[str]
    custom_constraints: Dict[str, Any]

@dataclass
class ValidationConfig:
    """Configuración de validación"""
    validation_level: ValidationLevel
    medical_rules: List[str]
    range_checks: Dict[str, tuple]
    consistency_checks: List[str]
    temporal_validation: bool
    cross_validation: bool
    custom_validators: List[str]

@dataclass
class SimulationConfig:
    """Configuración de simulación"""
    simulation_type: str
    time_steps: int
    progression_models: List[str]
    intervention_points: List[str]
    outcome_measures: List[str]
    stochastic_elements: List[str]
    custom_scenarios: Dict[str, Any]

class DynamicPipelineConfig:
    """Configurador dinámico de pipelines"""
    
    def __init__(self):
        self.config_templates = self._load_config_templates()
        self.medical_domains = self._load_medical_domains()
        self.validation_rules = self._load_validation_rules()
        
    def _load_config_templates(self) -> Dict[str, Any]:
        """Carga plantillas de configuración"""
        templates = {
            'covid19': {
                'analysis': {
                    'key_columns': ['edad', 'sexo', 'pcr_result', 'sintomas'],
                    'temporal_column': 'fecha_ingreso',
                    'target_columns': ['gravedad', 'dias_hospitalizacion'],
                    'clustering_method': 'kmeans',
                    'outlier_detection': True
                },
                'synthesis': {
                    'preferred_method': SynthesisMethod.TVAE,
                    'privacy_level': 0.8,
                    'constraint_preservation': ['age_range', 'gender_distribution'],
                    'sample_size': 1000
                },
                'validation': {
                    'validation_level': ValidationLevel.STANDARD,
                    'medical_rules': ['age_consistency', 'symptom_coherence'],
                    'range_checks': {'edad': (0, 120), 'dias_hospitalizacion': (0, 365)}
                }
            },
            'diabetes': {
                'analysis': {
                    'key_columns': ['age', 'glucose', 'bmi', 'blood_pressure'],
                    'temporal_column': 'date',
                    'target_columns': ['hba1c', 'complications'],
                    'clustering_method': 'hierarchical',
                    'outlier_detection': True
                },
                'synthesis': {
                    'preferred_method': SynthesisMethod.CTGAN,
                    'privacy_level': 0.9,
                    'constraint_preservation': ['glucose_range', 'bmi_consistency'],
                    'sample_size': 1500
                },
                'validation': {
                    'validation_level': ValidationLevel.STRICT,
                    'medical_rules': ['glucose_diabetes_correlation', 'medication_consistency'],
                    'range_checks': {'glucose': (70, 400), 'bmi': (15, 50)}
                }
            },
            'cardiovascular': {
                'analysis': {
                    'key_columns': ['age', 'systolic_bp', 'diastolic_bp', 'cholesterol'],
                    'temporal_column': 'admission_date',
                    'target_columns': ['cardiac_events', 'mortality'],
                    'clustering_method': 'dbscan',
                    'outlier_detection': True
                },
                'synthesis': {
                    'preferred_method': SynthesisMethod.GAUSSIAN_COPULA,
                    'privacy_level': 0.85,
                    'constraint_preservation': ['bp_correlation', 'risk_factors'],
                    'sample_size': 2000
                },
                'validation': {
                    'validation_level': ValidationLevel.STRICT,
                    'medical_rules': ['bp_consistency', 'cardiac_risk_factors'],
                    'range_checks': {'systolic_bp': (70, 250), 'diastolic_bp': (40, 150)}
                }
            },
            'general_medical': {
                'analysis': {
                    'key_columns': ['age', 'gender', 'diagnosis', 'treatment'],
                    'temporal_column': 'date',
                    'target_columns': ['outcome', 'length_of_stay'],
                    'clustering_method': 'kmeans',
                    'outlier_detection': True
                },
                'synthesis': {
                    'preferred_method': SynthesisMethod.TVAE,
                    'privacy_level': 0.7,
                    'constraint_preservation': ['age_gender_distribution'],
                    'sample_size': 1000
                },
                'validation': {
                    'validation_level': ValidationLevel.STANDARD,
                    'medical_rules': ['basic_consistency'],
                    'range_checks': {'age': (0, 120)}
                }
            }
        }
        return templates
    
    def _load_medical_domains(self) -> Dict[str, Any]:
        """Carga dominios médicos específicos"""
        domains = {
            'covid19': {
                'icd_codes': ['U07.1', 'U07.2', 'J12.8', 'J44.0'],
                'symptoms': ['fever', 'cough', 'dyspnea', 'fatigue', 'headache'],
                'medications': ['remdesivir', 'dexamethasone', 'tocilizumab'],
                'lab_tests': ['pcr', 'antigen', 'ct_scan', 'blood_gas'],
                'complications': ['ards', 'pneumonia', 'sepsis']
            },
            'diabetes': {
                'icd_codes': ['E10', 'E11', 'E13', 'E14'],
                'symptoms': ['polyuria', 'polydipsia', 'polyphagia', 'fatigue'],
                'medications': ['metformin', 'insulin', 'glyburide', 'glipizide'],
                'lab_tests': ['glucose', 'hba1c', 'ketones', 'microalbumin'],
                'complications': ['neuropathy', 'retinopathy', 'nephropathy']
            },
            'cardiovascular': {
                'icd_codes': ['I20', 'I21', 'I25', 'I50'],
                'symptoms': ['chest_pain', 'dyspnea', 'palpitations', 'syncope'],
                'medications': ['aspirin', 'statins', 'ace_inhibitors', 'beta_blockers'],
                'lab_tests': ['troponin', 'ck_mb', 'bnp', 'cholesterol'],
                'complications': ['heart_failure', 'arrhythmia', 'stroke']
            }
        }
        return domains
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Carga reglas de validación médica"""
        rules = {
            'age_consistency': {
                'rule': 'age >= 0 and age <= 120',
                'message': 'Age must be between 0 and 120'
            },
            'gender_values': {
                'rule': "gender in ['M', 'F', 'Male', 'Female', 'male', 'female']",
                'message': 'Gender must be valid value'
            },
            'glucose_diabetes_correlation': {
                'rule': 'if diabetes_diagnosis then glucose >= 126',
                'message': 'Diabetic patients should have elevated glucose'
            },
            'bp_consistency': {
                'rule': 'systolic_bp >= diastolic_bp',
                'message': 'Systolic BP must be higher than diastolic'
            },
            'symptom_coherence': {
                'rule': 'if severe_symptoms then hospitalization_required',
                'message': 'Severe symptoms should lead to hospitalization'
            }
        }
        return rules
    
    def generate_analysis_config(self, dataset_type: str, detected_columns: Dict[str, Any]) -> AnalysisConfig:
        """Genera configuración de análisis específica"""
        template = self.config_templates.get(dataset_type, self.config_templates['general_medical'])
        analysis_template = template['analysis']
        
        # Adaptar columnas detectadas
        key_columns = self._map_columns(analysis_template['key_columns'], detected_columns)
        target_columns = self._map_columns(analysis_template['target_columns'], detected_columns)
        
        # Detectar columna temporal
        temporal_column = self._detect_temporal_column(detected_columns)
        
        # Detectar columnas de agrupación
        grouping_columns = self._detect_grouping_columns(detected_columns)
        
        return AnalysisConfig(
            analysis_types=[AnalysisType.DESCRIPTIVE, AnalysisType.CORRELATIONAL],
            key_columns=key_columns,
            grouping_columns=grouping_columns,
            temporal_column=temporal_column,
            target_columns=target_columns,
            correlation_threshold=0.5,
            outlier_detection=analysis_template.get('outlier_detection', True),
            clustering_method=analysis_template.get('clustering_method', 'kmeans'),
            custom_parameters={}
        )
    
    def generate_synthesis_config(self, dataset_type: str, domain_patterns: List[str], 
                                 dataset_size: int) -> SynthesisConfig:
        """Genera configuración de síntesis específica"""
        template = self.config_templates.get(dataset_type, self.config_templates['general_medical'])
        synthesis_template = template['synthesis']
        
        # Calcular tamaño de muestra recomendado
        recommended_size = min(dataset_size * 2, synthesis_template['sample_size'])
        
        return SynthesisConfig(
            preferred_method=synthesis_template['preferred_method'],
            alternative_methods=[SynthesisMethod.CTGAN, SynthesisMethod.TVAE],
            sample_size=recommended_size,
            privacy_level=synthesis_template['privacy_level'],
            constraint_preservation=synthesis_template['constraint_preservation'],
            categorical_columns=[],  # Se detectará automáticamente
            continuous_columns=[],   # Se detectará automáticamente
            datetime_columns=[],     # Se detectará automáticamente
            custom_constraints={}
        )
    
    def generate_validation_rules(self, dataset_type: str, medical_domain: str) -> ValidationConfig:
        """Genera reglas de validación específicas"""
        template = self.config_templates.get(dataset_type, self.config_templates['general_medical'])
        validation_template = template['validation']
        
        # Obtener reglas médicas específicas del dominio
        domain_info = self.medical_domains.get(medical_domain, {})
        
        return ValidationConfig(
            validation_level=validation_template['validation_level'],
            medical_rules=validation_template['medical_rules'],
            range_checks=validation_template['range_checks'],
            consistency_checks=['age_consistency', 'gender_values'],
            temporal_validation=True,
            cross_validation=True,
            custom_validators=[]
        )
    
    def generate_simulation_config(self, dataset_type: str, temporal_columns: List[str]) -> SimulationConfig:
        """Genera configuración de simulación"""
        has_temporal = len(temporal_columns) > 0
        
        return SimulationConfig(
            simulation_type="longitudinal" if has_temporal else "cross_sectional",
            time_steps=30 if has_temporal else 1,
            progression_models=["linear", "exponential"],
            intervention_points=["baseline", "follow_up"],
            outcome_measures=["primary_outcome", "secondary_outcome"],
            stochastic_elements=["random_variation", "measurement_error"],
            custom_scenarios={}
        )
    
    def generate_complete_pipeline_config(self, dataset_type: str, detected_columns: Dict[str, Any], 
                                        domain_patterns: List[str], dataset_size: int) -> Dict[str, Any]:
        """Genera configuración completa del pipeline"""
        
        # Generar configuraciones específicas
        analysis_config = self.generate_analysis_config(dataset_type, detected_columns)
        synthesis_config = self.generate_synthesis_config(dataset_type, domain_patterns, dataset_size)
        validation_config = self.generate_validation_rules(dataset_type, dataset_type)
        simulation_config = self.generate_simulation_config(dataset_type, 
                                                          [analysis_config.temporal_column] if analysis_config.temporal_column else [])
        
        return {
            'dataset_type': dataset_type,
            'analysis': asdict(analysis_config),
            'synthesis': asdict(synthesis_config),
            'validation': asdict(validation_config),
            'simulation': asdict(simulation_config),
            'generated_at': pd.Timestamp.now().isoformat(),
            'version': '1.0.0'
        }
    
    def _map_columns(self, template_columns: List[str], detected_columns: Dict[str, Any]) -> List[str]:
        """Mapea columnas de plantilla a columnas detectadas"""
        mapped_columns = []
        
        for template_col in template_columns:
            # Buscar columna similar en el dataset
            for detected_col, info in detected_columns.items():
                if (template_col.lower() in detected_col.lower() or 
                    detected_col.lower() in template_col.lower()):
                    mapped_columns.append(detected_col)
                    break
        
        return mapped_columns
    
    def _detect_temporal_column(self, detected_columns: Dict[str, Any]) -> Optional[str]:
        """Detecta columna temporal principal"""
        temporal_indicators = ['date', 'time', 'timestamp', 'fecha', 'dia']
        
        for col_name, info in detected_columns.items():
            if (any(indicator in col_name.lower() for indicator in temporal_indicators) or
                'datetime' in str(info.get('data_type', '')).lower()):
                return col_name
        
        return None
    
    def _detect_grouping_columns(self, detected_columns: Dict[str, Any]) -> List[str]:
        """Detecta columnas de agrupación"""
        grouping_indicators = ['id', 'patient', 'subject', 'grupo', 'category']
        grouping_columns = []
        
        for col_name, info in detected_columns.items():
            if (any(indicator in col_name.lower() for indicator in grouping_indicators) or
                info.get('detected_type') == 'categorical'):
                grouping_columns.append(col_name)
        
        return grouping_columns
    
    def save_config(self, config: Dict[str, Any], filepath: str):
        """Guarda configuración en archivo"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuración guardada en {filepath}")
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
    
    def load_config(self, filepath: str) -> Dict[str, Any]:
        """Carga configuración desde archivo"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuración cargada desde {filepath}")
            return config
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return {}

# Función de conveniencia para uso rápido
def create_pipeline_config(dataset_type: str, detected_columns: Dict[str, Any], 
                          domain_patterns: List[str], dataset_size: int) -> Dict[str, Any]:
    """Función de conveniencia para crear configuración completa"""
    configurator = DynamicPipelineConfig()
    return configurator.generate_complete_pipeline_config(
        dataset_type, detected_columns, domain_patterns, dataset_size
    )

# Importar pandas para fechas
import pandas as pd

if __name__ == "__main__":
    # Ejemplo de uso
    configurator = DynamicPipelineConfig()
    
    # Simular columnas detectadas
    detected_columns = {
        'patient_id': {'data_type': 'int', 'detected_type': 'categorical'},
        'age': {'data_type': 'int', 'detected_type': 'numeric'},
        'gender': {'data_type': 'str', 'detected_type': 'categorical'},
        'fever': {'data_type': 'bool', 'detected_type': 'categorical'},
        'pcr_result': {'data_type': 'str', 'detected_type': 'categorical'},
        'admission_date': {'data_type': 'datetime', 'detected_type': 'temporal'}
    }
    
    # Generar configuración
    config = configurator.generate_complete_pipeline_config(
        'covid19', detected_columns, ['symptoms', 'lab_results'], 1000
    )
    
    print("Configuración generada:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
