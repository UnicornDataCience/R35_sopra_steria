"""
Medical Column Selector - Sistema de Selección Inteligente de Columnas
Permite al usuario seleccionar columnas para generación sintética con validación médica
"""

import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
# import logging
from src.utils.logging_config import get_logger

from .universal_dataset_detector import (
    UniversalDatasetDetector, DatasetType, ColumnType, ColumnMapping
)

logger = get_logger(__name__)

class MedicalRequirement(Enum):
    """Requisitos médicos mínimos para generación sintética"""
    MANDATORY = "mandatory"  # Obligatorio
    RECOMMENDED = "recommended"  # Recomendado
    OPTIONAL = "optional"  # Opcional

@dataclass
class ColumnRequirement:
    """Definición de requisito de columna"""
    column_type: ColumnType
    requirement_level: MedicalRequirement
    description: str
    alternatives: List[str] = None  # Nombres alternativos aceptables

@dataclass
class ColumnSelection:
    """Selección de columnas para generación sintética"""
    selected_columns: List[str]
    mandatory_fulfilled: bool
    missing_requirements: List[str]
    recommendations: List[str]
    quality_score: float

class MedicalColumnSelector:
    """Selector inteligente de columnas para datasets médicos"""
    
    def __init__(self):
        self.detector = UniversalDatasetDetector()
        
        # Requisitos mínimos por tipo de dataset
        self.medical_requirements = {
            DatasetType.COVID19: [
                ColumnRequirement(
                    ColumnType.PATIENT_ID, 
                    MedicalRequirement.MANDATORY,
                    "Identificador único del paciente"
                ),
                ColumnRequirement(
                    ColumnType.AGE, 
                    MedicalRequirement.MANDATORY,
                    "Edad del paciente"
                ),
                ColumnRequirement(
                    ColumnType.GENDER, 
                    MedicalRequirement.RECOMMENDED,
                    "Género del paciente"
                ),
                ColumnRequirement(
                    ColumnType.DIAGNOSIS, 
                    MedicalRequirement.MANDATORY,
                    "Diagnóstico principal (COVID-19 relacionado)"
                ),
                ColumnRequirement(
                    ColumnType.VITAL_SIGN, 
                    MedicalRequirement.RECOMMENDED,
                    "Signos vitales (temperatura, saturación O2, etc.)"
                ),
                ColumnRequirement(
                    ColumnType.LAB_RESULT, 
                    MedicalRequirement.OPTIONAL,
                    "Resultados de laboratorio (PCR, antígenos)"
                )
            ],
            DatasetType.DIABETES: [
                ColumnRequirement(
                    ColumnType.PATIENT_ID, 
                    MedicalRequirement.MANDATORY,
                    "Identificador único del paciente"
                ),
                ColumnRequirement(
                    ColumnType.AGE, 
                    MedicalRequirement.MANDATORY,
                    "Edad del paciente"
                ),
                ColumnRequirement(
                    ColumnType.GENDER, 
                    MedicalRequirement.RECOMMENDED,
                    "Género del paciente"
                ),
                ColumnRequirement(
                    ColumnType.DIAGNOSIS, 
                    MedicalRequirement.MANDATORY,
                    "Diagnóstico de diabetes (tipo 1, tipo 2, etc.)"
                ),
                ColumnRequirement(
                    ColumnType.LAB_RESULT, 
                    MedicalRequirement.RECOMMENDED,
                    "Resultados de glucosa, HbA1c, etc."
                ),
                ColumnRequirement(
                    ColumnType.MEDICATION, 
                    MedicalRequirement.OPTIONAL,
                    "Medicamentos para diabetes (insulina, metformina, etc.)"
                )
            ],
            DatasetType.GENERAL_MEDICAL: [
                ColumnRequirement(
                    ColumnType.PATIENT_ID, 
                    MedicalRequirement.MANDATORY,
                    "Identificador único del paciente"
                ),
                ColumnRequirement(
                    ColumnType.AGE, 
                    MedicalRequirement.MANDATORY,
                    "Edad del paciente"
                ),
                ColumnRequirement(
                    ColumnType.DIAGNOSIS, 
                    MedicalRequirement.MANDATORY,
                    "Diagnóstico principal o condición médica"
                ),
                ColumnRequirement(
                    ColumnType.GENDER, 
                    MedicalRequirement.RECOMMENDED,
                    "Género del paciente"
                )
            ]
        }
    
    def validate_medical_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Valida si el dataset cumple requisitos mínimos médicos"""
        
        # Detectar tipo de dataset
        dataset_type = self.detector.detect_dataset_type(df)
        
        # Inferir columnas médicas
        column_mappings = self.detector.infer_medical_columns(df)
        
        # Obtener requisitos para este tipo de dataset
        requirements = self.medical_requirements.get(
            dataset_type, 
            self.medical_requirements[DatasetType.GENERAL_MEDICAL]
        )
        
        validation_errors = []
        
        # Verificar requisitos obligatorios
        for req in requirements:
            if req.requirement_level == MedicalRequirement.MANDATORY:
                if not self._has_column_type(column_mappings, req.column_type):
                    validation_errors.append(
                        f"Falta columna obligatoria: {req.description} ({req.column_type.value})"
                    )
        
        # Verificar tamaño mínimo del dataset
        if len(df) < 50:
            validation_errors.append("Dataset muy pequeño (mínimo 50 filas para generación sintética)")
        
        # Verificar que no todas las columnas sean texto
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) < 2:
            validation_errors.append("Dataset debe tener al menos 2 columnas numéricas")
        
        is_valid = len(validation_errors) == 0
        return is_valid, validation_errors
    
    def generate_column_selection_interface(self, df: pd.DataFrame) -> ColumnSelection:
        """Genera interfaz Streamlit para selección de columnas"""
        
        st.subheader("🔍 Selección de Columnas para Generación Sintética")
        
        # Detectar tipo de dataset y columnas
        dataset_type = self.detector.detect_dataset_type(df)
        column_mappings = self.detector.infer_medical_columns(df)
        
        st.info(f"**Tipo de Dataset Detectado:** {dataset_type.value.title()}")
        
        # Mostrar información del dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas", len(df))
        with col2:
            st.metric("Columnas Totales", len(df.columns))
        with col3:
            st.metric("Columnas Numéricas", len(df.select_dtypes(include=['int64', 'float64']).columns))
        
        # Análisis automático de columnas
        st.subheader("📋 Análisis Automático de Columnas")
        
        # Crear categorías de columnas
        categorized_columns = self._categorize_columns(column_mappings)
        
        # Mostrar recomendaciones automáticas
        recommended_columns = self._get_recommended_columns(dataset_type, column_mappings, df)
        
        st.success(f"**Columnas Recomendadas Automáticamente:** {len(recommended_columns)}")
        
        # Interfaz de selección manual
        st.subheader("✅ Selección Manual de Columnas")
        
        selected_columns = []
        
        # Mostrar columnas por categoría con checkboxes
        for category, columns in categorized_columns.items():
            if columns:
                st.write(f"**{category}:**")
                for col in columns:
                    mapping = column_mappings[col]
                    is_recommended = col in recommended_columns
                    
                    # Checkbox con información adicional
                    default_value = is_recommended
                    selected = st.checkbox(
                        f"{col} ({mapping.detected_type.value}) - Confianza: {mapping.confidence:.2f}",
                        value=default_value,
                        key=f"col_{col}",
                        help=f"Tipo: {mapping.data_type}, Valores únicos: {mapping.unique_values}, Nulos: {mapping.null_percentage:.1f}%"
                    )
                    
                    if selected:
                        selected_columns.append(col)
        
        # Validar selección
        validation_result = self._validate_selection(selected_columns, dataset_type, column_mappings)
        
        # Mostrar resultados de validación
        if validation_result.mandatory_fulfilled:
            st.success("✅ Selección válida para generación sintética")
        else:
            st.error("❌ Selección incompleta")
            for missing in validation_result.missing_requirements:
                st.warning(f"⚠️ {missing}")
        
        # Mostrar recomendaciones
        if validation_result.recommendations:
            st.info("💡 **Recomendaciones:**")
            for rec in validation_result.recommendations:
                st.write(f"• {rec}")
        
        # Score de calidad
        st.metric("Puntuación de Calidad", f"{validation_result.quality_score:.2f}/1.0")
        
        return validation_result
    
    def _categorize_columns(self, column_mappings: Dict[str, ColumnMapping]) -> Dict[str, List[str]]:
        """Categoriza columnas por tipo detectado"""
        
        categories = {
            "Identificadores": [],
            "Demográficos": [],
            "Clínicos": [],
            "Laboratorio": [],
            "Medicamentos": [],
            "Fechas": [],
            "Otros": []
        }
        
        for col, mapping in column_mappings.items():
            if mapping.detected_type == ColumnType.PATIENT_ID:
                categories["Identificadores"].append(col)
            elif mapping.detected_type in [ColumnType.AGE, ColumnType.GENDER]:
                categories["Demográficos"].append(col)
            elif mapping.detected_type in [ColumnType.DIAGNOSIS, ColumnType.VITAL_SIGN]:
                categories["Clínicos"].append(col)
            elif mapping.detected_type == ColumnType.LAB_RESULT:
                categories["Laboratorio"].append(col)
            elif mapping.detected_type == ColumnType.MEDICATION:
                categories["Medicamentos"].append(col)
            elif mapping.detected_type == ColumnType.DATE:
                categories["Fechas"].append(col)
            else:
                categories["Otros"].append(col)
        
        return categories
    
    def _get_recommended_columns(self, dataset_type: DatasetType, column_mappings: Dict[str, ColumnMapping], df: pd.DataFrame) -> List[str]:
        """Obtiene columnas recomendadas automáticamente"""
        
        recommended = []
        requirements = self.medical_requirements.get(
            dataset_type, 
            self.medical_requirements[DatasetType.GENERAL_MEDICAL]
        )
        
        # Añadir columnas obligatorias y recomendadas
        for req in requirements:
            if req.requirement_level in [MedicalRequirement.MANDATORY, MedicalRequirement.RECOMMENDED]:
                matching_cols = self._find_columns_by_type(column_mappings, req.column_type)
                if matching_cols:
                    # Tomar la columna con mayor confianza
                    best_col = max(matching_cols, key=lambda x: column_mappings[x].confidence)
                    recommended.append(best_col)
        
        # Para COVID-19, limitar a 10 columnas como máximo
        if dataset_type == DatasetType.COVID19 and len(recommended) > 10:
            # Ordenar por confianza y tomar las 10 mejores
            recommended.sort(key=lambda x: column_mappings[x].confidence, reverse=True)
            recommended = recommended[:10]
        
        # Para otros datasets, incluir columnas numéricas importantes
        elif dataset_type != DatasetType.COVID19:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if col not in recommended and len(recommended) < 15:  # Límite flexible
                    recommended.append(col)
        
        return recommended
    
    def _find_columns_by_type(self, column_mappings: Dict[str, ColumnMapping], column_type: ColumnType) -> List[str]:
        """Encuentra columnas que coinciden con un tipo específico"""
        return [col for col, mapping in column_mappings.items() if mapping.detected_type == column_type]
    
    def _has_column_type(self, column_mappings: Dict[str, ColumnMapping], column_type: ColumnType) -> bool:
        """Verifica si existe al menos una columna del tipo especificado"""
        return len(self._find_columns_by_type(column_mappings, column_type)) > 0
    
    def _validate_selection(self, selected_columns: List[str], dataset_type: DatasetType, column_mappings: Dict[str, ColumnMapping]) -> ColumnSelection:
        """Valida la selección de columnas del usuario"""
        
        requirements = self.medical_requirements.get(
            dataset_type, 
            self.medical_requirements[DatasetType.GENERAL_MEDICAL]
        )
        
        missing_requirements = []
        recommendations = []
        
        # Verificar requisitos obligatorios
        for req in requirements:
            if req.requirement_level == MedicalRequirement.MANDATORY:
                matching_cols = [col for col in selected_columns if column_mappings[col].detected_type == req.column_type]
                if not matching_cols:
                    missing_requirements.append(f"Falta {req.description}")
        
        # Verificar número mínimo de columnas
        if len(selected_columns) < 3:
            missing_requirements.append("Seleccione al menos 3 columnas")
        
        # Generar recomendaciones
        if len(selected_columns) > 20:
            recommendations.append("Considere reducir el número de columnas para mejor rendimiento")
        
        if dataset_type == DatasetType.COVID19 and len(selected_columns) > 10:
            recommendations.append("Para datasets COVID-19, se recomienda máximo 10 columnas")
        
        # Calcular score de calidad
        quality_score = self._calculate_selection_quality(selected_columns, column_mappings, requirements)
        
        mandatory_fulfilled = len(missing_requirements) == 0
        
        return ColumnSelection(
            selected_columns=selected_columns,
            mandatory_fulfilled=mandatory_fulfilled,
            missing_requirements=missing_requirements,
            recommendations=recommendations,
            quality_score=quality_score
        )
    
    def _calculate_selection_quality(self, selected_columns: List[str], column_mappings: Dict[str, ColumnMapping], requirements: List[ColumnRequirement]) -> float:
        """Calcula un score de calidad para la selección"""
        
        if not selected_columns:
            return 0.0
        
        total_score = 0.0
        max_score = 0.0
        
        # Score por cumplimiento de requisitos
        for req in requirements:
            weight = 1.0 if req.requirement_level == MedicalRequirement.MANDATORY else 0.5
            max_score += weight
            
            matching_cols = [col for col in selected_columns if column_mappings[col].detected_type == req.column_type]
            if matching_cols:
                # Score basado en la confianza de la mejor columna
                best_confidence = max(column_mappings[col].confidence for col in matching_cols)
                total_score += weight * best_confidence
        
        # Score por diversidad de tipos de columnas
        unique_types = set(column_mappings[col].detected_type for col in selected_columns)
        diversity_score = min(len(unique_types) / 5.0, 1.0)  # Máximo 5 tipos diferentes
        total_score += diversity_score
        max_score += 1.0
        
        return total_score / max_score if max_score > 0 else 0.0
