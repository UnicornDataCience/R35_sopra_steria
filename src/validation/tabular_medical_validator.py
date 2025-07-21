"""
Validador específico para datos médicos en formato tabular (CSV/DataFrame)
"""
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from typing import Dict, List, Any, Tuple, Optional

class TabularMedicalValidator:
    """Validador especializado para datos médicos en formato CSV/DataFrame"""
    
    def __init__(self):
        # Lista expandida de medicamentos conocidos para fuzzy matching
        self.known_medications = [
            # COVID-19 medications
            "DEXAMETASONA", "AZITROMICINA", "ENOXAPARINA", "FUROSEMIDA", 
            "REMDESIVIR", "TOCILIZUMAB", "METILPREDNISOLONA", "HEPARINA",
            "CEFTRIAXONA", "LEVOFLOXACINO", "PREDNISONA", "OSELTAMIVIR",
            "HYDROXYCHLOROQUINE", "DOLQUINE", "HIDROXICLOROQUINA",
            
            # Cardiology medications  
            "ATORVASTATINA", "SIMVASTATINA", "ENALAPRIL", "LOSARTAN", 
            "METOPROLOL", "BISOPROLOL", "AMLODIPINO", "VALSARTAN",
            "CLOPIDOGREL", "ASPIRINA", "WARFARINA", "RIVAROXABAN",
            "CARVEDILOL", "LISINOPRIL", "ROSUVASTATIN", "DILTIAZEM",
            
            # Common medications
            "PARACETAMOL", "IBUPROFENO", "TRAMADOL", "OMEPRAZOL",
            "PANTOPRAZOL", "METAMIZOL", "DICLOFENACO", "NAPROXENO",
            "AMOXICILINA", "CIPROFLOXACINO", "METFORMINA", "INSULINA",
            "MORFINA", "ACETILCISTEINA", "ACIDO ASCORBICO", "ALCOHOL",
            
            # Vitaminas y suplementos
            "ACFOL", "FOLICO", "VITAMINA", "ASCORBICO", "TIAMINA",
            "COMPLEJO B", "CALCIO", "MAGNESIO", "POTASIO",
            
            # Immunosuppressive and specialty drugs
            "SANDIMMUN", "NEORAL", "CICLOSPORINA", "TACROLIMUS",
            "MICOFENOLATO", "SIROLIMUS", "EVEROLIMUS",
            
            # Additional common drugs from synthetic data
            "AGUA BIDESTILADA", "SUERO SALINO", "GLUCOSA", "MANITOL",
            "ALBUMINA", "PLASMA", "FACTOR VIII", "FACTOR IX",
            
            # Standard cardiology from original
            "Metoprolol", "Atorvastatin", "Lisinopril", "Warfarin", "None",
            
            # Synthetic/generic names for testing
            "MEDICOMP_A", "MEDICOMP_B", "TRATAMIENTO_X", "FARMACO_Y",
            "COMPUESTO_1", "MEDICINA_2", "DROGA_3", "PILL_4"
        ]
        
        # Rangos normales para validación clínica (más flexibles)
        self.clinical_ranges = {
            # Columnas estándar (para datos sintéticos simples) - más flexibles
            'age': {'min': 0, 'max': 150, 'type': 'numeric'},  # Más amplio
            'sex': {'valid_values': [0, 1, '0', '1', 'M', 'F', 'MALE', 'FEMALE'], 'type': 'categorical'},
            'diabetes': {'valid_values': [0, 1], 'type': 'binary'},
            'hypertension': {'valid_values': [0, 1], 'type': 'binary'},
            'patient_type': {'valid_values': [1, 2], 'type': 'categorical'},
            'pneumonia': {'valid_values': [0, 1], 'type': 'binary'},
            'copd': {'valid_values': [0, 1], 'type': 'binary'},
            'asthma': {'valid_values': [0, 1], 'type': 'binary'},
            'inmsupr': {'valid_values': [0, 1], 'type': 'binary'},
            'cardiovascular': {'valid_values': [0, 1], 'type': 'binary'},
            'pcr': {'min': 0, 'max': 1000, 'type': 'numeric'},  # Más amplio
            'temperature': {'min': 30, 'max': 45, 'type': 'numeric'},  # Más amplio
            'oxygen_saturation': {'min': 40, 'max': 110, 'type': 'numeric'},  # Más amplio
            
            # Columnas reales del dataset COVID (formato columnas del CSV real) - más flexibles
            'edad/age': {'min': 0, 'max': 150, 'type': 'numeric'},
            'sexo/sex': {'valid_values': ['MALE', 'FEMALE', 'M', 'F', '0', '1', 0, 1], 'type': 'categorical'},
            'temp_ing/inpat': {'min': 30, 'max': 45, 'type': 'numeric'},  # Más amplio
            'sat_02_ing/inpat': {'min': 40, 'max': 110, 'type': 'numeric'},  # Más amplio
            'fc/hr_ing/inpat': {'min': 20, 'max': 250, 'type': 'numeric'},  # Más amplio
            'ta_min_ing/inpat': {'min': 30, 'max': 150, 'type': 'numeric'},  # Más amplio
            'ta_max_ing/inpat': {'min': 60, 'max': 250, 'type': 'numeric'},  # Más amplio
            'uci_dias/icu_days': {'min': 0, 'max': 365, 'type': 'numeric'},  # Más amplio
            'resultado/val_result': {'min': -10, 'max': 1000, 'type': 'numeric'},  # Permite negativos
            'diag ing/inpat': {'type': 'text'},  # Más flexible, no lista fija
            
            # Columnas de cardiología (cardiology_fict_data.csv) - más flexibles
            'gender': {'valid_values': ['M', 'F', 'MALE', 'FEMALE', '0', '1', 0, 1], 'type': 'categorical'},
            'diagnosis': {'type': 'text'},  # Más flexible
            'systolic_bp': {'min': 40, 'max': 300, 'type': 'numeric'},  # Más amplio
            'diastolic_bp': {'min': 20, 'max': 200, 'type': 'numeric'},  # Más amplio
            'cholesterol': {'min': 50, 'max': 600, 'type': 'numeric'},  # Más amplio
            'heart_rate': {'min': 20, 'max': 250, 'type': 'numeric'},  # Más amplio
            'ejection_fraction': {'min': 10, 'max': 90, 'type': 'numeric'},  # Más amplio
            'medication': {'type': 'medication'}  # Usar validación fuzzy
        }
        
        # Correlaciones clínicas esperadas
        self.clinical_correlations = [
            ('diabetes', 'cardiovascular'),  # Diabetes y problemas cardiovasculares correlacionan
            ('copd', 'pneumonia'),           # COPD aumenta riesgo de neumonía
            ('age', 'hypertension'),         # Hipertensión más común en mayores
            ('age', 'diabetes'),             # Diabetes tipo 2 más común en mayores
            # Correlaciones cardiovasculares
            ('age', 'systolic_bp'),          # Presión sistólica aumenta con la edad
            ('age', 'cholesterol'),          # Colesterol tiende a aumentar con la edad
            ('systolic_bp', 'diastolic_bp'), # Presiones sistólica y diastólica correlacionan
            ('cholesterol', 'systolic_bp')   # Colesterol alto y presión alta correlacionan
        ]
    
    def _similarity(self, a: str, b: str) -> float:
        """Calcula la similitud entre dos strings (0.0 a 1.0)"""
        return SequenceMatcher(None, a.upper(), b.upper()).ratio()
    
    def _fuzzy_match_medication(self, medication: str, threshold: float = 0.6) -> Optional[str]:
        """
        Encuentra coincidencias parciales/fuzzy para medicamentos.
        Versión mejorada con múltiples estrategias de matching.
        
        Args:
            medication: Medicamento a buscar
            threshold: Umbral de similitud (0.0 a 1.0)
            
        Returns:
            El medicamento más similar o None
        """
        if not medication or pd.isna(medication):
            return None
            
        medication_str = str(medication).strip()
        if not medication_str:
            return None
        
        # Múltiples estrategias de limpieza y matching
        variants = []
        
        # 1. Original
        variants.append(medication_str.upper())
        
        # 2. Sin dosis ni formas farmacéuticas
        clean1 = re.sub(r'\s+(COMP|SOL|AMP|CAP|MG|ML|G|CAPSULE|TABLET)\s*\d*.*$', '', medication_str.upper())
        clean1 = re.sub(r'\s+\d+\s*(MG|ML|G|%).*$', '', clean1)
        variants.append(clean1.strip())
        
        # 3. Solo la primera palabra (principio activo)
        first_word = medication_str.upper().split()[0] if medication_str.split() else ""
        if first_word and len(first_word) > 2:
            variants.append(first_word)
        
        # 4. Primeras dos palabras
        words = medication_str.upper().split()
        if len(words) >= 2:
            variants.append(f"{words[0]} {words[1]}")
        
        # 5. Sin espacios ni guiones
        no_spaces = re.sub(r'[\s\-_/]', '', medication_str.upper())
        variants.append(no_spaces)
        
        best_match = None
        best_score = 0.0
        
        # Probar cada variante contra cada medicamento conocido
        for variant in variants:
            if not variant or len(variant) < 2:
                continue
                
            for known in self.known_medications:
                known_clean = known.upper().strip()
                
                # 1. Coincidencia exacta (prioridad máxima)
                if variant == known_clean:
                    return known
                
                # 2. Contenido: variant está en known o viceversa
                if variant in known_clean or known_clean in variant:
                    score = 0.95
                
                # 3. Comienza con la misma palabra
                elif variant.split()[0] == known_clean.split()[0] if variant.split() and known_clean.split() else False:
                    score = 0.85
                
                # 4. Similitud por diflib
                else:
                    score = self._similarity(variant, known_clean)
                
                # Bonificación por longitud similar
                len_ratio = min(len(variant), len(known_clean)) / max(len(variant), len(known_clean))
                if len_ratio > 0.7:
                    score += 0.1
                
                if score >= threshold and score > best_score:
                    best_score = score
                    best_match = known
        
        return best_match
    
    def _validate_medication_field(self, values: pd.Series) -> float:
        """
        Valida medicamentos usando fuzzy matching.
        
        Returns:
            Porcentaje de medicamentos reconocidos (0.0 a 1.0)
        """
        if values.empty:
            return 0.0
        
        # Filtrar valores no válidos
        valid_values = values.dropna()
        valid_values = valid_values[valid_values.astype(str).str.strip() != '']
        
        if len(valid_values) == 0:
            return 0.0
        
        recognized_count = 0
        for medication in valid_values:
            if self._fuzzy_match_medication(str(medication)):
                recognized_count += 1
        
        return recognized_count / len(valid_values)
    
    def _safe_numeric_conversion(self, series: pd.Series) -> pd.Series:
        """
        Convierte una serie a numérica de forma segura, manejando strings y valores faltantes.
        """
        try:
            # Primero intentar conversión directa
            return pd.to_numeric(series, errors='coerce')
        except Exception:
            return series.fillna(0)
    
    def _safe_string_conversion(self, series: pd.Series) -> pd.Series:
        """
        Convierte una serie a string de forma segura.
        """
        return series.astype(str).str.upper().replace('NAN', '')

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valida la calidad estructural de los datos tabulares"""
        results = {
            'total_records': len(df),
            'missing_values_score': 0.0,
            'data_types_score': 0.0,
            'range_validation_score': 0.0,
            'overall_quality_score': 0.0,
            'issues': []
        }
        
        if df.empty:
            results['issues'].append("Dataset vacío")
            return results
        
        # 1. Validar valores faltantes
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        results['missing_values_score'] = max(0, (100 - missing_percentage) / 100)
        
        if missing_percentage > 10:
            results['issues'].append(f"Alto porcentaje de valores faltantes: {missing_percentage:.1f}%")
        
        # 2. Validar tipos de datos y rangos
        valid_columns = 0
        range_valid_count = 0
        total_values = 0
        
        for column in df.columns:
            # Crear una clave normalizada para la búsqueda (minúsculas)
            column_key = column.lower()
            
            if column_key in self.clinical_ranges:
                valid_columns += 1
                range_config = self.clinical_ranges[column_key]
                
                if range_config['type'] == 'numeric':
                    # Validar rango numérico con conversión segura
                    min_val, max_val = range_config['min'], range_config['max']
                    
                    # Convertir columna a numérica de forma segura
                    numeric_data = self._safe_numeric_conversion(df[column])
                    
                    # Excluir valores 0.0 que pueden ser datos faltantes
                    non_zero_data = numeric_data[(numeric_data != 0.0) & (numeric_data.notna())]
                    
                    if len(non_zero_data) > 0:
                        valid_range = non_zero_data.between(min_val, max_val, inclusive='both')
                        range_valid_count += valid_range.sum()
                        total_values += len(non_zero_data)
                        
                        if not valid_range.all():
                            invalid_count = (~valid_range).sum()
                            results['issues'].append(f"Columna '{column}': {invalid_count} valores fuera del rango [{min_val}, {max_val}]")
                
                elif range_config['type'] in ['categorical', 'binary']:
                    # Validar valores categóricos con conversión segura
                    valid_values = set(str(v).upper() for v in range_config['valid_values'])
                    string_data = self._safe_string_conversion(df[column])
                    actual_values = set(string_data.dropna().unique())
                    actual_values.discard('')  # Remover strings vacíos
                    invalid_values = actual_values - valid_values
                    
                    if invalid_values:
                        results['issues'].append(f"Columna '{column}': valores inválidos encontrados: {invalid_values}")
                    else:
                        valid_count = len(string_data.dropna()) - (1 if '' in string_data.values else 0)
                        range_valid_count += valid_count
                        total_values += valid_count
                
                elif range_config['type'] == 'medication':
                    # Validar medicamentos usando fuzzy matching
                    medication_score = self._validate_medication_field(df[column])
                    valid_medications = int(medication_score * len(df[column].dropna()))
                    range_valid_count += valid_medications
                    total_values += len(df[column].dropna())
                    
                    if medication_score < 0.8:  # Si menos del 80% son reconocidos
                        results['issues'].append(f"Columna '{column}': solo {medication_score:.1%} de medicamentos reconocidos")
                
                elif range_config['type'] == 'text':
                    # Validación flexible para campos de texto
                    text_data = df[column].dropna()
                    valid_text_count = len(text_data[text_data.astype(str).str.strip() != ''])
                    range_valid_count += valid_text_count
                    total_values += len(text_data)
        
        # Calcular puntuaciones
        results['data_types_score'] = valid_columns / max(1, len(df.columns))
        results['range_validation_score'] = range_valid_count / max(1, total_values) if total_values > 0 else 0
        
        # Puntuación general de calidad
        results['overall_quality_score'] = (
            results['missing_values_score'] * 0.3 +
            results['data_types_score'] * 0.3 +
            results['range_validation_score'] * 0.4
        )
        
        return results
    
    def validate_clinical_coherence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valida coherencia clínica específica para datos tabulares"""
        results = {
            'coherence_score': 0.0,
            'correlation_score': 0.0,
            'logic_violations': 0,
            'total_checks': 0,
            'issues': []
        }
        
        if df.empty:
            return results
        
        # 1. Validaciones lógicas básicas
        logic_checks = 0
        logic_violations = 0
        
        # Edad vs condiciones (personas jóvenes con múltiples comorbilidades es menos común)
        if 'age' in df.columns and 'diabetes' in df.columns and 'hypertension' in df.columns:
            # Convertir edad a numérica de forma segura
            age_numeric = self._safe_numeric_conversion(df['age'])
            diabetes_numeric = self._safe_numeric_conversion(df['diabetes'])
            hypertension_numeric = self._safe_numeric_conversion(df['hypertension'])
            
            young_with_multiple_conditions = df[
                (age_numeric < 30) & 
                (diabetes_numeric == 1) & 
                (hypertension_numeric == 1)
            ]
            logic_violations += len(young_with_multiple_conditions)
            logic_checks += len(df)
            
            if len(young_with_multiple_conditions) > 0:
                results['issues'].append(f"{len(young_with_multiple_conditions)} casos de pacientes jóvenes (<30 años) con diabetes e hipertensión simultáneas")
        
        # Patient type vs comorbilidades (tipo 2 debería tener más comorbilidades)
        if 'patient_type' in df.columns:
            patient_type_numeric = self._safe_numeric_conversion(df['patient_type'])
            type_1_patients = df[patient_type_numeric == 1]
            type_2_patients = df[patient_type_numeric == 2]
            
            # Contar comorbilidades
            comorbidity_cols = ['diabetes', 'hypertension', 'cardiovascular', 'copd', 'asthma']
            available_comorbidities = [col for col in comorbidity_cols if col in df.columns]
            
            if available_comorbidities:
                type_1_avg_comorbidities = type_1_patients[available_comorbidities].sum(axis=1).mean() if len(type_1_patients) > 0 else 0
                type_2_avg_comorbidities = type_2_patients[available_comorbidities].sum(axis=1).mean() if len(type_2_patients) > 0 else 0
                
                # Tipo 2 debería tener más comorbilidades en promedio
                if type_1_avg_comorbidities > type_2_avg_comorbidities:
                    results['issues'].append("Patrón anómalo: pacientes tipo 1 tienen más comorbilidades promedio que tipo 2")
                    logic_violations += 1
                logic_checks += 1
        
        # 2. Validar correlaciones clínicas esperadas
        correlation_score = 0
        correlation_checks = 0
        
        for col1, col2 in self.clinical_correlations:
            if col1 in df.columns and col2 in df.columns:
                correlation = df[col1].corr(df[col2])
                if not np.isnan(correlation):
                    # Esperamos correlaciones positivas pero moderadas (0.1 - 0.6)
                    if correlation >= 0.1:
                        correlation_score += 1
                    elif correlation < -0.1:  # Solo reportar correlaciones fuertemente negativas
                        results['issues'].append(f"Correlación fuertemente negativa entre {col1} y {col2}: {correlation:.3f}")
                    elif correlation >= -0.1 and correlation < 0.1:
                        # Correlación débil, aceptable pero no ideal
                        correlation_score += 0.5
                    correlation_checks += 1
        
        # Calcular puntuaciones finales
        results['logic_violations'] = logic_violations
        results['total_checks'] = logic_checks
        results['coherence_score'] = max(0, (logic_checks - logic_violations) / max(1, logic_checks))
        results['correlation_score'] = correlation_score / max(1, correlation_checks) if correlation_checks > 0 else 0.5
        
        return results
    
    def validate_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Valida completitud de los datos"""
        results = {
            'completeness_score': 0.0,
            'essential_columns_present': False,
            'issues': []
        }
        
        # Columnas esenciales para datos médicos
        essential_columns = ['age', 'sex', 'edad/age', 'sexo/sex', 'gender']  # Acepta formato estándar, real o cardiología
        optional_important = ['diabetes', 'hypertension', 'patient_type', 'diag ing/inpat', 'resultado/val_result', 'diagnosis', 'systolic_bp', 'medication']
        
        # Verificar columnas esenciales (al menos una variante debe estar presente)
        present_essential = [col for col in essential_columns if col in df.columns]
        has_age = any(col in df.columns for col in ['age', 'edad/age', 'EDAD/AGE'])
        has_sex = any(col in df.columns for col in ['sex', 'sexo/sex', 'SEXO/SEX', 'gender'])
        results['essential_columns_present'] = has_age and has_sex
        
        if not results['essential_columns_present']:
            missing_info = []
            if not has_age:
                missing_info.append("edad/age")
            if not has_sex:
                missing_info.append("sexo/sex")
            results['issues'].append(f"Faltan datos esenciales: {missing_info}")
        
        # Verificar columnas importantes opcionales
        present_optional = [col for col in optional_important if col in df.columns]
        
        # Calcular puntuación de completitud (basado en presencia de información esencial)
        completeness_points = 0
        total_points = 4  # Edad, sexo, diagnóstico, datos clínicos
        
        if has_age:
            completeness_points += 1
        if has_sex:
            completeness_points += 1
        if any(col in df.columns for col in ['diag ing/inpat', 'DIAG ING/INPAT', 'diabetes', 'hypertension', 'diagnosis']):
            completeness_points += 1
        if any(col in df.columns for col in ['resultado/val_result', 'RESULTADO/VAL_RESULT', 'temp_ing/inpat', 'TEMP_ING/INPAT', 'sat_02_ing/inpat', 'SAT_02_ING/INPAT', 'systolic_bp', 'heart_rate']):
            completeness_points += 1
            
        results['completeness_score'] = completeness_points / total_points
        
        return results
