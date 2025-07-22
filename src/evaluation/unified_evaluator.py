"""
Evaluador unificado para datos sintéticos médicos.
Combina funcionalidades de ML, estadística y extracción de entidades médicas.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Importaciones opcionales para análisis médico
try:
    import medspacy
    from medspacy.ner import TargetRule
    MEDSPACY_AVAILABLE = True
except ImportError:
    MEDSPACY_AVAILABLE = False
    print("Warning: medspaCy no disponible. Funcionalidades de NLP médico deshabilitadas.")

try:
    from medspacy.umls_lookup import QuickUMLS
    QUICKUMLS_AVAILABLE = True
except ImportError:
    QUICKUMLS_AVAILABLE = False

class UnifiedMedicalEvaluator:
    """
    Evaluador unificado que combina múltiples métricas de calidad para datos sintéticos médicos.
    """
    
    def __init__(self):
        self.nlp = None
        self.medical_entities = self._get_medical_entities()
        self._init_medspacy()
    
    def _get_medical_entities(self) -> Dict[str, List[str]]:
        """Define entidades médicas conocidas para validación"""
        return {
            'medications': [
                'DEXAMETASONA', 'AZITROMICINA', 'ENOXAPARINA', 'FUROSEMIDA',
                'REMDESIVIR', 'TOCILIZUMAB', 'METILPREDNISOLONA', 'HEPARINA',
                'ATORVASTATINA', 'SIMVASTATINA', 'ENALAPRIL', 'LOSARTAN',
                'PARACETAMOL', 'IBUPROFENO', 'TRAMADOL', 'OMEPRAZOL'
            ],
            'diagnoses': [
                'COVID-19', 'PNEUMONIA', 'RESPIRATORY FAILURE', 'SEPSIS',
                'DIABETES', 'HYPERTENSION', 'HEART FAILURE', 'COPD',
                'ACUTE RESPIRATORY DISTRESS', 'MULTI-ORGAN FAILURE'
            ],
            'procedures': [
                'INTUBATION', 'MECHANICAL VENTILATION', 'DIALYSIS',
                'ECMO', 'SURGERY', 'BIOPSY', 'CATHETERIZATION'
            ]
        }
    
    def _init_medspacy(self):
        """Inicializa medspaCy si está disponible"""
        if not MEDSPACY_AVAILABLE:
            return
        
        try:
            # Cargar modelo básico
            self.nlp = medspacy.load("en_core_web_sm")
            
            # Añadir reglas personalizadas para entidades médicas
            self._add_custom_rules()
            
        except Exception as e:
            print(f"Warning: No se pudo inicializar medspaCy: {e}")
            self.nlp = None
    
    def _add_custom_rules(self):
        """Añade reglas personalizadas para extracción de entidades médicas"""
        if self.nlp is None:
            return
        
        # Añadir reglas para medicamentos
        medication_rules = [
            TargetRule(literal=med, category="MEDICATION")
            for med in self.medical_entities['medications']
        ]
        
        # Añadir reglas para diagnósticos
        diagnosis_rules = [
            TargetRule(literal=diag, category="DIAGNOSIS")
            for diag in self.medical_entities['diagnoses']
        ]
        
        # Añadir al pipeline si el componente existe
        if "medspacy_target_matcher" in self.nlp.pipe_names:
            for rule in medication_rules + diagnosis_rules:
                self.nlp.get_pipe("medspacy_target_matcher").add(rule)
    
    def evaluate_basic_metrics(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Calcula métricas básicas de comparación"""
        return {
            'original_rows': len(original),
            'synthetic_rows': len(synthetic),
            'original_cols': len(original.columns),
            'synthetic_cols': len(synthetic.columns),
            'size_ratio': len(synthetic) / len(original) if len(original) > 0 else 0,
            'column_match_ratio': len(set(original.columns) & set(synthetic.columns)) / len(original.columns) if len(original.columns) > 0 else 0
        }
    
    def evaluate_statistical_fidelity(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Evalúa fidelidad estadística usando múltiples métricas"""
        fidelity_scores = []
        distribution_matches = 0
        total_numeric_cols = 0
        column_details = {}
        
        for col in original.columns:
            if col in synthetic.columns:
                if original[col].dtype in ['int64', 'float64'] and synthetic[col].dtype in ['int64', 'float64']:
                    total_numeric_cols += 1
                    
                    orig_clean = original[col].dropna()
                    synth_clean = synthetic[col].dropna()
                    
                    if len(orig_clean) > 0 and len(synth_clean) > 0:
                        try:
                            # Test de Kolmogorov-Smirnov
                            ks_stat, ks_p = stats.ks_2samp(orig_clean, synth_clean)
                            ks_score = min(ks_p * 2, 1.0)
                            fidelity_scores.append(ks_score)
                            
                            # Comparación de estadísticas descriptivas
                            orig_mean = orig_clean.mean()
                            synth_mean = synth_clean.mean()
                            orig_std = orig_clean.std()
                            synth_std = synth_clean.std()
                            
                            mean_similarity = 1 - abs(orig_mean - synth_mean) / (orig_mean + 1e-8)
                            std_similarity = 1 - abs(orig_std - synth_std) / (orig_std + 1e-8)
                            
                            if ks_p > 0.05:
                                distribution_matches += 1
                            
                            column_details[col] = {
                                'ks_statistic': ks_stat,
                                'ks_p_value': ks_p,
                                'mean_similarity': max(0, mean_similarity),
                                'std_similarity': max(0, std_similarity),
                                'distribution_match': ks_p > 0.05
                            }
                            
                        except Exception as e:
                            column_details[col] = {'error': str(e)}
        
        avg_fidelity = np.mean(fidelity_scores) if fidelity_scores else 0.0
        distribution_match_rate = distribution_matches / total_numeric_cols if total_numeric_cols > 0 else 0.0
        
        return {
            'average_fidelity_score': avg_fidelity,
            'distribution_match_rate': distribution_match_rate,
            'total_numeric_columns': total_numeric_cols,
            'columns_analyzed': len(fidelity_scores),
            'column_details': column_details
        }
    
    def evaluate_ml_performance(self, original: pd.DataFrame, synthetic: pd.DataFrame, 
                              target_col: str = None) -> Dict[str, Any]:
        """Evalúa performance de ML comparando modelos entrenados con datos originales vs sintéticos"""
        
        # Auto-detectar columna target si no se especifica
        if target_col is None:
            possible_targets = ['RESULTADO/VAL_RESULT', 'OUTCOME', 'TARGET', 'LABEL']
            for col in possible_targets:
                if col in original.columns and col in synthetic.columns:
                    target_col = col
                    break
        
        if target_col is None or target_col not in original.columns:
            return {
                'ml_evaluation_error': 'No se encontró columna target válida',
                'f1_preservation': 0.5,
                'accuracy_preservation': 0.5
            }
        
        try:
            # Preparar datos originales
            X_orig = original.drop(columns=[target_col]).select_dtypes(include=[np.number])
            y_orig = original[target_col]
            
            # Preparar datos sintéticos
            X_synth = synthetic.drop(columns=[target_col]).select_dtypes(include=[np.number])
            y_synth = synthetic[target_col]
            
            # Asegurar que las columnas coincidan
            common_cols = list(set(X_orig.columns) & set(X_synth.columns))
            X_orig = X_orig[common_cols]
            X_synth = X_synth[common_cols]
            
            if len(common_cols) == 0:
                return {
                    'ml_evaluation_error': 'No hay columnas numéricas comunes',
                    'f1_preservation': 0.5,
                    'accuracy_preservation': 0.5
                }
            
            # Entrenar modelo con datos originales
            X_train, X_test, y_train, y_test = train_test_split(
                X_orig, y_orig, test_size=0.2, random_state=42
            )
            
            model_orig = RandomForestClassifier(random_state=42, n_estimators=50)
            model_orig.fit(X_train, y_train)
            y_pred_orig = model_orig.predict(X_test)
            
            # Entrenar modelo con datos sintéticos
            model_synth = RandomForestClassifier(random_state=42, n_estimators=50)
            model_synth.fit(X_synth, y_synth)
            y_pred_synth = model_synth.predict(X_test)
            
            # Calcular métricas
            f1_orig = f1_score(y_test, y_pred_orig, average='weighted')
            acc_orig = accuracy_score(y_test, y_pred_orig)
            f1_synth = f1_score(y_test, y_pred_synth, average='weighted')
            acc_synth = accuracy_score(y_test, y_pred_synth)
            
            return {
                'original_f1': f1_orig,
                'original_accuracy': acc_orig,
                'synthetic_f1': f1_synth,
                'synthetic_accuracy': acc_synth,
                'f1_preservation': f1_synth / f1_orig if f1_orig > 0 else 0,
                'accuracy_preservation': acc_synth / acc_orig if acc_orig > 0 else 0,
                'common_features': len(common_cols),
                'target_column': target_col
            }
            
        except Exception as e:
            return {
                'ml_evaluation_error': str(e),
                'f1_preservation': 0.5,
                'accuracy_preservation': 0.5
            }
    
    def evaluate_medical_entities(self, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Evalúa la extracción y calidad de entidades médicas"""
        
        text_columns = [
            "DIAG ING/INPAT",
            "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME",
            "MOTIVO_ALTA/DESTINY_DISCHARGE_ING"
        ]
        
        results = {
            'entities_found': {},
            'total_entities': 0,
            'medical_coherence_score': 0.0,
            'entity_distribution': {}
        }
        
        if not MEDSPACY_AVAILABLE or self.nlp is None:
            # Evaluación básica sin medspaCy
            return self._basic_medical_evaluation(synthetic, text_columns)
        
        try:
            all_entities = []
            
            for col in text_columns:
                if col in synthetic.columns:
                    col_entities = []
                    for text in synthetic[col].dropna():
                        if isinstance(text, str) and text.strip():
                            doc = self.nlp(text)
                            entities = [(ent.text, ent.label_) for ent in doc.ents]
                            col_entities.extend(entities)
                            all_entities.extend(entities)
                    
                    results['entities_found'][col] = len(col_entities)
            
            results['total_entities'] = len(all_entities)
            
            # Calcular score de coherencia médica
            if all_entities:
                medical_entities = [ent for ent in all_entities if ent[1] in ['MEDICATION', 'DIAGNOSIS', 'PROCEDURE']]
                results['medical_coherence_score'] = len(medical_entities) / len(all_entities)
            
            # Distribución por tipo de entidad
            entity_types = {}
            for _, ent_type in all_entities:
                entity_types[ent_type] = entity_types.get(ent_type, 0) + 1
            results['entity_distribution'] = entity_types
            
        except Exception as e:
            results['medical_evaluation_error'] = str(e)
        
        return results
    
    def _basic_medical_evaluation(self, synthetic: pd.DataFrame, text_columns: List[str]) -> Dict[str, Any]:
        """Evaluación médica básica sin medspaCy"""
        
        results = {
            'entities_found': {},
            'total_entities': 0,
            'medical_coherence_score': 0.5,
            'known_medications_found': 0,
            'known_diagnoses_found': 0
        }
        
        all_text = ""
        for col in text_columns:
            if col in synthetic.columns:
                texts = synthetic[col].dropna().astype(str)
                all_text += " ".join(texts).upper()
        
        # Contar medicamentos conocidos
        med_count = 0
        for med in self.medical_entities['medications']:
            if med.upper() in all_text:
                med_count += 1
        
        # Contar diagnósticos conocidos
        diag_count = 0
        for diag in self.medical_entities['diagnoses']:
            if diag.upper() in all_text:
                diag_count += 1
        
        results['known_medications_found'] = med_count
        results['known_diagnoses_found'] = diag_count
        results['total_entities'] = med_count + diag_count
        
        return results
    
    def comprehensive_evaluation(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                                validation_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Realiza una evaluación completa combinando todas las métricas"""
        
        print("🔍 Ejecutando evaluación completa...")
        
        # 1. Métricas básicas
        basic_metrics = self.evaluate_basic_metrics(original, synthetic)
        
        # 2. Fidelidad estadística
        statistical_metrics = self.evaluate_statistical_fidelity(original, synthetic)
        
        # 3. Performance ML
        ml_metrics = self.evaluate_ml_performance(original, synthetic)
        
        # 4. Entidades médicas
        medical_metrics = self.evaluate_medical_entities(synthetic)
        
        # 5. Combinar con resultados de validación si están disponibles
        validation_score = 0.0
        if validation_results:
            validation_score = validation_results.get('success_rate', 0) / 100.0
        
        # 6. Calcular score final ponderado
        weights = {
            'statistical': 0.3,
            'ml': 0.3,
            'medical': 0.2,
            'validation': 0.2
        }
        
        final_score = (
            statistical_metrics.get('average_fidelity_score', 0) * weights['statistical'] +
            ml_metrics.get('f1_preservation', 0) * weights['ml'] +
            medical_metrics.get('medical_coherence_score', 0) * weights['medical'] +
            validation_score * weights['validation']
        )
        
        # 7. Generar recomendación
        if final_score >= 0.8:
            recommendation = "Excelente - Datos listos para uso en producción"
        elif final_score >= 0.6:
            recommendation = "Bueno - Datos útiles con algunas limitaciones"
        elif final_score >= 0.4:
            recommendation = "Aceptable - Requiere mejoras antes del uso"
        else:
            recommendation = "Insuficiente - Regenerar con diferentes parámetros"
        
        return {
            'final_quality_score': final_score,
            'usage_recommendation': recommendation,
            'basic_metrics': basic_metrics,
            'statistical_fidelity': statistical_metrics,
            'ml_performance': ml_metrics,
            'medical_entity_analysis': medical_metrics,
            'validation_integration': validation_score,
            'weights_used': weights
        }

# Función de conveniencia para uso directo
def evaluate_synthetic_data(original: pd.DataFrame, synthetic: pd.DataFrame,
                          validation_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para evaluación completa de datos sintéticos.
    
    Args:
        original: DataFrame con datos originales
        synthetic: DataFrame con datos sintéticos
        validation_results: Resultados de validación médica (opcional)
        
    Returns:
        Dict con resultados completos de evaluación
    """
    evaluator = UnifiedMedicalEvaluator()
    return evaluator.comprehensive_evaluation(original, synthetic, validation_results)

# Funciones legacy para compatibilidad hacia atrás
def evaluate_predictions(y_true, y_pred):
    """Función legacy para compatibilidad"""
    return {
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'accuracy': accuracy_score(y_true, y_pred)
    }

def evaluate_ml_performance(original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
    """Función legacy para compatibilidad"""
    evaluator = UnifiedMedicalEvaluator()
    return evaluator.evaluate_ml_performance(original, synthetic)

def evaluate_medical_entities(synthetic: pd.DataFrame) -> Dict[str, Any]:
    """Función legacy para compatibilidad"""
    evaluator = UnifiedMedicalEvaluator()
    return evaluator.evaluate_medical_entities(synthetic)
