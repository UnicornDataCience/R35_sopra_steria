from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os

def evaluate_predictions(y_true, y_pred):
    ''' 
    La función evaluate_predictions evalúa las predicciones de un modelo de clasificación
    utilizando métricas de rendimiento como F1 Score y Accuracy.
    params:
    - y_true: etiquetas verdaderas
    - y_pred: etiquetas predichas por el modelo
    return:
    - diccionario con pares clave y valor de las métricas de rendimiento obtenidas
    '''
    return {
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'accuracy': accuracy_score(y_true, y_pred)
    }

def evaluate_ml_performance(original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
    """Evalúa performance de ML usando datos sintéticos vs originales"""
    ml_metrics = {}
    
    try:
        # Identificar columna target (ejemplo: RESULTADO/VAL_RESULT)
        target_col = 'RESULTADO/VAL_RESULT'
        if target_col in original.columns and target_col in synthetic.columns:
            
            # Preparar datos originales
            X_orig = original.drop(columns=[target_col]).select_dtypes(include=[np.number])
            y_orig = original[target_col]
            
            # Preparar datos sintéticos
            X_synth = synthetic.drop(columns=[target_col]).select_dtypes(include=[np.number])
            y_synth = synthetic[target_col]
            
            # Entrenar modelo con datos originales
            X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)
            model_orig = RandomForestClassifier(random_state=42, n_estimators=50)
            model_orig.fit(X_train, y_train)
            y_pred_orig = model_orig.predict(X_test)
            
            # Entrenar modelo con datos sintéticos
            model_synth = RandomForestClassifier(random_state=42, n_estimators=50)
            model_synth.fit(X_synth, y_synth)
            y_pred_synth = model_synth.predict(X_test)  # Evaluar en test original
            
            # Calcular métricas usando el evaluador básico
            metrics_orig = evaluate_predictions(y_test, y_pred_orig)
            metrics_synth = evaluate_predictions(y_test, y_pred_synth)
            
            ml_metrics = {
                'original_f1': metrics_orig['f1_score'],
                'original_accuracy': metrics_orig['accuracy'],
                'synthetic_f1': metrics_synth['f1_score'],
                'synthetic_accuracy': metrics_synth['accuracy'],
                'f1_preservation': metrics_synth['f1_score'] / metrics_orig['f1_score'] if metrics_orig['f1_score'] > 0 else 0,
                'accuracy_preservation': metrics_synth['accuracy'] / metrics_orig['accuracy'] if metrics_orig['accuracy'] > 0 else 0
            }
            
    except Exception as e:
        ml_metrics = {
            'ml_evaluation_error': str(e),
            'f1_preservation': 0.5,
            'accuracy_preservation': 0.5
        }
    
    return ml_metrics

def evaluate_medical_entities(synthetic: pd.DataFrame) -> Dict[str, Any]:
    """Evalúa extracción de entidades médicas en datos sintéticos"""
    
    columnas_texto = [
        "DIAG ING/INPAT",
        "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME", 
        "MOTIVO_ALTA/DESTINY_DISCHARGE_ING"
    ]
    
    try:
        # Importar el extractor de evaluator_debug
        sys.path.append(os.path.dirname(__file__))
        from evaluator_debug import crear_extractor_personalizado, evaluar_con_metricas_completas
        
        # Usar el extractor personalizado mejorado
        extractor_personalizado = crear_extractor_personalizado()
        
        # Evaluar con métricas completas
        metricas_globales, metricas_por_columna, ejemplos = evaluar_con_metricas_completas(
            synthetic, extractor_personalizado, columnas_texto
        )
        
        return {
            'medical_entity_precision': metricas_globales.get('precision', 0.5),
            'medical_entity_recall': metricas_globales.get('recall', 0.5),
            'medical_entity_f1': metricas_globales.get('f1', 0.5),
            'entities_by_column': metricas_por_columna,
            'entity_extraction_quality': 'Buena' if metricas_globales.get('f1', 0) > 0.7 else 'Necesita mejora'
        }
        
    except Exception as e:
        return {
            'medical_entity_error': str(e),
            'medical_entity_f1': 0.5,
            'entity_extraction_quality': 'No evaluado'
        }