from sklearn.metrics import f1_score, accuracy_score

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
        'f1_score': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred)
    }