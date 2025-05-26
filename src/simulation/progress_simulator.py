import numpy as np

def simulate_disease_progression(lab_value, improvement=True):
    '''
    La función simulate_disease_progression simula la progresión de una variable
    clínica con ruido controlado.
    params:
    - lab_value: valor de laboratorio inicial del paciente
    - improvement: booleano que indica si la enfermedad está mejorando o empeorando
    return:
    - máximo valor entre 0 y el nuevo valor de laboratorio simulado
    '''
    delta = np.random.normal(-5 if improvement else 5, 2)
    return max(0, lab_value + delta)