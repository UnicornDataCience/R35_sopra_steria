import numpy as np
import pandas as pd
import os

script_dir = os.path.dirname(__file__)
CSV_REAL_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'real', 'df_final_v2.csv'))
CSV_SYN_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic', 'datos_sinteticos_sdv.csv'))
CSV_SYN_PATH_2 = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic', 'datos_sinteticos_tvae.csv'))
df_real = pd.read_csv(CSV_REAL_PATH, sep =',', low_memory=False,encoding="utf-8")
df_synthetic = pd.read_csv(CSV_SYN_PATH, sep =',', low_memory=False,encoding="utf-8")
df_synthetic_2 = pd.read_csv(CSV_SYN_PATH_2, sep =',', low_memory=False,encoding="utf-8")
columnas = ['PATIENT ID', 'EDAD/AGE', 'SEXO/SEX', 'DIAG ING/INPAT',
                'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME', 'UCI_DIAS/ICU_DAYS',
                'TEMP_ING/INPAT', 'SAT_02_ING/INPAT', 'RESULTADO/VAL_RESULT',
                'MOTIVO_ALTA/DESTINY_DISCHARGE_ING']
df_real = df_real[columnas]
df_synthetic.drop_duplicates(subset=['PATIENT ID'], keep='first', inplace=True)
df_synthetic.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
df_synthetic_2.drop_duplicates(subset=['PATIENT ID'], keep='first', inplace=True)
df_synthetic_2.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
df = pd.concat([df_real, df_synthetic, df_synthetic_2], ignore_index=True)

import numpy as np

def simulate_disease_progression(lab_value, improvement=True, param_type='general'):
    '''
    La función simulate_disease_progression simula la progresión de una variable
    clínica con ruido controlado y límites fisiológicos.
    
    params:
    - lab_value: valor de laboratorio inicial del paciente
    - improvement: booleano que indica si la enfermedad está mejorando o empeorando
    - param_type: tipo de parámetro ('temp', 'sat', 'pcr', 'general')
    
    return:
    - nuevo valor de laboratorio simulado con límites aplicados
    '''
    delta = np.random.normal(-5 if improvement else 5, 2)
    new_value = lab_value + delta
    
    # Aplicar límites específicos según el tipo de parámetro
    if param_type == 'temp':
        if improvement:
            # Si hay mejora, temperatura no puede ser inferior a 36.5°C
            new_value = max(36.5, new_value)
        else:
            # Si no hay mejora, temperatura no puede superar los 45°C
            new_value = min(45.0, new_value)
    
    elif param_type == 'sat':
        if improvement:
            # Si hay mejora, saturación no puede bajar de 90%
            new_value = max(90.0, new_value)
            # Límite superior fisiológico de saturación
            new_value = min(100.0, new_value)
        else:
            # Si no hay mejora, saturación puede bajar pero no menos de 70%
            new_value = max(70.0, new_value)
            new_value = min(100.0, new_value)
    
    elif param_type == 'pcr':
        # PCR no puede ser negativa
        new_value = max(0, new_value)
    
    else:
        # Para otros parámetros, solo aplicar límite inferior de 0
        new_value = max(0, new_value)
    
    return new_value

def simulate_progression_for_patient(patient_id):
    '''
    La función simulate_progression_for_patient simula la progresión de la enfermedad
    para un paciente específico aplicando reglas clínicas específicas.
    
    Reglas:
    - PCR < 10.0: mejora (improvement = True)
    - PCR >= 10.0: no mejora (improvement = False)
    - Temperatura: mejora >=36.5°C, no mejora <=45°C
    - Saturación: mejora >=90%, no mejora >=70%
    
    params:
    - patient_id: ID del paciente
    
    return:
    - None, pero imprime el resultado de la simulación
    '''
    patient_data = df_real[df_real['PATIENT ID'] == patient_id]
    if patient_data.empty:
        print(f"No se encontraron datos para el paciente con ID {patient_id}.")
        return
    
    # Obtener valores iniciales
    temp = patient_data['TEMP_ING/INPAT'].values[0]
    sat = patient_data['SAT_02_ING/INPAT'].values[0]
    pcr = patient_data['RESULTADO/VAL_RESULT'].values[0]
    
    # REGLA CORREGIDA: PCR < 10.0 = mejora, PCR >= 10.0 = no mejora
    improvement = pcr < 10.0
    
    # Simular progresión con reglas específicas
    new_temp = simulate_disease_progression(temp, improvement, 'temp')
    new_sat = simulate_disease_progression(sat, improvement, 'sat')
    new_pcr = simulate_disease_progression(pcr, improvement, 'pcr')
    
    # Mostrar resultados
    status = "MEJORANDO" if improvement else "SIN MEJORA/EMPEORANDO"
    print(f"Paciente {patient_id} - Estado: {status}")
    print(f"  PCR inicial: {pcr:.2f} mg/L")
    print(f"  Temperatura: {temp:.2f} → {new_temp:.2f} °C")
    print(f"  Saturación O2: {sat:.2f} → {new_sat:.2f} %")
    print(f"  PCR: {pcr:.2f} → {new_pcr:.2f} mg/L")

# Ejecutar simulación
simulate_progression_for_patient(30)

    
