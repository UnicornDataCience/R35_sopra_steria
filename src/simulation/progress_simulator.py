import numpy as np
import pandas as pd
import os
from datetime import datetime

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
            # Si hay mejora, saturación no puede bajar de 95%
            new_value = max(95.0, new_value)
            # Límite superior fisiológico de saturación
            new_value = min(100.0, new_value)
        else:
            # Si no hay mejora, saturación puede bajar pero no menos de 89%
            new_value = max(94.0, new_value)
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
    - Saturación: diversos umbrales determinan la gravedad
    
    params:
    - patient_id: ID del paciente
    
    return:
    - Objeto JSON con el resultado de la simulación
    '''
    patient_data = df_real[df_real['PATIENT ID'] == patient_id]
    if patient_data.empty:
        return {"error": f"No se encontraron datos para el paciente con ID {patient_id}."}
    
    # Obtener valores iniciales
    sat = patient_data['SAT_02_ING/INPAT'].values[0]
    pcr = patient_data['RESULTADO/VAL_RESULT'].values[0]
    # generar una fecha aleatoria comprendida entre Enero y Diciembre de 2024
    date = datetime(2024, np.random.randint(1, 13), np.random.randint(1, 29)).strftime('%Y-%m-%d')
    new_pcr = simulate_disease_progression(pcr,'pcr')
    improvement = new_pcr < 10.0
    new_sat = simulate_disease_progression(sat, improvement, 'sat')

    # Crear la estructura base del visit
    visit = {
        'date': date,
        'labs': {
            "PCR": float(round(new_pcr, 2)),
            "SAT_O2": float(round(new_sat, 2))
        }
    }

    if improvement and new_sat >= 94.0:
        visit['syntoms'] = ["Estabilizado"]
    elif new_sat >= 94.0:
        visit['syntoms'] = ["Fiebre, tos sin disnea, cefalea, mialgias, náuseas, vómitos, diarrea"]
        visit['acciones'] = ["Descanso en cama", "Hidratación oral", "Paracetamol 500mg cada 8 horas", "Aislamiento", "Uso de mascarilla"]
    elif new_sat >= 90.0:
        new_sat = simulate_disease_progression(sat, improvement, 'sat')
        visit['syntoms'] = ["Agotamiento, astenia, tos, disnea y signos de afectación pulmonar"]
        visit['acciones'] = ["Administración de oxígeno suplementario", "Ventilación no invasiva", "Remdesivir", "Dexametasona", "Inmunomoduladores"]
    elif new_sat < 90.0 and new_pcr >= 15.0:
        visit['syntoms'] = ["Estado grave con insuficiencia respiratoria, hipoxemia, alteración de funciones vitales y manifestaciones extrapulmonares"]
        visit['acciones'] = ["Ventilación mecánica invasiva", "OMEC", "Manejo en unidad de cuidados intensivos en cubículos con presión negativa"]
    else:  # new_sat < 90.0
        visit['syntoms'] = ["Agotamiento, astenia, tos, disnea y signos de afectación pulmonar"]
        visit['acciones'] = ["Administración de oxígeno suplementario", "Ventilación no invasiva", "Remdesivir", "Dexametasona", "Inmunomoduladores"]
    return print([visit])

# Ejecutar simulación
simulate_progression_for_patient(5)

    
