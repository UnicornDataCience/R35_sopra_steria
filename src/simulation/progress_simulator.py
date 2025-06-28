import numpy as np
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

script_dir = os.path.dirname(__file__)
CSV_REAL_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'real', 'df_final_v2.csv'))
CSV_SDV_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic', 'datos_sinteticos_sdv.csv'))
CSV_TVAE_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic', 'datos_sinteticos_tvae.csv'))
CSV_CTGAN_PATH = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'synthetic', 'datos_sinteticos_ctgan.csv'))

columnas = [
    'PATIENT ID', 'EDAD/AGE', 'SEXO/SEX', 'DIAG ING/INPAT',
    'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME', 'UCI_DIAS/ICU_DAYS',
    'TEMP_ING/INPAT', 'SAT_02_ING/INPAT', 'RESULTADO/VAL_RESULT',
    'MOTIVO_ALTA/DESTINY_DISCHARGE_ING'
]

def load_and_clean(path):
    """Carga y limpia un archivo CSV si existe"""
    if not os.path.exists(path):
        print(f"Warning: Archivo {path} no encontrado, saltando...")
        return pd.DataFrame()  # DataFrame vacío
    
    try:
        df = pd.read_csv(path, sep=',', low_memory=False, encoding="utf-8")
        if df.empty:
            return df
        
        # Verificar que las columnas necesarias existen
        missing_cols = [col for col in columnas if col not in df.columns]
        if missing_cols:
            print(f"Warning: Columnas faltantes en {path}: {missing_cols}")
            return pd.DataFrame()
        
        df = df[columnas]
        df.drop_duplicates(subset=['PATIENT ID'], keep='first', inplace=True)
        df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        
        # Conversión de tipos y nulos
        df['EDAD/AGE'] = pd.to_numeric(df['EDAD/AGE'], errors='coerce').fillna(0).astype(int)
        df['UCI_DIAS/ICU_DAYS'] = pd.to_numeric(df['UCI_DIAS/ICU_DAYS'], errors='coerce').fillna(0).astype(int)
        df['TEMP_ING/INPAT'] = pd.to_numeric(df['TEMP_ING/INPAT'], errors='coerce').fillna(36.5)
        df['SAT_02_ING/INPAT'] = pd.to_numeric(df['SAT_02_ING/INPAT'], errors='coerce').fillna(98.0)
        df['RESULTADO/VAL_RESULT'] = pd.to_numeric(df['RESULTADO/VAL_RESULT'], errors='coerce').fillna(0)
        df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'] = df['MOTIVO_ALTA/DESTINY_DISCHARGE_ING'].fillna('Domicilio')
        
        return df
    except Exception as e:
        print(f"Error cargando {path}: {e}")
        return pd.DataFrame()

def load_available_data():
    """Carga solo los datos disponibles"""
    datasets = []
    
    # Intentar cargar cada archivo
    for name, path in [
        ("real", CSV_REAL_PATH),
        ("sdv", CSV_SDV_PATH),
        ("tvae", CSV_TVAE_PATH),
        ("ctgan", CSV_CTGAN_PATH)
    ]:
        df = load_and_clean(path)
        if not df.empty:
            print(f"✅ Cargado {name}: {len(df)} registros")
            datasets.append(df)
        else:
            print(f"⚠️ No disponible {name}: {path}")
    
    if datasets:
        combined_df = pd.concat(datasets, ignore_index=True)
        print(f"📊 Total combinado: {len(combined_df)} registros")
        return combined_df
    else:
        print("❌ No se encontraron datos válidos")
        return pd.DataFrame()

# NUEVO: No cargar automáticamente al importar
# df_real = load_and_clean(CSV_REAL_PATH)
# df_sdv = load_and_clean(CSV_SDV_PATH)
# df_tvae = load_and_clean(CSV_TVAE_PATH)
# df_ctgan = load_and_clean(CSV_CTGAN_PATH)
# df = pd.concat([df_real, df_sdv, df_tvae, df_ctgan], ignore_index=True)

class ProgressSimulator:
    """Simulador de progresión de pacientes - Motor compartido"""
    
    def __init__(self, data_source: pd.DataFrame = None):
        """Inicializa con datos o carga los disponibles"""
        if data_source is not None:
            self.df = data_source
            print(f"🎯 Simulador inicializado con {len(data_source)} registros externos")
        else:
            # NUEVO: Carga lazy - solo cuando se necesita
            self.df = load_available_data()
            if self.df.empty:
                # Datos mínimos de fallback
                print("⚠️ Usando datos mínimos de fallback")
                self.df = self._create_fallback_data()
    
    def _create_fallback_data(self) -> pd.DataFrame:
        """Crea datos mínimos de ejemplo si no hay archivos disponibles"""
        return pd.DataFrame({
            'PATIENT ID': range(1, 11),
            'EDAD/AGE': np.random.randint(20, 90, 10),
            'SEXO/SEX': np.random.choice(['MALE', 'FEMALE'], 10),
            'DIAG ING/INPAT': ['COVID19 - POSITIVO'] * 10,
            'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME': ['PARACETAMOL'] * 10,
            'UCI_DIAS/ICU_DAYS': np.random.randint(0, 20, 10),
            'TEMP_ING/INPAT': np.random.uniform(36.5, 39.5, 10),
            'SAT_02_ING/INPAT': np.random.uniform(88, 100, 10),
            'RESULTADO/VAL_RESULT': np.random.uniform(0, 20, 10),
            'MOTIVO_ALTA/DESTINY_DISCHARGE_ING': ['Domicilio'] * 10
        })
    
    def simulate_disease_progression(self, lab_value: float, improvement: bool = True, param_type: str = 'general') -> float:
        """Simula progresión de una variable clínica"""
        delta = np.random.normal(-5 if improvement else 5, 2)
        new_value = lab_value + delta

        if param_type == 'temp':
            new_value = max(36.5, min(45.0, new_value)) if improvement else min(45.0, new_value)
        elif param_type == 'sat':
            if improvement:
                new_value = max(95.0, min(100.0, new_value))
            else:
                new_value = max(94.0, min(100.0, new_value))
        elif param_type == 'pcr':
            new_value = max(0, new_value)
        else:
            new_value = max(0, new_value)
        return new_value
    
    def simulate_single_patient_visit(self, patient_id: int, visit_day: int = 1) -> Dict[str, Any]:
        """Simula una visita individual para un paciente"""
        patient_data = self.df[self.df['PATIENT ID'] == patient_id]
        if patient_data.empty:
            return {"error": f"No se encontraron datos para el paciente con ID {patient_id}."}

        sat = patient_data['SAT_02_ING/INPAT'].values[0]
        pcr = patient_data['RESULTADO/VAL_RESULT'].values[0]
        
        # Aplicar evolución basada en el día
        new_pcr = self.simulate_disease_progression(pcr + (visit_day * 0.5), param_type='pcr')
        improvement = new_pcr < 10.0
        new_sat = self.simulate_disease_progression(sat, improvement, 'sat')
        
        date = datetime(2024, np.random.randint(1, 13), np.random.randint(1, 29)).strftime('%Y-%m-%d')
        
        visit = {
            'date': date,
            'day_hospitalization': visit_day,
            'labs': {
                "PCR": float(round(new_pcr, 2)),
                "SAT_O2": float(round(new_sat, 2))
            }
        }

        # Añadir síntomas y acciones según el estado
        visit.update(self._assign_clinical_actions(new_sat, new_pcr, improvement))
        
        return visit
    
    def simulate_patient_timeline(self, patient_id: int, num_visits: int) -> List[Dict[str, Any]]:
        """Simula timeline completo para un paciente"""
        timeline = []
        
        for visit in range(num_visits):
            visit_data = self.simulate_single_patient_visit(patient_id, visit + 1)
            if 'error' not in visit_data:
                timeline.append(visit_data)
        
        return timeline
    
    def simulate_batch_evolution(self, patient_data: pd.DataFrame, visits_per_patient: Dict[int, int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Simula evolución para un lote de pacientes"""
        evolved_records = []
        stats = {
            'total_patients': len(patient_data),
            'total_visits': 0,
            'avg_visits_per_patient': 0,
            'patients_with_improvement': 0,
            'patients_with_deterioration': 0
        }
        
        for idx, patient in patient_data.iterrows():
            patient_id = patient.get('PATIENT ID', idx)
            
            # Determinar número de visitas
            if visits_per_patient and patient_id in visits_per_patient:
                num_visits = visits_per_patient[patient_id]
            else:
                # Lógica por defecto basada en severidad
                uci_days = patient.get('UCI_DIAS/ICU_DAYS', 0)
                num_visits = np.random.randint(5, 15) if uci_days > 0 else np.random.randint(2, 8)
            
            # Generar timeline para este paciente
            patient_timeline = self.simulate_patient_timeline(patient_id, num_visits)
            
            # Convertir timeline a registros de DataFrame
            for visit in patient_timeline:
                visit_record = patient.copy().to_dict()
                visit_record.update({
                    'PATIENT ID': f"{patient_id}_V{visit['day_hospitalization']}",
                    'TEMP_ING/INPAT': visit['labs'].get('TEMP', patient.get('TEMP_ING/INPAT', 38.5)),
                    'SAT_02_ING/INPAT': visit['labs']['SAT_O2'],
                    'RESULTADO/VAL_RESULT': visit['labs']['PCR'],
                    'FECHA_VISITA': visit['date'],
                    'DIA_HOSPITALIZACION': visit['day_hospitalization'],
                    'SINTOMAS': visit.get('syntoms', []),
                    'ACCIONES': visit.get('acciones', [])
                })
                evolved_records.append(visit_record)
            
            stats['total_visits'] += len(patient_timeline)
        
        stats['avg_visits_per_patient'] = stats['total_visits'] / stats['total_patients'] if stats['total_patients'] > 0 else 0
        
        return pd.DataFrame(evolved_records), stats
    
    def _assign_clinical_actions(self, new_sat: float, new_pcr: float, improvement: bool) -> Dict[str, Any]:
        """Asigna síntomas y acciones clínicas"""
        result = {}
        
        if improvement and new_sat >= 94.0:
            result['syntoms'] = ["Estabilizado"]
        elif new_sat >= 94.0:
            result['syntoms'] = ["Fiebre, tos sin disnea, cefalea, mialgias, náuseas, vómitos, diarrea"]
            result['acciones'] = ["Descanso en cama", "Hidratación oral", "Paracetamol 500mg cada 8 horas", "Aislamiento", "Uso de mascarilla"]
        elif new_sat >= 90.0:
            result['syntoms'] = ["Agotamiento, astenia, tos, disnea y signos de afectación pulmonar"]
            result['acciones'] = ["Administración de oxígeno suplementario", "Ventilación no invasiva", "Remdesivir", "Dexametasona", "Inmunomoduladores"]
        elif new_sat < 90.0 and new_pcr >= 15.0:
            result['syntoms'] = ["Estado grave con insuficiencia respiratoria, hipoxemia, alteración de funciones vitales y manifestaciones extrapulmonares"]
            result['acciones'] = ["Ventilación mecánica invasiva", "OMEC", "Manejo en unidad de cuidados intensivos en cubículos con presión negativa"]
        else:
            result['syntoms'] = ["Agotamiento, astenia, tos, disnea y signos de afectación pulmonar"]
            result['acciones'] = ["Administración de oxígeno suplementario", "Ventilación no invasiva", "Remdesivir", "Dexametasona", "Inmunomoduladores"]
        
        return result

# Función de compatibilidad con el script original
def simulate_progression_for_patient(patient_id: int) -> Dict[str, Any]:
    """Función de compatibilidad - usar la clase ProgressSimulator"""
    simulator = ProgressSimulator()
    return simulator.simulate_single_patient_visit(patient_id)

# Ejecución standalone si se ejecuta directamente
if __name__ == "__main__":
    print(simulate_progression_for_patient(5))


