import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from src.simulation.transition_model import LearnedTransitionModel
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ProgressSimulator:
    """Simulador de progresión de pacientes - Motor compartido con modelo aprendido"""
    
    def __init__(self, data_source: pd.DataFrame, disease_type: str = "general", use_learned_model: bool = True):
        """
        Inicializa con datos o crea datos de fallback si no se proporcionan
        
        Args:
            data_source: DataFrame con datos de pacientes
            disease_type: Tipo de enfermedad ('covid19', 'general', etc.)
            use_learned_model: Si True, usa modelo de transición aprendido
        """
        if data_source is not None and not data_source.empty:
            self.df = data_source
            logger.info(f"🎯 Simulador inicializado con {len(data_source)} registros externos")
        else:
            logger.warning("⚠️ No se proporcionaron datos válidos. Usando datos mínimos de fallback.")
            self.df = self._create_fallback_data()
        
        self.disease_type = disease_type
        self.use_learned_model = use_learned_model
        
        # Inicializar modelo de transición aprendido
        self.transition_model: Optional[LearnedTransitionModel] = None
        if use_learned_model and not self.df.empty:
            try:
                self.transition_model = LearnedTransitionModel()
                self.transition_model.fit(self.df, disease_type)
                logger.info(f"✅ Modelo de transición aprendido inicializado para {disease_type}")
            except Exception as e:
                logger.warning(f"⚠️ No se pudo inicializar modelo aprendido: {e}, usando simulación simple")
                self.use_learned_model = False
        
        logger.info(f"Simulador configurado para tipo de enfermedad: {self.disease_type}, modelo aprendido: {self.use_learned_model}")

    def _create_fallback_data(self) -> pd.DataFrame:
        """Crea datos mínimos de ejemplo si no hay archivos disponibles"""
        return pd.DataFrame({
            'patient_id': range(1, 11),
            'age': np.random.randint(20, 90, 10),
            'gender': np.random.choice(['MALE', 'FEMALE'], 10),
            'diagnosis': ['COVID-19 - POSITIVO'] * 10,
            'medication': ['PARACETAMOL'] * 10,
            'icu_days': np.random.randint(0, 20, 10),
            'temperature': np.random.uniform(36.5, 39.5, 10),
            'oxygen_saturation': np.random.uniform(88, 100, 10),
            'pcr_result': np.random.uniform(0, 20, 10),
            'discharge_motive': ['Domicilio'] * 10
        })
    
    def simulate_disease_progression(
        self,
        current_value: float,
        improvement: bool = True,
        param_type: str = 'general',
        visit_number: int = 1,
        severity: str = "moderate"
    ) -> Tuple[float, bool]:
        """
        Simula progresión de una variable clínica
        
        Args:
            current_value: Valor actual
            improvement: Hint de si debería mejorar (usado solo si no hay modelo aprendido)
            param_type: Tipo de parámetro
            visit_number: Número de visita para modelo temporal
            severity: Severidad del caso
            
        Returns:
            Tuple[nuevo_valor, es_mejora_real]
        """
        # Si tenemos modelo aprendido, usarlo
        if self.use_learned_model and self.transition_model and self.transition_model.is_fitted:
            try:
                new_value, actual_improvement = self.transition_model.predict_next_value(
                    current_value, param_type, visit_number, severity
                )
                return new_value, actual_improvement
            except Exception as e:
                logger.warning(f"⚠️ Error en modelo aprendido para {param_type}: {e}, usando fallback")
        
        # Fallback: simulación simple (código original mejorado)
        delta = np.random.normal(-5 if improvement else 5, 2)
        new_value = current_value + delta

        if self.disease_type == 'covid19':
            if param_type == 'temperature':
                new_value = max(36.5, min(45.0, new_value)) if improvement else min(45.0, new_value)
            elif param_type == 'oxygen_saturation':
                if improvement:
                    new_value = max(95.0, min(100.0, new_value))
                else:
                    new_value = max(94.0, min(100.0, new_value))
            elif param_type == 'pcr_result':
                new_value = max(0, new_value)
        else:
            new_value = max(0, new_value)

        return new_value, improvement
    
    def simulate_single_patient_visit(self, patient_data: pd.Series, visit_day: int = 1) -> Dict[str, Any]:
        """Simula una visita individual para un paciente"""
        
        # Usar nombres de columnas estandarizados
        sat = patient_data.get('oxygen_saturation', 98.0)
        pcr = patient_data.get('pcr_result', 0.0)
        temp = patient_data.get('temperature', 37.0)
        icu_days = patient_data.get('icu_days', 0)
        
        # Determinar severidad basada en parámetros actuales
        if icu_days > 0 or sat < 90 or temp > 39:
            severity = "severe"
        elif sat < 95 or temp > 38 or pcr > 10:
            severity = "moderate"
        else:
            severity = "mild"
        
        # Aplicar evolución basada en el día y tipo de enfermedad
        new_pcr, pcr_improved = self.simulate_disease_progression(
            pcr + (visit_day * 0.5), True, 'pcr_result', visit_day, severity
        )
        
        # Determinar mejora general basada en PCR
        improvement = new_pcr < 10.0
        
        new_sat, sat_improved = self.simulate_disease_progression(
            sat, improvement, 'oxygen_saturation', visit_day, severity
        )
        new_temp, temp_improved = self.simulate_disease_progression(
            temp, improvement, 'temperature', visit_day, severity
        )
        
        # La mejora real es si la mayoría de parámetros mejoraron
        actual_improvement = sum([pcr_improved, sat_improved, temp_improved]) >= 2
        
        date = (datetime.now() + timedelta(days=visit_day)).strftime('%Y-%m-%d')
        
        visit = {
            'date': date,
            'day_hospitalization': visit_day,
            'labs': {
                "PCR": float(round(new_pcr, 2)),
                "SAT_O2": float(round(new_sat, 2)),
                "TEMP": float(round(new_temp, 2))
            },
            'severity': severity,
            'improved': actual_improvement
        }

        # Asignar síntomas y acciones según el estado y tipo de enfermedad
        visit.update(self._assign_clinical_actions(new_sat, new_pcr, actual_improvement))
        
        return visit
    
    def simulate_patient_timeline(self, patient_data: pd.Series, num_visits: int) -> List[Dict[str, Any]]:
        """Simula timeline completo para un paciente"""
        timeline = []
        
        for visit_day in range(1, num_visits + 1):
            visit_data = self.simulate_single_patient_visit(patient_data, visit_day)
            if 'error' not in visit_data:
                timeline.append(visit_data)
        
        return timeline
    
    def simulate_batch_evolution(self, patient_data: pd.DataFrame, visits_per_patient: Dict[Any, int] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Simula evolución para un lote de pacientes"""
        evolved_records = []
        stats = {
            'total_patients': len(patient_data),
            'total_visits': 0,
            'avg_visits_per_patient': 0,
            'patients_with_improvement': 0,
            'patients_with_deterioration': 0
        }
        
        for idx, patient_series in patient_data.iterrows():
            patient_id = patient_series.get('patient_id', idx) # Usar nombre estandarizado
            
            # Determinar número de visitas
            if visits_per_patient and patient_id in visits_per_patient:
                num_visits = visits_per_patient[patient_id]
            else:
                # Lógica por defecto basada en severidad (usando nombres estandarizados)
                icu_days = patient_series.get('icu_days', 0)
                oxygen_saturation = patient_series.get('oxygen_saturation', 95.0)
                temperature = patient_series.get('temperature', 37.0)

                if icu_days > 0 or oxygen_saturation < 90 or temperature > 39:
                    num_visits = np.random.randint(7, 15)  # Casos graves
                elif oxygen_saturation < 95 or temperature > 38:
                    num_visits = np.random.randint(4, 10)  # Casos moderados
                else:
                    num_visits = np.random.randint(2, 6)   # Casos leves
            
            # Generar timeline para este paciente
            patient_timeline = self.simulate_patient_timeline(patient_series, num_visits)
            
            # Convertir timeline a registros de DataFrame
            for visit in patient_timeline:
                visit_record = patient_series.copy().to_dict()
                visit_record.update({
                    'patient_id': f"{patient_id}_V{visit['day_hospitalization']}", # Mantener ID original para trazabilidad
                    'temperature': visit['labs'].get('TEMP', patient_series.get('temperature', 38.5)),
                    'oxygen_saturation': visit['labs']['SAT_O2'],
                    'pcr_result': visit['labs']['PCR'],
                    'visit_date': visit['date'],
                    'day_hospitalization': visit['day_hospitalization'],
                    'symptoms': visit.get('symptoms', []),
                    'actions': visit.get('actions', [])
                })
                evolved_records.append(visit_record)
            
            stats['total_visits'] += len(patient_timeline)

            # Actualizar estadísticas de mejora/deterioro
            initial_sat = patient_series.get('oxygen_saturation', 95.0)
            final_sat = patient_timeline[-1]['labs']['SAT_O2'] if patient_timeline else initial_sat
            if final_sat > initial_sat + 2: # Mejora significativa
                stats['patients_with_improvement'] += 1
            elif final_sat < initial_sat - 2: # Deterioro significativo
                stats['patients_with_deterioration'] += 1

        stats['avg_visits_per_patient'] = stats['total_visits'] / stats['total_patients'] if stats['total_patients'] > 0 else 0
        
        return pd.DataFrame(evolved_records), stats
    
    def _assign_clinical_actions(self, new_sat: float, new_pcr: float, improvement: bool) -> Dict[str, Any]:
        """Asigna síntomas y acciones clínicas basadas en el estado y tipo de enfermedad"""
        result = {'symptoms': [], 'actions': []}
        
        if self.disease_type == 'covid19':
            if improvement and new_sat >= 94.0:
                result['symptoms'] = ["Estabilizado"]
                result['actions'] = ["Monitoreo continuo"]
            elif new_sat >= 94.0:
                result['symptoms'] = ["Fiebre, tos sin disnea, cefalea, mialgias, náuseas, vómitos, diarrea"]
                result['actions'] = ["Descanso en cama", "Hidratación oral", "Paracetamol 500mg cada 8 horas", "Aislamiento", "Uso de mascarilla"]
            elif new_sat >= 90.0:
                result['symptoms'] = ["Agotamiento, astenia, tos, disnea y signos de afectación pulmonar"]
                result['actions'] = ["Administración de oxígeno suplementario", "Ventilación no invasiva", "Remdesivir", "Dexametasona", "Inmunomoduladores"]
            elif new_sat < 90.0 and new_pcr >= 15.0:
                result['symptoms'] = ["Estado grave con insuficiencia respiratoria, hipoxemia, alteración de funciones vitales y manifestaciones extrapulmonares"]
                result['actions'] = ["Ventilación mecánica invasiva", "OMEC", "Manejo en unidad de cuidados intensivos en cubículos con presión negativa"]
            else:
                result['symptoms'] = ["Agotamiento, astenia, tos, disnea y signos de afectación pulmonar"]
                result['actions'] = ["Administración de oxígeno suplementario", "Ventilación no invasiva", "Remdesivir", "Dexametasona", "Inmunomoduladores"]
        else: # Lógica general para otras enfermedades
            if improvement:
                result['symptoms'] = ["Mejoría general"]
                result['actions'] = ["Seguimiento ambulatorio"]
            elif new_sat < 90.0:
                result['symptoms'] = ["Deterioro respiratorio"]
                result['actions'] = ["Soporte respiratorio"]
            else:
                result['symptoms'] = ["Condición estable"]
                result['actions'] = ["Tratamiento habitual"]
        
        return result

# Función de compatibilidad con el script original (si es necesario)
def simulate_progression_for_patient(patient_id: int) -> Dict[str, Any]:
    """Función de compatibilidad - usar la clase ProgressSimulator"""
    # Esta función ahora requiere un DataFrame de entrada
    # Para mantener la compatibilidad, se podría cargar un DF de ejemplo o pasar uno vacío
    # y dejar que el simulador use sus datos de fallback.
    simulator = ProgressSimulator(data_source=pd.DataFrame())
    # Necesitamos un patient_data como pd.Series para simulate_single_patient_visit
    # Esto es un ejemplo, en un uso real, patient_id debería mapear a una fila real
    example_patient_data = simulator.df[simulator.df['patient_id'] == patient_id].iloc[0] if not simulator.df.empty else simulator._create_fallback_data().iloc[0]
    return simulator.simulate_single_patient_visit(example_patient_data)

# Ejecución standalone si se ejecuta directamente
if __name__ == "__main__":
    # Ejemplo de uso con datos de fallback
    simulator = ProgressSimulator(data_source=pd.DataFrame(), disease_type="covid19")
    # Simular un paciente específico del fallback data
    patient_to_simulate = simulator.df.iloc[0] # Tomar el primer paciente de los datos de fallback
    
    print(f"Simulando evolución para paciente: {patient_to_simulate['patient_id']}")
    timeline = simulator.simulate_patient_timeline(patient_to_simulate, num_visits=5)
    for visit in timeline:
        print(visit)

    # Ejemplo de simulación por lotes
    evolved_df, stats = simulator.simulate_batch_evolution(simulator.df.head(3)) # Simular los primeros 3 pacientes
    print("\n--- Simulación por lotes --- ")
    print(evolved_df.head())
    print(stats)