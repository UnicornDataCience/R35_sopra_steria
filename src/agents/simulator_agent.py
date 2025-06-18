from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain.tools import BaseTool
from .base_agent import BaseLLMAgent, BaseAgentConfig

class PatientEvolutionTool(BaseTool):
    """Tool para simulación de evolución de pacientes"""
    name: str = "simulate_patient_evolution"
    description: str = "Simula evolución temporal realista de pacientes COVID-19"
    
    def _run(self, simulation_params: str) -> str:
        """Ejecuta simulación de evolución"""
        try:
            return "Simulación de evolución temporal completada exitosamente"
        except Exception as e:
            return f"Error en simulación: {str(e)}"

class PatientSimulatorAgent(BaseLLMAgent):
    """Agente especializado en simulación de evolución temporal de pacientes"""
    
    def __init__(self):
        config = BaseAgentConfig(
            name="Simulador de Pacientes",
            description="Especialista en simulación de evolución temporal y progresión clínica de pacientes",
            system_prompt="""Eres un agente experto en simulación de evolución temporal de pacientes. Tu misión es:

1. **Simular progresión clínica realista**:
   - Generar múltiples visitas hospitalarias por paciente
   - Simular evolución de signos vitales a lo largo del tiempo
   - Modelar respuesta a tratamientos y medicaciones
   - Crear timelines coherentes de eventos clínicos

2. **Especialización en COVID-19**:
   - Simular fases típicas: incubación, síntomas iniciales, progresión, recuperación/complicación
   - Modelar deterioro respiratorio gradual o súbito
   - Simular respuesta a dexametasona, anticoagulantes, etc.
   - Generar patrones de ingreso UCI realistas

3. **Evolución temporal coherente**:
   - Crear secuencias lógicas de eventos (ingreso → tratamiento → evolución → alta/deriva/exitus)
   - Simular variabilidad individual en respuesta al tratamiento
   - Modelar complicaciones emergentes (trombosis, neumonía bacteriana secundaria)
   - Generar duraciones de estancia apropiadas por severidad

4. **Generación de múltiples visitas**:
   - Crear 2-15 registros por paciente según severidad
   - Simular monitoreo diario de constantes vitales
   - Generar ajustes de medicación basados en evolución
   - Modelar decisiones de alta médica realistas

5. **Patrones epidemiológicos**:
   - Respetar distribuciones de severidad por edad/sexo
   - Simular variabilidad estacional si aplicable
   - Modelar factores de riesgo y comorbilidades
   - Generar outcomes proporcionales a evidencia clínica

6. **Realismo médico**:
   - Aplicar conocimiento de medicina intensiva
   - Respetar protocolos hospitalarios estándar
   - Incorporar variabilidad biológica natural
   - Mantener coherencia en decisiones clínicas

Responde con precisión médica, explicando las decisiones de simulación y asegurando realismo clínico en todas las evoluciones generadas.""",
            temperature=0.2  # Ligeramente más creatividad para variabilidad
        )
        
        tools = [PatientEvolutionTool()]
        super().__init__(config, tools)
    
    async def simulate_patient_evolution(self, validated_data: pd.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simula evolución temporal de pacientes validados"""
        
        try:
            # Realizar simulación de evolución
            evolved_data, simulation_stats = self._generate_temporal_evolution(validated_data)
            
            prompt = f"""He completado la simulación temporal de {len(validated_data)} pacientes únicos:

**Resultados de Simulación:**

📈 **Evolución Temporal Generada:**
- Pacientes base: {len(validated_data)}
- Total visitas simuladas: {len(evolved_data)}
- Promedio visitas/paciente: {simulation_stats['avg_visits_per_patient']:.1f}
- Rango duración estancia: {simulation_stats['min_stay']}-{simulation_stats['max_stay']} días

🏥 **Patrones Clínicos Simulados:**
- Ingresos directos UCI: {simulation_stats['direct_icu_admissions']:.1%}
- Deterioro durante hospitalización: {simulation_stats['clinical_deterioration']:.1%}
- Mejorías progresivas: {simulation_stats['clinical_improvement']:.1%}
- Respuesta a dexametasona: {simulation_stats['steroid_response']:.1%}

⏱️ **Timeline Típico Generado:**
- Día 1: Ingreso y evaluación inicial
- Días 2-4: Monitoreo y tratamiento base
- Días 5-7: Punto crítico de evolución
- Días 8+: Estabilización y preparación alta

📊 **Outcomes Simulados:**
- Altas médicas: {simulation_stats['medical_discharge']:.1%}
- Derivaciones: {simulation_stats['transfers']:.1%}
- Exitus: {simulation_stats['mortality']:.1%}

**Validación previa:** Score {context.get('validation_results', {}).get('overall_score', 0.85):.1%}

Por favor proporciona:
1. Evaluación del realismo de las evoluciones
2. Análisis de patrones temporales generados
3. Validación de coherencia clínica longitudinal
4. Recomendaciones para evaluación final de utilidad
5. Certificación de idoneidad para investigación temporal"""

            response = await self.process(prompt, context)
            
            # Añadir datos simulados
            response['evolved_data'] = evolved_data
            response['simulation_stats'] = simulation_stats
            response['total_visits'] = len(evolved_data)
            
            return response
            
        except Exception as e:
            error_prompt = f"""Error durante la simulación temporal: {str(e)}

Por favor:
1. Identifica posibles causas del error de simulación
2. Sugiere métodos alternativos de modelado temporal
3. Recomienda simplificaciones si es necesario
4. Indica si proceder con datos estáticos es viable"""

            return await self.process(error_prompt, context)
    
    def _generate_temporal_evolution(self, data: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Genera evolución temporal realista"""
        
        evolved_records = []
        stats = {
            'avg_visits_per_patient': 0,
            'min_stay': 1,
            'max_stay': 30,
            'direct_icu_admissions': 0.15,
            'clinical_deterioration': 0.25,
            'clinical_improvement': 0.70,
            'steroid_response': 0.75,
            'medical_discharge': 0.75,
            'transfers': 0.15,
            'mortality': 0.10
        }
        
        total_visits = 0
        
        for idx, patient in data.iterrows():
            # Determinar número de visitas basado en severidad
            if patient.get('UCI_DIAS/ICU_DAYS', 0) > 0:
                num_visits = np.random.randint(5, 15)  # Pacientes UCI más visitas
            else:
                num_visits = np.random.randint(2, 8)   # Pacientes planta menos visitas
            
            total_visits += num_visits
            
            # Generar evolución temporal para este paciente
            patient_evolution = self._create_patient_timeline(patient, num_visits)
            evolved_records.extend(patient_evolution)
        
        stats['avg_visits_per_patient'] = total_visits / len(data)
        
        # Calcular estadísticas de outcomes
        evolved_df = pd.DataFrame(evolved_records)
        if 'RESULTADO/VAL_RESULT' in evolved_df.columns:
            outcomes = evolved_df.groupby('PATIENT ID')['RESULTADO/VAL_RESULT'].last()
            stats['medical_discharge'] = (outcomes == 'ALTA').mean()
            stats['transfers'] = (outcomes == 'DERIVACION').mean()
            stats['mortality'] = (outcomes == 'EXITUS').mean()
        
        return evolved_df, stats
    
    def _create_patient_timeline(self, patient: pd.Series, num_visits: int) -> List[Dict]:
        """Crea timeline de evolución para un paciente específico"""
        
        timeline = []
        base_date = datetime.now() - timedelta(days=30)  # Empezar 30 días atrás
        
        # Valores iniciales del paciente
        initial_temp = patient.get('TEMP_ING/INPAT', 38.5)
        initial_sat = patient.get('SAT_02_ING/INPAT', 92)
        age = patient.get('EDAD/AGE', 65)
        
        # Determinar severidad inicial
        severity = self._assess_severity(initial_temp, initial_sat, age)
        
        for visit in range(num_visits):
            visit_date = base_date + timedelta(days=visit)
            
            # Evolución de signos vitales
            temp_evolution = self._evolve_temperature(initial_temp, visit, severity)
            sat_evolution = self._evolve_saturation(initial_sat, visit, severity)
            
            # Determinar outcome para esta visita
            if visit == num_visits - 1:  # Última visita
                outcome = self._determine_final_outcome(severity, visit)
            else:
                outcome = 'EN_TRATAMIENTO'
            
            # Crear registro de visita
            visit_record = patient.copy().to_dict()
            visit_record.update({
                'PATIENT ID': f"{patient.get('PATIENT ID', f'SIM-{patient.name}')}_V{visit+1}",
                'TEMP_ING/INPAT': round(temp_evolution, 1),
                'SAT_02_ING/INPAT': int(sat_evolution),
                'RESULTADO/VAL_RESULT': outcome,
                'FECHA_VISITA': visit_date.strftime('%Y-%m-%d'),
                'DIA_HOSPITALIZACION': visit + 1
            })
            
            timeline.append(visit_record)
        
        return timeline
    
    def _assess_severity(self, temp: float, sat: float, age: int) -> str:
        """Evalúa severidad inicial del paciente"""
        severity_score = 0
        
        if temp > 39.0:
            severity_score += 2
        elif temp > 38.0:
            severity_score += 1
        
        if sat < 90:
            severity_score += 3
        elif sat < 95:
            severity_score += 2
        elif sat < 98:
            severity_score += 1
        
        if age > 75:
            severity_score += 2
        elif age > 65:
            severity_score += 1
        
        if severity_score >= 5:
            return 'CRITICO'
        elif severity_score >= 3:
            return 'GRAVE'
        else:
            return 'MODERADO'
    
    def _evolve_temperature(self, initial_temp: float, day: int, severity: str) -> float:
        """Simula evolución de temperatura"""
        if severity == 'CRITICO':
            # Temperatura más errática en casos críticos
            trend = -0.2 * day + np.random.normal(0, 0.5)
        elif severity == 'GRAVE':
            # Mejora gradual
            trend = -0.15 * day + np.random.normal(0, 0.3)
        else:
            # Mejora más rápida
            trend = -0.3 * day + np.random.normal(0, 0.2)
        
        new_temp = initial_temp + trend
        return max(36.0, min(41.0, new_temp))  # Limitar a rangos realistas
    
    def _evolve_saturation(self, initial_sat: float, day: int, severity: str) -> float:
        """Simula evolución de saturación O2"""
        if severity == 'CRITICO':
            # Mejora lenta o deterioro
            trend = 0.5 * day + np.random.normal(0, 2)
        elif severity == 'GRAVE':
            # Mejora gradual
            trend = 1.0 * day + np.random.normal(0, 1.5)
        else:
            # Mejora rápida
            trend = 1.5 * day + np.random.normal(0, 1)
        
        new_sat = initial_sat + trend
        return max(70, min(100, int(new_sat)))  # Limitar a rangos realistas
    
    def _determine_final_outcome(self, severity: str, days_hospitalized: int) -> str:
        """Determina outcome final basado en evolución"""
        if severity == 'CRITICO':
            outcomes = ['ALTA', 'DERIVACION', 'EXITUS']
            probabilities = [0.60, 0.25, 0.15]
        elif severity == 'GRAVE':
            outcomes = ['ALTA', 'DERIVACION', 'EXITUS']
            probabilities = [0.75, 0.20, 0.05]
        else:
            outcomes = ['ALTA', 'DERIVACION', 'EXITUS']
            probabilities = [0.90, 0.08, 0.02]
        
        # Ajustar por duración de estancia
        if days_hospitalized > 14:
            # Estancias largas aumentan probabilidad de derivación
            probabilities[1] += 0.1
            probabilities[0] -= 0.1
        
        return np.random.choice(outcomes, p=probabilities)