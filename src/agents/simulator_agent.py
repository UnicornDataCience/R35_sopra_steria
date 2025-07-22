from typing import Dict, Any
import pandas as pd
from .base_agent import BaseLLMAgent, BaseAgentConfig
from src.simulation.progress_simulator import ProgressSimulator

class PatientSimulatorAgent(BaseLLMAgent):
    """Agente especializado en simulación de evolución temporal de pacientes"""
    
    def __init__(self):
        config = BaseAgentConfig(
            name="Simulador de Pacientes",
            description="Especialista en simulación de evolución temporal y progresión clínica de pacientes",
            system_prompt="Eres un agente experto en simulación de evolución temporal de pacientes. Recibes un resumen de simulación y tu tarea es interpretarlo y presentar un informe claro y conciso en Markdown, evaluando el realismo de las evoluciones generadas."
        )
        super().__init__(config, tools=[])  # Explícitamente sin herramientas
        # No inicializar el simulador aquí, se hará cuando se use

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Punto de entrada principal para la simulación."""
        context = context or {}
        validated_data = context.get("synthetic_data") # Usar datos sintéticos validados

        if validated_data is None:
            return {"message": "Error: Se necesitan datos validados para la simulación.", "agent": self.name, "error": True}

        try:
            # Determinar el tipo de enfermedad basado en el análisis universal
            is_covid = context.get('universal_analysis', {}).get('dataset_type') == 'COVID-19'
            disease_type = "covid19" if is_covid else "general"  # Corregir nombre para el simulador
            
            # Inicializar el simulador con los datos
            simulation_engine = ProgressSimulator(validated_data, disease_type)
            
            # Llamar sin el parámetro is_covid que no existe en la función
            evolved_data, stats = simulation_engine.simulate_batch_evolution(validated_data)

            prompt = self._create_llm_prompt(stats, len(validated_data))
            informe_markdown = await self.agent_executor.ainvoke({"input": prompt, "chat_history": self.memory.chat_memory.messages})

            return {
                "message": informe_markdown.content,
                "agent": self.name,
                "evolved_data": evolved_data,
                "simulation_stats": stats
            }
        except Exception as e:
            return {"message": f"Error durante la simulación: {e}", "agent": self.name, "error": True}

    def _create_llm_prompt(self, stats: Dict[str, Any], num_patients: int) -> str:
        """Crea el prompt para el LLM a partir de las estadísticas de simulación."""
        return f"""Resultados de la simulación para {num_patients} pacientes:
        - Total visitas simuladas: {stats.get('total_visits', 0)}
        - Promedio visitas/paciente: {stats.get('avg_visits_per_patient', 0):.1f}
        - Pacientes con mejoría: {stats.get('patients_with_improvement', 0)}
        - Pacientes con deterioro: {stats.get('patients_with_deterioration', 0)}

Por favor, genera el informe en Markdown."""