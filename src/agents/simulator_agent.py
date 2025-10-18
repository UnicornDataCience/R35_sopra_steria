from typing import Dict, Any
import pandas as pd
from .base_agent import BaseLLMAgent, BaseAgentConfig
from src.simulation.progress_simulator import ProgressSimulator
from src.simulation.visualization import SimulationVisualizer
from src.utils.logging_config import get_logger
from src.utils.image_embedding import enhance_response_with_images

logger = get_logger(__name__)

class PatientSimulatorAgent(BaseLLMAgent):
    """Agente especializado en simulación de evolución temporal de pacientes con visualizaciones"""
    
    def __init__(self, enable_visualizations: bool = True):
        config = BaseAgentConfig(
            name="Simulador de Pacientes",
            description="Especialista en simulación de evolución temporal y progresión clínica de pacientes",
            system_prompt="Eres un agente experto en simulación de evolución temporal de pacientes. Recibes un resumen de simulación con estadísticas y visualizaciones, y tu tarea es interpretarlo y presentar un informe claro y conciso en Markdown, evaluando el realismo de las evoluciones generadas y destacando patrones interesantes.",
            max_tokens=3000
        )
        super().__init__(config, tools=[])
        self.enable_visualizations = enable_visualizations
        self.visualizer = SimulationVisualizer() if enable_visualizations else None

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Punto de entrada principal para la simulación."""
        context = context or {}
        
        # 🔥 Priorizar datos sintéticos, pero aceptar originales si no hay sintéticos
        validated_data = context.get("synthetic_data")
        if validated_data is None:
            validated_data = context.get("dataframe")
        
        if validated_data is None or (isinstance(validated_data, pd.DataFrame) and validated_data.empty):
            logger.error("❌ No hay datos disponibles para simulación")
            return {"message": "Error: Se necesitan datos para la simulación.", "agent": self.name, "error": True}

        try:
            logger.info("🔬 Iniciando simulación con %d registros", len(validated_data))
            
            # Determinar el tipo de enfermedad basado en el análisis universal
            is_covid = context.get('universal_analysis', {}).get('dataset_type') == 'COVID-19'
            disease_type = "covid19" if is_covid else "general"
            logger.info("📋 Tipo de enfermedad detectado: %s", disease_type)
            
            # Inicializar el simulador con los datos y modelo aprendido
            simulation_engine = ProgressSimulator(validated_data, disease_type, use_learned_model=True)
            
            # Ejecutar simulación
            evolved_data, stats = simulation_engine.simulate_batch_evolution(validated_data)
            logger.info("✅ Simulación completada - Stats: %s", stats)

            # Generar visualizaciones si están habilitadas
            timeline_path = None
            heatmap_path = None
            
            if self.enable_visualizations and self.visualizer and not evolved_data.empty:
                try:
                    logger.info("📊 Generando visualizaciones...")
                    
                    # Determinar parámetros según tipo de enfermedad
                    if disease_type == "covid19":
                        params = ['oxygen_saturation', 'temperature', 'pcr_result']
                    else:
                        # Detectar columnas numéricas automáticamente
                        numeric_cols = evolved_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
                        params = [col for col in numeric_cols if col not in ['patient_id', 'visit_number', 'day_hospitalization']][:3]
                    
                    timeline_path = self.visualizer.create_patient_timeline_plot(evolved_data, params=params)
                    heatmap_path = self.visualizer.create_summary_heatmap(evolved_data, params=params)
                    
                    logger.info("✅ Visualizaciones generadas")
                except Exception as e:
                    logger.warning(f"⚠️ Error generando visualizaciones: {e}")

            # Crear prompt enriquecido con información de visualizaciones
            prompt = self._create_llm_prompt(stats, len(validated_data), timeline_path, heatmap_path)
            informe_markdown = await self.agent_executor.ainvoke({"input": prompt, "chat_history": self.memory.chat_memory.messages})

            # Log del contenido generado
            content = informe_markdown.content if hasattr(informe_markdown, 'content') else str(informe_markdown)
            logger.info("📄 Informe de simulación generado - Longitud: %d caracteres", len(content))

            # Crear respuesta base
            response = {
                "message": content,
                "agent": self.name,
                "evolved_data": evolved_data,
                "simulation_stats": stats,
                "timeline_path": timeline_path,
                "heatmap_path": heatmap_path
            }
            
            # 🎨 Mejorar respuesta con imágenes embebidas (base64)
            response = enhance_response_with_images(response, embed_images=True)
            
            return response
        except Exception as e:
            logger.error("❌ Error durante la simulación: %s", e, exc_info=True)
            return {"message": f"Error durante la simulación: {e}", "agent": self.name, "error": True}

    def _create_llm_prompt(
        self,
        stats: Dict[str, Any],
        num_patients: int,
        timeline_path: str = None,
        heatmap_path: str = None
    ) -> str:
        """Crea el prompt para el LLM a partir de las estadísticas de simulación."""
        prompt = f"""Resultados de la simulación temporal de evolución para {num_patients} pacientes:

📊 **Estadísticas Generales:**
- Total visitas simuladas: {stats.get('total_visits', 0)}
- Promedio visitas/paciente: {stats.get('avg_visits_per_patient', 0):.1f}
- Pacientes con mejoría: {stats.get('patients_with_improvement', 0)} ({stats.get('patients_with_improvement', 0) / num_patients * 100:.1f}%)
- Pacientes con deterioro: {stats.get('patients_with_deterioration', 0)} ({stats.get('patients_with_deterioration', 0) / num_patients * 100:.1f}%)
"""
        
        if timeline_path:
            prompt += f"\n📈 **Visualización de Evolución Temporal disponible en:** {timeline_path}"
        
        if heatmap_path:
            prompt += f"\n🔥 **Mapa de Calor de Evolución disponible en:** {heatmap_path}"
        
        prompt += """

Por favor, genera un informe clínico en Markdown que:
1. Resuma los resultados de forma clara y estructurada
2. Interprete las tasas de mejoría y deterioro
3. Evalúe el realismo de las evoluciones simuladas
4. Destaque patrones interesantes observados
5. Si hay visualizaciones, menciónelas y explica cómo interpretarlas

Usa formato Markdown profesional con secciones, listas y énfasis donde sea apropiado."""
        
        return prompt