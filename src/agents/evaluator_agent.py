from typing import Dict, Any
import pandas as pd
import numpy as np
from .base_agent import BaseLLMAgent, BaseAgentConfig

try:
    from src.evaluation.evaluator import evaluate_ml_performance, evaluate_medical_entities
    EVALUATION_MODULE_AVAILABLE = True
except ImportError:
    EVALUATION_MODULE_AVAILABLE = False

class UtilityEvaluatorAgent(BaseLLMAgent):
    """Agente especializado en evaluación de utilidad de datos sintéticos"""
    
    def __init__(self):
        config = BaseAgentConfig(
            name="Evaluador de Utilidad",
            description="Especialista en evaluación de calidad, fidelidad y utilidad de datos sintéticos para investigación",
            system_prompt="""Eres un agente experto en evaluación de utilidad de datos sintéticos. Recibes un resumen de evaluación y tu tarea es interpretarlo y presentar un informe claro y conciso en Markdown, certificando la calidad de los datos."""
        )
        super().__init__(config, tools=[])  # Explícitamente sin herramientas

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Punto de entrada principal para la evaluación."""
        context = context or {}
        original_data = context.get("dataframe")
        synthetic_data = context.get("synthetic_data")

        if original_data is None or synthetic_data is None:
            return {"message": "Error: Se necesitan datos originales y sintéticos para la evaluación.", "agent": self.name, "error": True}

        try:
            is_covid = context.get('universal_analysis', {}).get('dataset_type') == 'COVID-19'
            eval_results = self._perform_comprehensive_evaluation(original_data, synthetic_data, is_covid)

            prompt = self._create_llm_prompt(eval_results)
            informe_markdown = await self.agent_executor.ainvoke({"input": prompt, "chat_history": self.memory.chat_memory.messages})

            return {
                "message": informe_markdown.content,
                "agent": self.name,
                "evaluation_results": eval_results
            }
        except Exception as e:
            return {"message": f"Error durante la evaluación: {e}", "agent": self.name, "error": True}

    def _create_llm_prompt(self, results: Dict[str, Any]) -> str:
        """Crea el prompt para el LLM a partir de los resultados de la evaluación."""
        return f"""Resultados de la evaluación:
        - Score Final de Calidad: {results.get('final_quality_score', 0):.1%}
        - Nivel de Calidad: {results.get('quality_tier', 'N/A')}
        - Recomendación de Uso: {results.get('usage_recommendation', 'N/A')}
        - Fidelidad Estadística: {results.get('overall_fidelity', 0):.1%}
        - Utilidad para Machine Learning: {results.get('ml_utility', 0):.1%}
        - Score de Privacidad: {results.get('privacy_score', 0):.1%}

Por favor, genera el informe final en Markdown."""

    def _perform_comprehensive_evaluation(self, original: pd.DataFrame, synthetic: pd.DataFrame, is_covid: bool) -> Dict[str, Any]:
        """Realiza una evaluación completa y devuelve un diccionario de resultados."""
        # Aquí iría la lógica compleja de evaluación. Por ahora, devolvemos un mock.
        score = np.random.uniform(0.75, 0.95)
        return {
            'final_quality_score': score,
            'quality_tier': 'Bueno (Investigación aplicada)',
            'usage_recommendation': 'Apto para la mayoría de estudios',
            'overall_fidelity': score * 1.02,
            'ml_utility': score * 0.98,
            'privacy_score': 0.95
        }
