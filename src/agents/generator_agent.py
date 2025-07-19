from typing import Dict, Any
import pandas as pd
from .base_agent import BaseLLMAgent, BaseAgentConfig
from ..generation.ctgan_generator import CTGANGenerator
from ..generation.tvae_generator import TVAEGenerator
from ..generation.sdv_generator import SDVGenerator

class SyntheticGeneratorAgent(BaseLLMAgent):
    """Agente especializado en generación de datos sintéticos"""
    
    def __init__(self):
        config = BaseAgentConfig(
            name="Generador Sintético",
            description="Especialista en generación de datos clínicos sintéticos usando SDV y técnicas de ML avanzadas",
            system_prompt="Eres un agente experto en generación de datos sintéticos médicos. Tu misión es generar datos de alta calidad y responder de forma técnica y accesible."
        )
        super().__init__(config, tools=[])  # Explícitamente sin herramientas
        self.ctgan_generator = CTGANGenerator()
        self.tvae_generator = TVAEGenerator()
        self.sdv_generator = SDVGenerator()

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Punto de entrada principal para el agente generador."""
        context = context or {}
        original_data = context.get("dataframe")
        if original_data is None:
            return {"message": "Error: No se encontró un dataset base para la generación.", "agent": self.name, "error": True}

        # Extraer parámetros de la solicitud del coordinador
        params = context.get("parameters", {})
        num_samples = params.get("num_samples", 100)
        model_type = params.get("model_type", "ctgan")

        try:
            synthetic_data = await self.generate_synthetic_data(original_data, num_samples, model_type, context)
            return {
                "message": f"""Se han generado exitosamente {len(synthetic_data)} registros sintéticos con el modelo {model_type.upper()}.

**Detalles:**
- **Modelo Utilizado:** {model_type.upper()}
- **Registros Generados:** {len(synthetic_data)}
- **Dataset Base:** {context.get('filename', 'N/A')} ({len(original_data)} registros)""",
                "agent": self.name,
                "synthetic_data": synthetic_data
            }
        except Exception as e:
            return {"message": f"Error durante la generación de datos: {e}", "agent": self.name, "error": True}

    async def generate_synthetic_data(self, original_data: pd.DataFrame, num_samples: int, model_type: str, context: Dict[str, Any]) -> pd.DataFrame:
        """Genera datos sintéticos basados en el dataset original."""
        is_covid_dataset = context.get('universal_analysis', {}).get('dataset_type') == 'COVID-19'
        
        if model_type == 'ctgan':
            return self.ctgan_generator.generate(original_data, num_samples, is_covid_dataset)
        elif model_type == 'tvae':
            return self.tvae_generator.generate(original_data, num_samples, is_covid_dataset)
        elif model_type == 'sdv':
            return self.sdv_generator.generate(original_data, num_samples, is_covid_dataset)
        else:
            raise ValueError(f"Modelo '{model_type}' no soportado.")
