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
            
            # Crear información de generación detallada
            generation_info = {
                "model_type": model_type,
                "num_samples": len(synthetic_data),
                "columns_used": len(synthetic_data.columns),
                "selection_method": "MedicalColumnSelector" if context.get('selected_columns') else "Default",
                "timestamp": pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            }
            
            return {
                "message": f"""Se han generado exitosamente {len(synthetic_data)} registros sintéticos con el modelo {model_type.upper()}.

**Detalles:**
- **Modelo Utilizado:** {model_type.upper()}
- **Registros Generados:** {len(synthetic_data)}
- **Dataset Base:** {context.get('filename', 'N/A')} ({len(original_data)} registros)""",
                "agent": self.name,
                "synthetic_data": synthetic_data,
                "generation_info": generation_info
            }
        except Exception as e:
            return {"message": f"Error durante la generación de datos: {e}", "agent": self.name, "error": True}

    async def generate_synthetic_data(self, original_data: pd.DataFrame, num_samples: int, model_type: str, context: Dict[str, Any]) -> pd.DataFrame:
        """Genera datos sintéticos basados en el dataset original."""
        is_covid_dataset = context.get('universal_analysis', {}).get('dataset_type') == 'COVID-19'
        
        # Obtener columnas seleccionadas del contexto
        selected_columns = context.get('selected_columns')
        
        # Filtrar DataFrame si hay columnas seleccionadas
        if selected_columns:
            # Verificar que las columnas seleccionadas existen en el DataFrame
            available_columns = [col for col in selected_columns if col in original_data.columns]
            if available_columns:
                print(f"🎯 Usando {len(available_columns)} columnas seleccionadas: {available_columns}")
                original_data = original_data[available_columns].copy()
            else:
                print("⚠️ Ninguna de las columnas seleccionadas existe en el DataFrame. Usando dataset completo.")
        
        if model_type == 'ctgan':
            return self.ctgan_generator.generate(original_data, num_samples, is_covid_dataset, selected_columns)
        elif model_type == 'tvae':
            return self.tvae_generator.generate(original_data, num_samples, is_covid_dataset, selected_columns)
        elif model_type == 'sdv':
            return self.sdv_generator.generate(original_data, num_samples, is_covid_dataset, selected_columns)
        else:
            raise ValueError(f"Modelo '{model_type}' no soportado.")
