from typing import Dict, Any, Optional
import pandas as pd
from .base_agent import BaseLLMAgent, BaseAgentConfig
from ..extraction.data_extractor import DataExtractor

class ClinicalAnalyzerAgent(BaseLLMAgent):
    """Agente especializado en análisis de datos clínicos"""
    
    def __init__(self):
        config = BaseAgentConfig(
            name="Analista Clínico",
            description="Especialista en análisis de datasets médicos y extracción de patrones clínicos",
            system_prompt="""Eres un agente especializado en análisis de datos clínicos. Tu misión es:

1. **Analizar datasets médicos** con enfoque en:
   - Patrones demográficos (edad, sexo, distribuciones)
   - Diagnósticos y comorbilidades frecuentes
   - Patrones de medicación y tratamientos
   - Tendencias temporales en los datos
   - Variables clínicas críticas

2. **Identificar características relevantes**:
   - Variables con alta correlación clínica
   - Outliers médicamente significativos
   - Distribuciones anómalas que requieren atención
   - Calidad y completitud de los datos

3. **Comunicar hallazgos de manera clara**:
   - Usa terminología médica apropiada pero accesible
   - Destaca patrones clínicamente relevantes
   - Sugiere próximos pasos para generación sintética
   - Identifica posibles sesgos en los datos

4. **Contexto de COVID-19**: Presta especial atención a:
   - Criterios de ingreso y severidad
   - Patrones de medicación COVID-específicos
   - Evolución temporal de pacientes
   - Factores de riesgo y comorbilidades

Responde de manera profesional, concisa y orientada a la acción. Siempre pregunta si se necesita análisis adicional.""",
            temperature=0.1
        )
        super().__init__(config)
    
    async def analyze_dataset(self, dataframe: pd.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analiza un dataset específico"""

        # 1. Extraer patrones reales con DataExtractor
        extractor = DataExtractor()
        extraction_results = extractor.extract_patterns(dataframe)

        # 2. Preparar prompt para el LLM usando los resultados reales
        prompt = f"""Analiza este dataset clínico:

**Patrones clínicos extraídos automáticamente:**
{extraction_results['clinical_patterns']}

**Estadísticas generales:**
{extraction_results['statistics']}

Por favor interpreta estos hallazgos, identifica limitaciones y sugiere próximos pasos para generación sintética.

Contexto adicional: {context}
"""
        return await self.process(prompt, context)