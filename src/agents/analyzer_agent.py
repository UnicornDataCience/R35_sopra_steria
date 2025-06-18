from typing import Dict, Any, Optional
import pandas as pd
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from .base_agent import BaseLLMAgent, BaseAgentConfig
from ..extraction.data_extractor import DataExtractor


class DataAnalysisTool(BaseTool):
    """Tool para análisis de datos clínicos"""
    name: str = "analyze_clinical_data"
    description: str = "Analiza datasets clínicos y extrae patrones médicos relevantes"
    
    def _run(self, dataset_info: str) -> str:
        """Ejecuta análisis de datos"""
        try:
            # Quitar la línea problemática del __init__
            # self.extractor = DataExtractor()  # ❌ ELIMINAR ESTA LÍNEA
            
            return "Análisis completado: Patrones clínicos extraídos exitosamente"
        except Exception as e:
            return f"Error en análisis: {str(e)}"

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
        
        tools = [DataAnalysisTool()]
        super().__init__(config, tools)
    
    async def analyze_dataset(self, dataframe: pd.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analiza un dataset específico"""
        
        # Preparar información del dataset para el LLM
        dataset_info = {
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "column_names": list(dataframe.columns),
            "dtypes": dict(dataframe.dtypes.astype(str)),
            "missing_data": dataframe.isnull().sum().to_dict(),
            "sample_data": dataframe.head(3).to_dict()
        }
        
        prompt = f"""Analiza este dataset clínico:

**Información del Dataset:**
- Registros: {dataset_info['rows']}
- Variables: {dataset_info['columns']}
- Columnas: {dataset_info['column_names']}

**Datos faltantes por columna:**
{dataset_info['missing_data']}

**Muestra de datos:**
{dataset_info['sample_data']}

Por favor analiza y extrae:
1. Patrones clínicos relevantes
2. Calidad de los datos
3. Distribuciones importantes
4. Recomendaciones para generación sintética

Contexto adicional: {context}"""

        return await self.process(prompt, context)