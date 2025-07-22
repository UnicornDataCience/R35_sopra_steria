"""
Agente Analizador Clínico - Especializado en la interpretación y generación de informes.
"""

import json
from typing import Dict, Any
from .base_agent import BaseLLMAgent, BaseAgentConfig

ANALYZER_SYSTEM_PROMPT = """
Eres un Científico de Datos especializado en salud. Tu tarea es recibir un análisis técnico de un dataset (en JSON) y redactar un informe de análisis exploratorio (EDA) en Markdown.

**ENTRADA (JSON):**
Recibirás un JSON con la estructura:

```json
{{
  "dataset_type": "<tipo>",
  "column_mapping": {{"age_col": "<col>", ...}},
  "basic_stats": {{"rows": <num>, ...}},
  "missing_values": {{"total_missing": <num>, ...}},
  "column_analysis": {{"<col_1>": {{"type": "...", ...}}}}
}}
```

**TAREA (MARKDOWN):**
Tu ÚNICA salida debe ser un informe en MARKDOWN con estas secciones:

1.  **`### 📝 Resumen Ejecutivo`**: Párrafo con los hallazgos clave.
2.  **`### 📊 Análisis Descriptivo`**: Características del dataset.
3.  **`### 🩺 Calidad de los Datos`**: Evaluación de nulos y duplicados.
4.  **`### 🔬 Análisis de Variables Clave`**: Descripción de las variables más importantes (edad, género, diagnóstico).
5.  **`### 💡 Conclusiones y Recomendaciones`**: Idoneidad del dataset para IA y posibles sesgos.

Basa todas tus afirmaciones en los datos del JSON. Sé profesional y objetivo.
"""

class ClinicalAnalyzerConfig(BaseAgentConfig):
    name: str = "Analizador Clínico"
    description: str = "Especialista en análisis estadístico y exploratorio de datos médicos."
    system_prompt: str = ANALYZER_SYSTEM_PROMPT
    max_tokens: int = 2500

class ClinicalAnalyzerAgent(BaseLLMAgent):
    def __init__(self):
        super().__init__(ClinicalAnalyzerConfig(), tools=[])  # Explícitamente sin herramientas

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        # El análisis se dispara por `analyze_dataset`, no por `process`.
        if context and context.get("universal_analysis"):
            return await self.analyze_dataset(None, context)
        return {"message": "El analizador necesita un análisis universal previo.", "agent": self.name, "error": True}

    async def analyze_dataset(self, dataframe, context: Dict[str, Any] = None) -> Dict[str, Any]:
        universal_analysis_result = context.get("universal_analysis")
        if not universal_analysis_result:
            return {"message": "Error: No se encontró el resultado del análisis universal.", "agent": self.name, "error": True}

        prompt_input = json.dumps(universal_analysis_result, indent=2)
        
        # Construir el diccionario de entrada para el prompt
        prompt_variables = {
            "input": prompt_input,
        }
        
        # Agregar chat_history si hay mensajes en memoria
        if self.memory.chat_memory.messages:
            prompt_variables["chat_history"] = self.memory.chat_memory.messages
        
        llm_response = await self.agent_executor.ainvoke(prompt_variables)
        
        # Manejar diferentes tipos de respuesta del LLM
        if hasattr(llm_response, 'content'):
            response_text = llm_response.content
        elif isinstance(llm_response, str):
            response_text = llm_response
        else:
            response_text = str(llm_response)
        
        return {
            "message": response_text,
            "agent": self.name,
            "analysis_complete": True
        }