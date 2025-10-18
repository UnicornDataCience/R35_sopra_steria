"""
Agente Analizador Clínico - Especializado en la interpretación y generación de informes.
"""

import json
from typing import Dict, Any
from src.utils.logging_config import get_logger
from .base_agent import BaseLLMAgent, BaseAgentConfig

logger = get_logger(__name__)

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
        if context and context.get("universal_analysis"):
            return await self.analyze_dataset(None, context)
        return {"message": "El analizador necesita un análisis universal previo.", "agent": self.name, "error": True}

    async def analyze_dataset(self, dataframe, context: Dict[str, Any] = None) -> Dict[str, Any]:
        universal_analysis_result = context.get("universal_analysis")
        if not universal_analysis_result:
            return {"message": "Error: No se encontró el resultado del análisis universal.", "agent": self.name, "error": True}

        # 🔥 OPTIMIZACIÓN: Reducir el tamaño del contexto para evitar exceder límites de tokens
        # Especialmente importante para datasets grandes (>1000 filas o >50 columnas)
        summary_analysis = self._summarize_analysis_for_llm(universal_analysis_result)
        
        prompt_input = json.dumps(summary_analysis, indent=2)
        
        # 🔥 LÍMITE ADICIONAL: Truncar el JSON si es muy grande (>8000 caracteres)
        MAX_JSON_CHARS = 8000
        if len(prompt_input) > MAX_JSON_CHARS:
            logger.warning("JSON muy grande (%d chars), truncando a %d chars", len(prompt_input), MAX_JSON_CHARS)
            prompt_input = prompt_input[:MAX_JSON_CHARS] + "\n... (truncado por límite de tokens)"
        
        prompt_variables = {"input": prompt_input}
        if self.memory.chat_memory.messages:
            prompt_variables["chat_history"] = self.memory.chat_memory.messages
        
        logger.info("Generando informe EDA con LLM (%s) - Tamaño JSON: %d chars", 
                   type(self.llm).__name__ if self.llm else "mock", len(prompt_input))
        llm_response = await self.agent_executor.ainvoke(prompt_variables)
        response_text = self._extract_response_text(llm_response)
        
        return {
            "message": response_text,
            "agent": self.name,
            "analysis_complete": True
        }
    
    def _summarize_analysis_for_llm(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Resumir análisis para evitar exceder límites de tokens del LLM"""
        summary = {}
        
        # Información de muestreo (si aplica)
        if "sampling_info" in analysis:
            summary["sampling_info"] = analysis["sampling_info"]
        
        # Resumen básico
        if "basic_info" in analysis:
            summary["basic_info"] = {
                "rows": analysis["basic_info"].get("rows", 0),
                "columns": analysis["basic_info"].get("columns", 0),
                "memory_usage": analysis["basic_info"].get("memory_usage", "N/A")
            }
        
        # Dominio médico
        if "medical_domain" in analysis:
            summary["medical_domain"] = analysis["medical_domain"]
        
        # Columnas: SOLO conteo y primeros 10 nombres
        if "columns" in analysis:
            cols = analysis["columns"]
            numerical = [c["name"] for c in cols if c.get("type") == "numerical"]
            categorical = [c["name"] for c in cols if c.get("type") == "categorical"]
            datetime_cols = [c["name"] for c in cols if c.get("type") == "datetime"]
            
            summary["columns_summary"] = {
                "total": len(cols),
                "numerical_count": len(numerical),
                "categorical_count": len(categorical),
                "datetime_count": len(datetime_cols),
                "sample_numerical": numerical[:10],
                "sample_categorical": categorical[:10],
                "sample_datetime": datetime_cols[:5]
            }
        
        # Valores nulos: SOLO total y porcentaje
        if "missing_values" in analysis:
            mv = analysis["missing_values"]
            total_cells = summary["basic_info"]["rows"] * summary["basic_info"]["columns"]
            missing_pct = (mv.get("total_missing", 0) / total_cells * 100) if total_cells > 0 else 0
            
            summary["missing_values"] = {
                "total_missing": mv.get("total_missing", 0),
                "percentage": round(missing_pct, 2),
                "columns_affected": len(mv.get("columns_with_missing", {}))
            }
        
        # Correlaciones: solo top 5
        if "correlations" in analysis and "high_correlations" in analysis["correlations"]:
            high_corr = analysis["correlations"]["high_correlations"]
            summary["high_correlations"] = high_corr[:5] if isinstance(high_corr, list) else []
        
        # Patrones médicos: solo resumen
        if "medical_patterns" in analysis:
            mp = analysis["medical_patterns"]
            summary["medical_patterns"] = {
                "has_patient_id": mp.get("has_patient_id", False),
                "has_age": mp.get("has_age", False),
                "has_dates": mp.get("has_dates", False),
                "has_diagnoses": mp.get("has_diagnoses", False),
                "key_medical_columns": mp.get("key_medical_columns", [])[:10]
            }
        
        return summary