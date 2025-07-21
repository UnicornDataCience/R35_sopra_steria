from typing import Dict, Any
import pandas as pd
import numpy as np
from .base_agent import BaseLLMAgent, BaseAgentConfig
from src.validation.clinical_rules import validate_patient_case
from src.validation.json_schema import validate_json, pacient_schema

class MedicalValidatorAgent(BaseLLMAgent):
    """Agente especializado en validación médica de datos sintéticos"""
    
    def __init__(self):
        config = BaseAgentConfig(
            name="Validador Médico",
            description="Especialista en validación de coherencia médica y clínica de datos sintéticos",
            system_prompt="""Eres un agente experto en validación médica de datos sintéticos. Recibirás un resumen de validación y tu tarea es interpretarlo y presentar un informe claro y conciso en Markdown, evaluando si los datos son aptos para investigación.

**Ejemplo de Informe:**

### 📋 Resumen de Validación Médica

El dataset sintético muestra una **alta coherencia general (XX.X%)**.

- **Coherencia Clínica (XX.X%):** Los signos vitales y las correlaciones demográficas son realistas.
- **Calidad de Datos (XX.X%):** La estructura de los datos es sólida, con un bajo número de errores de esquema.

**⚠️ Puntos de Atención:**
- Lista de problemas encontrados

**Conclusión:** Los datos son **aptos para su uso en investigación y entrenamiento de modelos**, aunque se recomienda revisar los puntos de atención mencionados."""
        )
        super().__init__(config, tools=[])  # Explícitamente sin herramientas

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Punto de entrada principal para la validación."""
        context = context or {}
        synthetic_data = context.get("synthetic_data")
        original_data = context.get("dataframe")

        if synthetic_data is None or original_data is None:
            return {"message": "Error: Se necesitan tanto datos originales como sintéticos para la validación.", "agent": self.name, "error": True}

        try:
            is_covid = context.get('universal_analysis', {}).get('dataset_type') == 'COVID-19'
            validation_results = self._perform_medical_validations(synthetic_data, is_covid)

            # Crear el prompt para el LLM con los resultados
            prompt = self._create_llm_prompt(validation_results)

            # Obtener el informe del LLM
            informe_markdown = await self.agent_executor.ainvoke({"input": prompt, "chat_history": self.memory.chat_memory.messages})

            return {
                "message": informe_markdown.content,
                "agent": self.name,
                "validation_results": validation_results
            }
        except Exception as e:
            return {"message": f"Error durante la validación: {e}", "agent": self.name, "error": True}

    def _create_llm_prompt(self, results: Dict[str, Any]) -> str:
        """Crea el prompt para el LLM a partir de los resultados de la validación."""
        issues_list = "\n- ".join(results.get('issues', ["No se detectaron issues críticos."]))
        prompt = f"""Resultados de la validación:
        - overall_score: {results.get('overall_score', 0):.2f}
        - clinical_coherence: {results.get('clinical_coherence', 0):.2f}
        - data_quality: {results.get('data_quality', 0):.2f}
        - issues_list: "{issues_list}"

Por favor, genera el informe en Markdown basado en estos datos."""
        return prompt

    def _perform_medical_validations(self, synthetic_data: pd.DataFrame, is_covid_dataset: bool) -> Dict[str, Any]:
        """Realiza validaciones médicas específicas y devuelve un diccionario de resultados."""
        results = {"issues": []}
        
        # 1. Calidad de Datos (Esquema)
        schema_errors = sum(1 for _, row in synthetic_data.iterrows() if not self._validate_row_schema(row))
        total_records = len(synthetic_data)
        results['data_quality'] = max(0.0, (total_records - schema_errors) / total_records) if total_records > 0 else 0.0
        if schema_errors > 0:
            results['issues'].append(f"{schema_errors} registros ({results['data_quality']:.1%}) no cumplen con el esquema JSON esperado.")

        # 2. Coherencia Clínica (Reglas)
        if is_covid_dataset:
            # Lógica específica de COVID
            temp_valid = synthetic_data['temperature'].between(35.0, 42.0).mean()
            sat_valid = synthetic_data['oxygen_saturation'].between(70, 100).mean()
            results['clinical_coherence'] = np.mean([temp_valid, sat_valid])
            if results['clinical_coherence'] < 0.9:
                results['issues'].append("Algunos signos vitales en el dataset de COVID están fuera de rangos plausibles.")
        else:
            # Lógica general
            age_valid = synthetic_data['age'].between(0, 120).mean()
            results['clinical_coherence'] = age_valid
            if results['clinical_coherence'] < 0.95:
                results['issues'].append("Se detectaron edades fuera del rango plausible (0-120 años).")

        # 3. Score General
        results['overall_score'] = np.mean([results['data_quality'], results['clinical_coherence']])
        return results

    def _validate_row_schema(self, row: pd.Series) -> bool:
        """Valida una única fila contra el esquema JSON."""
        try:
            record = row.to_dict()
            # Convertir NaNs a None para validación JSON
            clean_record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
            validate_json(clean_record, pacient_schema)
            return True
        except Exception:
            return False
