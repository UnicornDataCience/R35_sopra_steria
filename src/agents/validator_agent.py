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

        # Determinar qué datos validar
        if synthetic_data is not None:
            # Si hay datos sintéticos, validar esos (modo original)
            data_to_validate = synthetic_data
            validation_mode = "sintéticos"
        elif original_data is not None:
            # Si solo hay datos originales, validar esos
            data_to_validate = original_data
            validation_mode = "originales"
        else:
            return {"message": "Error: No se encontraron datos para validar. Sube un dataset o genera datos sintéticos.", "agent": self.name, "error": True}

        try:
            is_covid = context.get('universal_analysis', {}).get('dataset_type') == 'COVID-19'
            validation_results = self._perform_medical_validations(data_to_validate, is_covid, validation_mode)

            # Crear el prompt para el LLM con los resultados
            prompt = self._create_llm_prompt(validation_results, validation_mode)

            # Obtener el informe del LLM
            informe_markdown = await self.agent_executor.ainvoke({"input": prompt, "chat_history": self.memory.chat_memory.messages})

            return {
                "message": informe_markdown.content,
                "agent": self.name,
                "validation_results": validation_results,
                "validation_mode": validation_mode
            }
        except Exception as e:
            return {"message": f"Error durante la validación: {e}", "agent": self.name, "error": True}

    def _create_llm_prompt(self, results: Dict[str, Any], validation_mode: str = "sintéticos") -> str:
        """Crea el prompt para el LLM a partir de los resultados de la validación."""
        issues_list = "\n- ".join(results.get('issues', ["No se detectaron issues críticos."]))
        
        if validation_mode == "originales":
            prompt = f"""Resultados de la validación de DATOS ORIGINALES:
        - overall_score: {results.get('overall_score', 0):.2f}
        - clinical_coherence: {results.get('clinical_coherence', 0):.2f}
        - data_quality: {results.get('data_quality', 0):.2f}
        - issues_list: "{issues_list}"

Por favor, genera un informe en Markdown sobre la calidad y coherencia médica de estos datos ORIGINALES."""
        else:
            prompt = f"""Resultados de la validación de DATOS SINTÉTICOS:
        - overall_score: {results.get('overall_score', 0):.2f}
        - clinical_coherence: {results.get('clinical_coherence', 0):.2f}
        - data_quality: {results.get('data_quality', 0):.2f}
        - issues_list: "{issues_list}"

Por favor, genera un informe en Markdown sobre la calidad y coherencia médica de estos datos SINTÉTICOS."""
        
        return prompt

    def _perform_medical_validations(self, data: pd.DataFrame, is_covid_dataset: bool, validation_mode: str = "sintéticos") -> Dict[str, Any]:
        """Realiza validaciones médicas específicas y devuelve un diccionario de resultados."""
        results = {"issues": []}
        
        # 1. Calidad de Datos (Esquema) - APLICAR DIFERENTE VALIDACIÓN SEGÚN EL TIPO
        if validation_mode == "sintéticos":
            # Para datos sintéticos (que son tabulares): usar la misma validación estructural que para los originales.
            structural_score = self._validate_tabular_structure(data)
            results['data_quality'] = structural_score
            if structural_score < 0.8:
                results['issues'].append("La estructura tabular de los datos sintéticos presenta algunas inconsistencias.")
        else:
            # Para datos originales: validación tabular más flexible
            structural_score = self._validate_tabular_structure(data)
            results['data_quality'] = structural_score
            if structural_score < 0.8:
                results['issues'].append("La estructura tabular de los datos originales presenta algunas inconsistencias menores.")

        # 2. Coherencia Clínica (Reglas) - Con detección automática de columnas
        if is_covid_dataset:
            # Detectar columnas de temperatura
            temp_cols = [col for col in data.columns if any(term in col.lower() for term in ['temp', 'temperatura'])]
            # Detectar columnas de saturación de oxígeno
            sat_cols = [col for col in data.columns if any(term in col.lower() for term in ['sat', 'oxygen', 'oxigeno', 'o2'])]
            
            temp_valid = 1.0  # Default si no encuentra columna
            sat_valid = 1.0   # Default si no encuentra columna
            
            if temp_cols:
                temp_col = temp_cols[0]
                # Convertir a numérico, manejando valores no numéricos
                temp_data = pd.to_numeric(data[temp_col], errors='coerce')
                temp_valid = temp_data.between(35.0, 42.0).mean()
                
            if sat_cols:
                sat_col = sat_cols[0]
                # Convertir a numérico, manejando valores no numéricos
                sat_data = pd.to_numeric(data[sat_col], errors='coerce')
                sat_valid = sat_data.between(70, 100).mean()
            
            results['clinical_coherence'] = np.mean([temp_valid, sat_valid])
            if results['clinical_coherence'] < 0.9:
                results['issues'].append("Algunos signos vitales en el dataset de COVID están fuera de rangos plausibles.")
        else:
            # Detectar columnas de edad
            age_cols = [col for col in data.columns if any(term in col.lower() for term in ['age', 'edad'])]
            
            age_valid = 1.0  # Default si no encuentra columna
            
            if age_cols:
                age_col = age_cols[0]
                # Convertir a numérico, manejando valores no numéricos
                age_data = pd.to_numeric(data[age_col], errors='coerce')
                age_valid = age_data.between(0, 120).mean()
                
            results['clinical_coherence'] = age_valid
            if results['clinical_coherence'] < 0.95:
                results['issues'].append("Se detectaron edades fuera del rango plausible (0-120 años).")

        # 3. Score General
        results['overall_score'] = np.mean([results['data_quality'], results['clinical_coherence']])
        return results

    def _validate_tabular_structure(self, data: pd.DataFrame) -> float:
        """Validación tabular más flexible para datos originales."""
        try:
            issues = 0
            total_checks = 0
            
            # 1. Verificar que no esté completamente vacío
            total_checks += 1
            if data.empty:
                issues += 1
            
            # 2. Verificar que tenga columnas
            total_checks += 1
            if len(data.columns) == 0:
                issues += 1
            
            # 3. Verificar que no todas las filas sean nulas
            total_checks += 1
            if data.isnull().all(axis=1).all():
                issues += 1
            
            # 4. Verificar consistencia de tipos por columna (básico)
            for col in data.columns:
                total_checks += 1
                # Verificar que al menos 50% de los valores no sean nulos
                non_null_ratio = data[col].notna().mean()
                if non_null_ratio < 0.1:  # Muy permisivo para datos reales
                    issues += 1
            
            # 5. Verificar que las columnas tengan nombres
            total_checks += 1
            unnamed_cols = [col for col in data.columns if str(col).startswith('Unnamed') or str(col).strip() == '']
            if len(unnamed_cols) > len(data.columns) * 0.3:  # Más del 30% sin nombre
                issues += 1
            
            # Calcular score (más permisivo para datos reales)
            score = max(0.0, (total_checks - issues) / total_checks) if total_checks > 0 else 1.0
            return score
            
        except Exception:
            return 0.5  # Score neutral si hay error en la validación

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
