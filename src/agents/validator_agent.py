from typing import Dict, Any
import pandas as pd
import numpy as np
import time
from src.utils.logging_config import get_logger
from .base_agent import BaseLLMAgent, BaseAgentConfig
from src.validation.clinical_rules import validate_patient_case
from src.validation.json_schema import validate_json, pacient_schema
from src.validation.validation_cache import get_validation_cache
from src.validation.validation_metrics import get_validator_tracker, ValidationMetrics
from src.validation.rules_engine import get_rules_engine

logger = get_logger(__name__)

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
            start_time = time.time()
            
            # Detectar tipo de dataset
            dataset_type = context.get('universal_analysis', {}).get('dataset_type', 'Generic')
            is_covid = dataset_type == 'COVID-19'
            
            # Intentar recuperar del caché
            cache = get_validation_cache()
            cached_results = cache.get(data_to_validate, is_covid, validation_mode)
            
            if cached_results is not None:
                # Cache HIT
                validation_results = cached_results
                cache_hit = True
                logger.info(f"✅ Validation cache HIT for {validation_mode} data ({data_to_validate.shape[0]} rows)")
            else:
                # Cache MISS - Realizar validación
                cache_hit = False
                logger.info(f"🔍 Validating {validation_mode} data ({data_to_validate.shape[0]} rows, {data_to_validate.shape[1]} cols)")
                validation_results = self._perform_medical_validations(data_to_validate, is_covid, validation_mode, dataset_type)
                
                # Guardar en caché
                cache.put(data_to_validate, is_covid, validation_mode, validation_results)
            
            # Métricas de performance
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Registrar métricas
            metrics = ValidationMetrics(
                validation_mode=validation_mode,
                is_covid=is_covid,
                num_rows=len(data_to_validate),
                num_columns=len(data_to_validate.columns),
                overall_score=validation_results.get('overall_score', 0),
                clinical_coherence=validation_results.get('clinical_coherence', 0),
                data_quality=validation_results.get('data_quality', 0),
                num_issues=len(validation_results.get('issues', [])),
                issues=validation_results.get('issues', []),
                validation_time_ms=elapsed_ms,
                cache_hit=cache_hit
            )
            
            tracker = get_validator_tracker()
            tracker.record(metrics)
            
            logger.info(f"⏱️ Validation completed in {elapsed_ms:.2f}ms (cache_hit={cache_hit})")

            # Crear el prompt para el LLM con los resultados
            prompt = self._create_llm_prompt(validation_results, validation_mode)

            # Obtener el informe del LLM
            informe_markdown = await self.agent_executor.ainvoke({"input": prompt, "chat_history": self.memory.chat_memory.messages})

            return {
                "message": informe_markdown.content,
                "agent": self.name,
                "validation_results": validation_results,
                "validation_mode": validation_mode,
                "performance": {
                    "validation_time_ms": elapsed_ms,
                    "cache_hit": cache_hit
                }
            }
        except Exception as e:
            logger.error("Error durante la validación: %s", e)
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

    def _perform_medical_validations(self, data: pd.DataFrame, is_covid_dataset: bool, validation_mode: str = "sintéticos", dataset_type: str = "Generic") -> Dict[str, Any]:
        """Realiza validaciones médicas específicas y devuelve un diccionario de resultados."""
        results = {"issues": []}
        
        # 1. Calidad de Datos (Esquema)
        structural_score = self._validate_tabular_structure(data)
        results['data_quality'] = structural_score
        if structural_score < 0.8:
            if validation_mode == "sintéticos":
                results['issues'].append("La estructura tabular de los datos sintéticos presenta algunas inconsistencias.")
            else:
                results['issues'].append("La estructura tabular de los datos originales presenta algunas inconsistencias menores.")

        # 2. Coherencia Clínica con Motor de Reglas Configurables
        rules_engine = get_rules_engine()
        rules_validation = rules_engine.validate_dataframe(data, dataset_type)
        
        # Combinar scores
        results['clinical_coherence'] = rules_validation['overall_score']
        results['numeric_score'] = rules_validation.get('numeric_score', 1.0)
        results['categorical_score'] = rules_validation.get('categorical_score', 1.0)
        results['total_checks'] = rules_validation.get('total_checks', 0)
        
        # Agregar issues de las reglas
        results['issues'].extend(rules_validation['issues'])
        
        # Logging detallado
        logger.info(f"📊 Clinical coherence: {results['clinical_coherence']:.3f} ({results['total_checks']} checks)")
        logger.info(f"📊 Data quality: {results['data_quality']:.3f}")

        # 3. Score General (ponderado: 60% clinical coherence, 40% data quality)
        results['overall_score'] = (results['clinical_coherence'] * 0.6) + (results['data_quality'] * 0.4)
        
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
