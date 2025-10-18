"""
Agente Coordinador - Punto de entrada principal para el sistema de agentes médicos.
"""

import json
from typing import Dict, Any
from .base_agent import BaseLLMAgent, BaseAgentConfig
from src.agents.schemas import CoordinatorDecision
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

COORDINATOR_SYSTEM_PROMPT = """Eres el Coordinador de un sistema de IA para un hospital virtual. Tu rol es doble:

1.  **Asistente Médico de IA Conversacional**: Si el usuario hace una pregunta médica, saluda o conversa, responde de manera útil y amigable. 
    - IMPORTANTE: Si recibes contexto del dataset activo (registros, columnas, análisis previo), ÚSALO en tus respuestas para dar información REAL y ESPECÍFICA del dataset.
    - Cuando te pregunten sobre datos, columnas, estadísticas o contenido del dataset, consulta el CONTEXTO DEL DATASET proporcionado y responde con datos REALES.
    - 🔥 NUEVO: Si hay RESULTADOS DE OPERACIONES RECIENTES (análisis, generación, validación, evaluación, simulación), úsalos para responder preguntas sobre esas operaciones.
    - ⚠️ NUNCA menciones IDs técnicos internos (como dataset_id, task_id, UUIDs) en tus respuestas. El usuario no necesita verlos.
    - Refiere al dataset como "el dataset actual", "el dataset analizado", "los datos cargados", etc. - NO uses IDs técnicos.
    - Ejemplo: Si preguntan "¿cuántas columnas tiene?" → responde con el número exacto del contexto, NO con respuestas genéricas.
    - Ejemplo: Si preguntan "¿qué modelo usaste para generar?" → consulta el contexto de generación reciente y responde con el modelo exacto usado.
    - Ejemplo: Si preguntan "¿cuál fue el resultado de la validación?" → usa el contexto de validación reciente.
2.  **Orquestador de Tareas Inteligente**: Si el usuario da un comando para una tarea específica, tu trabajo es identificar la intención, extraer los parámetros y delegar al agente correcto.

**AGENTES DISPONIBLES**:
-   `analyzer`: Para analizar un dataset. Se activa con "analizar", "explorar", "revisar".
-   `generator`: Para crear datos sintéticos. Se activa con "generar", "crear", "sintetizar".
-   `validator`: Para comprobar la coherencia médica. Se activa con "validar", "verificar".
-   `simulator`: Para simular la evolución de pacientes. Se activa with "simular", "evolucionar".
-   `evaluator`: Para medir la calidad de los datos. Se activa con "evaluar", "calidad", "métricas".

**DETECCIÓN DE INTENCIONES**:
-   Si el input es una pregunta sobre temas de salud o medicina, `intention` debe ser `conversacion` y `is_medical_query` debe ser `true`.
-   Si el input contiene saludos, agradecimientos o conversación no médica, `intention` debe ser `conversacion` y `is_medical_query` debe ser `false`.
-   Si el input contiene comandos de acción específicos para los agentes, `intention` debe ser `comando`.

**FORMATO DE RESPUESTA OBLIGATORIO**:
IMPORTANTE: Tu salida DEBE ser SIEMPRE un JSON válido con esta estructura exacta. NO agregues texto adicional antes o después del JSON.

```json
{{
    "intention": "conversacion" | "comando",
    "agent": "analyzer" | "generator" | "validator" | "simulator" | "evaluator" | "coordinator",
    "is_medical_query": true | false,
    "parameters": {{}},
    "message": "tu respuesta completa aquí"
}}
```

**Ejemplos exactos**:
Para "hola":
```json
{{"intention": "conversacion", "agent": "coordinator", "is_medical_query": false, "parameters": {{}}, "message": "¡Hola! Soy tu asistente de IA médica. ¿En qué puedo ayudarte?"}}
```

Para "¿cuáles son los síntomas de la diabetes?":
```json
{{"intention": "conversacion", "agent": "coordinator", "is_medical_query": true, "parameters": {{}}, "message": "Los síntomas comunes de la diabetes incluyen aumento de la sed, micción frecuente, hambre extrema, pérdida de peso inexplicable y fatiga. También puede haber visión borrosa, cicatrización lenta de heridas y infecciones frecuentes."}}
```

Para "analizar datos":
```json
{{"intention": "comando", "agent": "analyzer", "is_medical_query": false, "parameters": {{}}, "message": "Iniciando análisis del dataset..."}}
```

Para "generar con CTGAN":
```json
{{"intention": "comando", "agent": "generator", "is_medical_query": false, "parameters": {{"model_type": "ctgan"}}, "message": "Generando datos sintéticos con CTGAN..."}}
```

RECUERDA: Responde ÚNICAMENTE con el JSON válido, sin texto adicional.
"""

class CoordinatorAgentConfig(BaseAgentConfig):
    name: str = "Coordinador"
    description: str = "Agente coordinador que dirige las solicitudes a los agentes especializados."
    system_prompt: str = COORDINATOR_SYSTEM_PROMPT
    temperature: float = 0.0

class CoordinatorAgent(BaseLLMAgent):
    def __init__(self):
        super().__init__(CoordinatorAgentConfig(), tools=[])  # Explícitamente sin herramientas

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        try:
            # Intentar extraer JSON de bloques de código
            json_str = ""
            if '```json' in response:
                start_idx = response.find('```json') + 7
                end_idx = response.find('```', start_idx)
                if end_idx != -1:
                    json_str = response[start_idx:end_idx].strip()
            elif '```' in response:
                start_idx = response.find('```') + 3
                end_idx = response.find('```', start_idx)
                if end_idx != -1:
                    json_str = response[start_idx:end_idx].strip()
            if not json_str:
                json_str = response.strip()
            if json_str.startswith('json'):
                json_str = json_str[4:].strip()
            if not json_str or not json_str.startswith('{'):
                raise ValueError("No se encontró JSON válido en la respuesta")

            parsed = json.loads(json_str)
            # Validar con Pydantic
            decision = CoordinatorDecision(**parsed)
            return decision.model_dump()
        except Exception as e:
            logger.warning("Error parseando JSON del Coordinador: %s", e)
            logger.debug("Respuesta recibida (trim): %s", response[:200])

            # Heurística de recuperación
            response_clean = (response or "").strip()
            medical_keywords = ['síntomas', 'diabetes', 'enfermedad', 'tratamiento', 'medicina', 
                                'salud', 'paciente', 'diagnóstico', 'dolor', 'hospital']
            greeting_keywords = ['hola', 'buenos', 'gracias', 'adiós', 'saludos']
            command_keywords = ['analizar', 'generar', 'validar', 'simular', 'evaluar', 'crear']
            low = response_clean.lower()
            is_medical = any(k in low for k in medical_keywords)
            is_greeting = any(k in low for k in greeting_keywords)
            is_command = any(k in low for k in command_keywords)

            if (is_medical or is_greeting) and not is_command:
                fallback = CoordinatorDecision(
                    intention="conversacion",
                    agent="coordinator",
                    is_medical_query=bool(is_medical),
                    parameters={},
                    message=response_clean or ""
                )
                return fallback.model_dump()
            elif is_command:
                fallback = CoordinatorDecision(
                    intention="comando",
                    agent="analyzer",
                    is_medical_query=False,
                    parameters={},
                    message="Procesando comando..."
                )
                return fallback.model_dump()
            else:
                fallback = CoordinatorDecision(
                    intention="conversacion",
                    agent="coordinator",
                    is_medical_query=False,
                    parameters={},
                    message=response_clean or "Lo siento, no pude entender tu solicitud. ¿Puedes reformularla?"
                )
                return fallback.model_dump()

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        prompt = f"{input_text}"
        
        # NUEVO: Agregar información del dataset al prompt si está disponible
        if context and context.get("has_dataset"):
            dataset_info = context.get("dataset_info", {})
            columns = context.get("columns", [])
            stats_summary = context.get("statistics_summary", {})
            recent_results = context.get("recent_results", {})
            active_operation = context.get("active_operation", "none")
            
            dataset_context = f"\n\n## CONTEXTO DEL DATASET ACTIVO:\n"
            dataset_context += f"- Total de registros/pacientes: {dataset_info.get('total_rows', 'desconocido')}\n"
            dataset_context += f"- Total de columnas: {dataset_info.get('total_columns', len(columns) if columns else 'desconocido')}\n"
            
            if columns:
                col_list_items = list(columns)[:15]  # Convertir a lista y tomar primeras 15
                col_list = ', '.join(str(c) for c in col_list_items)
                if len(columns) > 15:
                    col_list += f" (y {len(columns)-15} más)"
                dataset_context += f"- Columnas disponibles ({len(columns)}): {col_list}\n"
            
            if dataset_info.get('average_age'):
                dataset_context += f"- Edad promedio pacientes: {dataset_info.get('average_age')} años\n"
            
            if dataset_info.get('null_count') is not None:
                dataset_context += f"- Total valores nulos: {dataset_info.get('null_count')}\n"
            
            if dataset_info.get('selected_target'):
                dataset_context += f"- Variable objetivo (target): {dataset_info.get('selected_target')}\n"
            
            # Añadir información estadística adicional
            if stats_summary:
                if stats_summary.get('numerical_columns'):
                    dataset_context += f"- Columnas numéricas: {len(stats_summary['numerical_columns'])}\n"
                if stats_summary.get('categorical_columns'):
                    dataset_context += f"- Columnas categóricas: {len(stats_summary['categorical_columns'])}\n"
            
            # 🔥 NUEVO: Incluir resultados de TODOS los agentes ejecutados
            # Ajustar tamaño según tamaño del dataset (evitar exceder límites de tokens)
            total_cols = len(columns) if columns else 0
            is_large_dataset = (dataset_info.get('total_rows', 0) > 1000) or (total_cols > 50)
            max_summary_chars = 400 if is_large_dataset else 800
            
            if recent_results:
                dataset_context += f"\n## RESULTADOS DE OPERACIONES RECIENTES:\n"
                dataset_context += f"- Operación activa más reciente: {active_operation}\n\n"
                
                # Análisis
                if recent_results.get('analysis', {}).get('has_result'):
                    analysis_summary = recent_results['analysis'].get('summary', '')[:max_summary_chars]
                    dataset_context += f"### 📊 ANÁLISIS:\n{analysis_summary}...\n\n"
                
                # Generación
                if recent_results.get('generation', {}).get('has_result'):
                    gen_summary = recent_results['generation'].get('summary', '')[:max_summary_chars]
                    gen_meta = recent_results['generation'].get('meta', {})
                    dataset_context += f"### 🎲 GENERACIÓN:\n{gen_summary}...\n"
                    if gen_meta and not is_large_dataset:
                        dataset_context += f"Meta: {json.dumps(gen_meta, ensure_ascii=False)[:150]}\n\n"
                
                # Validación
                if recent_results.get('validation', {}).get('has_result'):
                    val_summary = recent_results['validation'].get('summary', '')[:max_summary_chars]
                    dataset_context += f"### ✅ VALIDACIÓN:\n{val_summary}...\n\n"
                
                # Evaluación (solo si no es dataset grande)
                if not is_large_dataset and recent_results.get('evaluation', {}).get('has_result'):
                    eval_summary = recent_results['evaluation'].get('summary', '')[:max_summary_chars]
                    dataset_context += f"### 📈 EVALUACIÓN:\n{eval_summary}...\n\n"
                
                # Simulación (solo si no es dataset grande)
                if not is_large_dataset and recent_results.get('simulation', {}).get('has_result'):
                    sim_summary = recent_results['simulation'].get('summary', '')[:max_summary_chars]
                    dataset_context += f"### 🧬 SIMULACIÓN:\n{sim_summary}...\n\n"
            
            prompt += dataset_context
            logger.info(f"✅ Prompt enriquecido: {dataset_info.get('total_rows')} registros, {len(columns) if columns else 0} columnas, target='{dataset_info.get('selected_target')}', operaciones={list(recent_results.keys()) if recent_results else []}")
        
        # Mantener compatibilidad con contexto antiguo
        elif context and context.get("dataset_uploaded"):
            prompt += f"\n\nContexto Adicional: Ya hay un dataset cargado llamado '{context.get('filename')}'."
        
        if context and context.get("parameters"):
            params = context["parameters"]
            if params.get("model_type"):
                prompt += f"\n\nModelo solicitado: {params['model_type'].upper()}"
            if params.get("num_samples"):
                prompt += f"\nNúmero de muestras: {params['num_samples']}"

        llm_response = await self.agent_executor.ainvoke({"input": prompt, "chat_history": self.memory.chat_memory.messages})
        response_text = llm_response.content if hasattr(llm_response, 'content') else (llm_response if isinstance(llm_response, str) else str(llm_response))
        parsed_response = self._parse_llm_response(response_text)
        if context and context.get("parameters"):
            # No pisar parámetros si ya existen
            parsed_response["parameters"] = {**parsed_response.get("parameters", {}), **context["parameters"]}
        return parsed_response