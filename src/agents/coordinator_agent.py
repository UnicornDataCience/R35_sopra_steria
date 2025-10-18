"""
Agente Coordinador - Punto de entrada principal para el sistema de agentes médicos.

Mejoras Fase 1:
- Caché de respuestas comunes para reducir latencia
- Performance tracking para métricas de clasificación
- Logging estructurado de decisiones
- Métricas de aciertos/fallos de clasificación
"""

import json
import time
import hashlib
from typing import Dict, Any, Optional
from .base_agent import BaseLLMAgent, BaseAgentConfig
from src.agents.schemas import CoordinatorDecision
from src.utils.logging_config import get_logger
from src.utils.optimization_utils import CacheManager, PerformanceTracker

logger = get_logger(__name__)

# 🚀 Fase 1: Caché de respuestas comunes
COMMON_RESPONSES_CACHE = {
    # Saludos
    "hola": {
        "intention": "conversacion",
        "agent": "coordinator",
        "is_medical_query": False,
        "parameters": {},
        "message": "¡Hola! Soy tu asistente de IA médica del Hospital Virtual Patientia. ¿En qué puedo ayudarte hoy?"
    },
    "buenos dias": {
        "intention": "conversacion",
        "agent": "coordinator",
        "is_medical_query": False,
        "parameters": {},
        "message": "¡Buenos días! Soy tu asistente de IA médica. Estoy aquí para ayudarte con análisis de datos, generación de datos sintéticos y simulaciones médicas."
    },
    "buenas tardes": {
        "intention": "conversacion",
        "agent": "coordinator",
        "is_medical_query": False,
        "parameters": {},
        "message": "¡Buenas tardes! ¿En qué puedo asistirte hoy?"
    },
    "gracias": {
        "intention": "conversacion",
        "agent": "coordinator",
        "is_medical_query": False,
        "parameters": {},
        "message": "¡De nada! Estoy aquí para ayudarte. Si necesitas algo más, no dudes en preguntar."
    },
    "adios": {
        "intention": "conversacion",
        "agent": "coordinator",
        "is_medical_query": False,
        "parameters": {},
        "message": "¡Hasta luego! Que tengas un excelente día. Vuelve cuando necesites ayuda."
    },
    # Preguntas frecuentes sobre el sistema
    "que puedes hacer": {
        "intention": "conversacion",
        "agent": "coordinator",
        "is_medical_query": False,
        "parameters": {},
        "message": "Puedo ayudarte con:\n📊 **Análisis** de datasets médicos\n🎲 **Generación** de datos sintéticos (CTGAN, TVAE)\n✅ **Validación** de coherencia médica\n📈 **Evaluación** de calidad de datos\n🧬 **Simulación** de evolución de pacientes\n\n¿Qué te gustaría hacer?"
    },
    "ayuda": {
        "intention": "conversacion",
        "agent": "coordinator",
        "is_medical_query": False,
        "parameters": {},
        "message": "Puedo ayudarte con:\n📊 **Análisis** de datasets médicos\n🎲 **Generación** de datos sintéticos (CTGAN, TVAE)\n✅ **Validación** de coherencia médica\n📈 **Evaluación** de calidad de datos\n🧬 **Simulación** de evolución de pacientes\n\n¿Qué te gustaría hacer?"
    }
}

# 📊 Fase 1: Métricas de clasificación
class CoordinatorMetrics:
    """Métricas de performance del coordinador"""
    def __init__(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.llm_calls = 0
        self.fallback_used = 0
        self.avg_response_time = 0.0
        self._response_times = []
    
    def record_cache_hit(self):
        self.cache_hits += 1
        self.total_requests += 1
    
    def record_cache_miss(self):
        self.cache_misses += 1
        self.llm_calls += 1
        self.total_requests += 1
    
    def record_fallback(self):
        self.fallback_used += 1
    
    def record_response_time(self, time_ms: float):
        self._response_times.append(time_ms)
        if self._response_times:
            self.avg_response_time = sum(self._response_times) / len(self._response_times)
    
    def get_stats(self) -> Dict[str, Any]:
        cache_hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "llm_calls": self.llm_calls,
            "fallback_used": self.fallback_used,
            "avg_response_time_ms": f"{self.avg_response_time:.2f}"
        }
    
    def reset(self):
        """Reset todas las métricas"""
        self.__init__()

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
        
        # 🚀 Fase 1: Inicializar tracking y métricas
        self.metrics = CoordinatorMetrics()
        self.performance_tracker = PerformanceTracker('coordinator')
        logger.info("✅ Coordinador inicializado con métricas y tracking de performance")

    def _check_common_response(self, input_text: str) -> Optional[Dict[str, Any]]:
        """
        🚀 Fase 1: Verificar si el input tiene una respuesta común cacheada
        
        Returns:
            Dict con respuesta si hay cache hit, None si no
        """
        if not input_text:
            return None
        
        # Normalizar input
        normalized = input_text.lower().strip()
        
        # Buscar match exacto
        if normalized in COMMON_RESPONSES_CACHE:
            self.metrics.record_cache_hit()
            logger.info(f"💨 Cache HIT para: '{input_text[:50]}'")
            return COMMON_RESPONSES_CACHE[normalized].copy()
        
        # Buscar match parcial (si el input contiene alguna key)
        for key, response in COMMON_RESPONSES_CACHE.items():
            if key in normalized and len(normalized) < 30:  # Solo para inputs cortos
                self.metrics.record_cache_hit()
                logger.info(f"💨 Cache HIT parcial para: '{input_text[:50]}' (match: '{key}')")
                return response.copy()
        
        return None
    
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
                self.metrics.record_fallback()  # 📊 Fase 1: Tracking de fallback
                logger.warning("⚠️ Usando heurística de recuperación (conversación médica/saludo)")
                fallback = CoordinatorDecision(
                    intention="conversacion",
                    agent="coordinator",
                    is_medical_query=bool(is_medical),
                    parameters={},
                    message=response_clean or ""
                )
                return fallback.model_dump()
            elif is_command:
                self.metrics.record_fallback()  # 📊 Fase 1: Tracking de fallback
                logger.warning("⚠️ Usando heurística de recuperación (comando detectado)")
                fallback = CoordinatorDecision(
                    intention="comando",
                    agent="analyzer",
                    is_medical_query=False,
                    parameters={},
                    message="Procesando comando..."
                )
                return fallback.model_dump()
            else:
                self.metrics.record_fallback()  # 📊 Fase 1: Tracking de fallback
                logger.warning("⚠️ Usando heurística de recuperación (fallback genérico)")
                fallback = CoordinatorDecision(
                    intention="conversacion",
                    agent="coordinator",
                    is_medical_query=False,
                    parameters={},
                    message=response_clean or "Lo siento, no pude entender tu solicitud. ¿Puedes reformularla?"
                )
                return fallback.model_dump()

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Procesa la entrada del usuario y decide qué agente debe manejarla.
        
        Mejoras Fase 1:
        - Caché de respuestas comunes
        - Performance tracking
        - Logging estructurado
        - Métricas de clasificación
        """
        # 📊 Fase 1: Iniciar tracking de performance
        start_time = time.time()
        
        # 🚀 Fase 1: Verificar caché de respuestas comunes
        cached_response = self._check_common_response(input_text)
        if cached_response:
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics.record_response_time(elapsed_ms)
            
            logger.info(f"✅ Respuesta cacheada servida en {elapsed_ms:.2f}ms")
            logger.debug(f"📊 Métricas actuales: {self.metrics.get_stats()}")
            
            return cached_response
        
        # Si no hay cache hit, continuar con procesamiento normal
        self.metrics.record_cache_miss()
        logger.info(f"🔍 Cache MISS - llamando al LLM para: '{input_text[:50]}'")
        
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
        
        # 📊 Fase 1: Registrar métricas
        elapsed_ms = (time.time() - start_time) * 1000
        self.metrics.record_response_time(elapsed_ms)
        
        # 📝 Fase 1: Logging estructurado de la decisión
        logger.info(
            f"✅ Coordinador procesó en {elapsed_ms:.2f}ms | "
            f"Intención: {parsed_response.get('intention')} | "
            f"Agente: {parsed_response.get('agent')} | "
            f"Médica: {parsed_response.get('is_medical_query')}"
        )
        logger.debug(f"📊 Métricas actuales: {self.metrics.get_stats()}")
        
        return parsed_response
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        🚀 Fase 1: Obtener métricas actuales del coordinador
        
        Returns:
            Dict con estadísticas de performance
        """
        stats = self.metrics.get_stats()
        
        return {
            "classification_metrics": stats,
            "status": "healthy" if stats["total_requests"] > 0 else "idle"
        }
    
    def reset_metrics(self):
        """
        🚀 Fase 1: Resetear métricas (útil para testing)
        """
        self.metrics.reset()
        self.performance_tracker = PerformanceTracker('coordinator')
        logger.info("🔄 Métricas del coordinador reseteadas")