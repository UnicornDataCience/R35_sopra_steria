"""
Agente Coordinador - Punto de entrada principal para el sistema de agentes médicos.
"""

import json
from typing import Dict, Any
from .base_agent import BaseLLMAgent, BaseAgentConfig

COORDINATOR_SYSTEM_PROMPT = """Eres el Coordinador de un sistema de IA para un hospital virtual. Tu rol es doble:

1.  **Asistente Médico de IA Conversacional**: Si el usuario hace una pregunta médica, saluda o conversa, responde de manera útil y amigable.
2.  **Orquestador de Tareas Inteligente**: Si el usuario da un comando para una tarea específica, tu trabajo es identificar la intención, extraer los parámetros y delegar al agente correcto.

**AGENTES DISPONIBLES**:
-   `analyzer`: Para analizar un dataset. Se activa con "analizar", "explorar", "revisar".
-   `generator`: Para crear datos sintéticos. Se activa con "generar", "crear", "sintetizar".
-   `validator`: Para comprobar la coherencia médica. Se activa con "validar", "verificar".
-   `simulator`: Para simular la evolución de pacientes. Se activa con "simular", "evolucionar".
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
            # MÉTODO 1: Intentar extraer JSON de un bloque de código
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
            
            # MÉTODO 2: Si no se extrajo JSON del bloque, buscar directamente
            if not json_str:
                json_str = response.strip()
            
            # MÉTODO 3: Limpiar prefijos adicionales
            if json_str.startswith('json'):
                json_str = json_str[4:].strip()
            
            # MÉTODO 4: Verificar que tenemos JSON válido antes de parsear
            if not json_str or not json_str.startswith('{'):
                raise ValueError("No se encontró JSON válido en la respuesta")
            
            # MÉTODO 5: Intentar parsear el JSON
            parsed_json = json.loads(json_str)
            
            # MÉTODO 6: Decodificar caracteres Unicode si es necesario
            if isinstance(parsed_json.get('message'), str):
                try:
                    # Decodificar secuencias Unicode como \u00a1 -> ¡
                    message = parsed_json['message']
                    # Usar json.loads para decodificar secuencias unicode automáticamente
                    try:
                        # Envolver en comillas para hacer un string JSON válido
                        decoded_message = json.loads(f'"{message}"')
                        parsed_json['message'] = decoded_message
                    except:
                        # Si falla, intentar decodificación manual
                        parsed_json['message'] = message.encode().decode('unicode_escape')
                except:
                    # Si falla la decodificación, mantener el original
                    pass
            
            return parsed_json
            
        except Exception as e:
            print(f"⚠️ Error parseando JSON del Coordinador: {e}")
            print(f"📝 Respuesta recibida: {response[:200]}...")
            
            # DEBUG: Mostrar el JSON extraído para diagnóstico
            if 'json_str' in locals():
                print(f"📝 JSON extraído: {json_str[:100]}...")
            
            # MEJORADO: Intento de recuperación más inteligente
            
            # MEJORADO: Intento de recuperación más inteligente
            response_clean = response.strip()
            
            # Si la respuesta contiene JSON válido pero mal extraído
            if '{"intention"' in response_clean or '{{"intention"' in response_clean:
                # Buscar el JSON dentro de la respuesta más agresivamente
                import re
                json_pattern = r'\{[^{}]*"intention"[^{}]*"message"[^{}]*\}'
                match = re.search(json_pattern, response_clean, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except:
                        pass
            
            # Detectar tipo de respuesta para fallback
            medical_keywords = ['síntomas', 'diabetes', 'enfermedad', 'tratamiento', 'medicina', 
                              'salud', 'paciente', 'diagnóstico', 'dolor', 'hospital']
            greeting_keywords = ['hola', 'buenos', 'gracias', 'adiós', 'saludos']
            command_keywords = ['analizar', 'generar', 'validar', 'simular', 'evaluar', 'crear']
            
            response_lower = response_clean.lower()
            
            # Determinar si es médico
            is_medical = any(keyword in response_lower for keyword in medical_keywords)
            is_greeting = any(keyword in response_lower for keyword in greeting_keywords)
            is_command = any(keyword in response_lower for keyword in command_keywords)
            
            # Si parece una respuesta conversacional médica o saludo
            if (is_medical or is_greeting) and not is_command:
                print(f"🔄 Interpretando como respuesta conversacional médica")
                return {
                    "intention": "conversacion",
                    "agent": "coordinator", 
                    "is_medical_query": is_medical,
                    "parameters": {},
                    "message": response_clean
                }
            
            # Si parece un comando pero no está bien formateado
            elif is_command:
                print(f"🔄 Interpretando como comando mal formateado")
                return {
                    "intention": "comando",
                    "agent": "analyzer",  # Agente por defecto
                    "is_medical_query": False,
                    "parameters": {},
                    "message": "Procesando comando..."
                }
            
            # Fallback general: tratar como conversación
            else:
                print(f"🔄 Fallback: interpretando como conversación general")
                return {
                    "intention": "conversacion",
                    "agent": "coordinator",
                    "is_medical_query": False,
                    "parameters": {},
                    "message": response_clean if response_clean else "Lo siento, no pude entender tu solicitud. ¿Puedes reformularla?"
                }

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        prompt = f"{input_text}"
        if context and context.get("dataset_uploaded"):
            prompt += f"\n\nContexto Adicional: Ya hay un dataset cargado llamado '{context.get('filename')}'."
        
        # Agregar información de parámetros si están presentes
        if context and context.get("parameters"):
            params = context["parameters"]
            if params.get("model_type"):
                prompt += f"\n\nModelo solicitado: {params['model_type'].upper()}"
            if params.get("num_samples"):
                prompt += f"\nNúmero de muestras: {params['num_samples']}"

        llm_response = await self.agent_executor.ainvoke({"input": prompt, "chat_history": self.memory.chat_memory.messages})
        
        # Manejar diferentes tipos de respuesta del LLM
        if hasattr(llm_response, 'content'):
            # Respuesta de AgentExecutor con herramientas
            response_text = llm_response.content
        elif isinstance(llm_response, str):
            # Respuesta directa de LLM simple
            response_text = llm_response
        else:
            # Otros tipos de respuesta
            response_text = str(llm_response)
        
        parsed_response = self._parse_llm_response(response_text)
        
        # Asegurar que los parámetros se mantengan en la respuesta
        if context and context.get("parameters"):
            parsed_response["parameters"] = context["parameters"]
        
        return parsed_response