"""
Agente Coordinador - Punto de entrada principal para el sistema de agentes médicos.
"""

import json
from typing import Dict, Any
from .base_agent import BaseLLMAgent, BaseAgentConfig

COORDINATOR_SYSTEM_PROMPT = """Eres el Coordinador de un sistema de IA para un hospital virtual. Tu rol es doble:

1. Asistente Médico de IA Conversacional: Si el usuario hace una pregunta general, saluda o conversa, responde de manera útil y amigable.

2. Orquestador de Tareas Inteligente: Si el usuario da un comando para una tarea específica, tu trabajo es identificar la intención, extraer los parámetros y delegar al agente correcto.

AGENTES DISPONIBLES:
- analyzer: Para analizar un dataset. Se activa con analizar, explorar, revisar.
- generator: Para crear datos sintéticos. Se activa con generar, crear, sintetizar.
- validator: Para comprobar la coherencia médica. Se activa con validar, verificar.
- simulator: Para simular la evolución de pacientes. Se activa con simular, evolucionar.
- evaluator: Para medir la calidad de los datos. Se activa con evaluar, calidad, métricas.

DETECCIÓN DE INTENCIONES:
- Si el input contiene saludos (hola, buenos días, hi), preguntas generales, agradecimientos o conversación: intention=conversacion
- Si el input contiene comandos de acción específicos: intention=comando

FORMATO DE RESPUESTA:
Tu salida DEBE ser SIEMPRE un JSON válido con esta estructura:
- intention: conversacion o comando
- agent: analyzer, generator, validator, simulator, evaluator o coordinator
- parameters: objeto con parámetros específicos (MANTÉN los parámetros recibidos)
- message: mensaje para el usuario

Ejemplos:
Para "hola": {{"intention": "conversacion", "agent": "coordinator", "message": "¡Hola! Soy tu asistente de IA médica. ¿En qué puedo ayudarte?"}}
Para "analizar datos": {{"intention": "comando", "agent": "analyzer", "message": "Iniciando análisis del dataset..."}}
Para "generar con CTGAN": {{"intention": "comando", "agent": "generator", "parameters": {{"model_type": "ctgan"}}, "message": "Generando datos sintéticos con CTGAN..."}}
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
            # Primero intentar extraer JSON de un bloque de código
            if '```json' in response:
                json_str = response[response.find('```json') + 7:response.rfind('```')]
            elif '```' in response:
                json_str = response[response.find('```') + 3:response.rfind('```')]
            else:
                # Si no hay bloques de código, buscar JSON directamente
                json_str = response.strip()
            
            # Limpiar la cadena
            json_str = json_str.strip()
            if json_str.startswith('json'):
                json_str = json_str[4:].strip()
            
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parseando JSON del Coordinador: {e}\nRespuesta recibida: {response}")
            return {"intention": "error", "agent": "coordinator", "message": "Error interno al interpretar la respuesta."}

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