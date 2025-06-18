from typing import Dict, Any
from .base_agent import BaseLLMAgent, BaseAgentConfig

class CoordinatorAgent(BaseLLMAgent):
    """Agente coordinador principal que maneja el flujo de conversación"""
    
    def __init__(self):
        config = BaseAgentConfig(
            name="Coordinador Principal",
            description="Agente coordinador que guía el flujo de generación de datos sintéticos",
            system_prompt="""Eres el agente coordinador principal del sistema PATIENTIA para generación de datos clínicos sintéticos. Tu misión es:

1. **Guiar la conversación de manera natural**:
   - Saludar usuarios y explicar capacidades del sistema
   - Identificar necesidades específicas de generación de datos
   - Dirigir el flujo hacia el agente especializado apropiado
   - Mantener contexto conversacional coherente

2. **Orquestar el pipeline de agentes**:
   - Análisis de datos → Generación sintética → Validación médica → Simulación temporal → Evaluación final
   - Decidir cuándo transferir control a cada agente especializado
   - Interpretar resultados de cada etapa para el usuario
   - Coordinar handoffs fluidos entre agentes

3. **Proporcionar orientación experta**:
   - Explicar cada etapa del proceso de manera comprensible
   - Traducir resultados técnicos a lenguaje accesible
   - Recomendar mejores prácticas para generación de datos sintéticos
   - Alertar sobre limitaciones y consideraciones importantes

4. **Manejo de contexto y estado**:
   - Recordar preferencias y decisiones del usuario
   - Mantener seguimiento del progreso en el pipeline
   - Adaptarse a diferentes tipos de usuarios (investigadores, clínicos, estudiantes)
   - Personalizar explicaciones según nivel de expertise

5. **Especialización en dominio médico**:
   - Entender terminología y conceptos clínicos
   - Reconocer tipos de datos médicos y sus características
   - Explicar importancia de validación médica en datos sintéticos
   - Contextualizar resultados en marco de investigación clínica

6. **Gestión de errores y excepciones**:
   - Manejar problemas en cualquier etapa del pipeline
   - Proponer soluciones alternativas cuando algo falle
   - Explicar limitaciones de manera constructiva
   - Mantener experiencia de usuario positiva ante dificultades

Comunícate de manera profesional pero accesible, siendo siempre útil y orientado a objetivos. Personaliza tu comunicación según el contexto y expertise del usuario.""",
            temperature=0.3  # Algo de creatividad para conversación natural
        )
        
        super().__init__(config)
    
    async def handle_greeting(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Maneja saludos iniciales y presentación del sistema"""
        
        prompt = f"""El usuario ha iniciado una conversación: "{user_input}"

Por favor proporciona:
1. Saludo cordial y profesional
2. Presentación clara de PATIENTIA y sus capacidades
3. Explicación del pipeline de generación de datos sintéticos (5 agentes especializados)
4. Orientación sobre cómo comenzar (subir dataset, usar datos de ejemplo, etc.)
5. Pregunta para entender necesidades específicas del usuario

Contexto del sistema: {context}

Mantén un tono profesional pero accesible, enfocándote en la utilidad práctica del sistema."""

        return await self.process(prompt, context)
    
    async def route_to_specialist(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Determina qué agente especializado debe manejar la solicitud"""
        
        prompt = f"""El usuario solicita: "{user_input}"

Contexto actual: {context}

Analiza la solicitud y determina:
1. ¿Qué agente especializado debe manejar esto?
   - Analista: Para análisis de datasets, extracción de patrones
   - Generador: Para crear datos sintéticos, configurar parámetros
   - Validador: Para verificar coherencia médica, validación clínica
   - Simulador: Para evolución temporal, múltiples visitas
   - Evaluador: Para métricas de calidad, utilidad final

2. ¿Qué información contextual necesita el agente especializado?
3. ¿Hay algún prerequisito que falta (ej: dataset no subido)?
4. ¿Cómo preparar al usuario para la siguiente etapa?

Proporciona una respuesta que guíe naturalmente hacia la acción apropiada."""

        return await self.process(prompt, context)
    
    async def interpret_results(self, results: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Interpreta y explica resultados de agentes especializados"""
        
        prompt = f"""Un agente especializado ha completado su trabajo con estos resultados:

{results}

Contexto del pipeline: {context}

Por favor:
1. Interpreta los resultados en términos comprensibles
2. Destaca los puntos más importantes y relevantes
3. Explica qué significan para el objetivo final del usuario
4. Indica claramente cuál es el siguiente paso recomendado
5. Identifica si hay algún problema que requiere atención

Traduce la información técnica a insights accionables, manteniendo precisión científica pero con claridad comunicativa."""

        return await self.process(prompt, context)
    
    async def handle_error(self, error: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Maneja errores de manera constructiva"""
        
        prompt = f"""Se ha producido un error en el sistema: {error}

Contexto cuando ocurrió: {context}

Por favor:
1. Explica el problema de manera comprensible (sin jerga técnica excesiva)
2. Sugiere posibles causas y soluciones
3. Ofrece alternativas para continuar el proceso
4. Mantén una actitud positiva y constructiva
5. Recomienda pasos específicos para resolver o evitar el problema

No te enfoques en el error sino en cómo seguir adelante de manera productiva."""

        return await self.process(prompt, context)
    
    async def provide_guidance(self, topic: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Proporciona orientación sobre temas específicos"""
        
        prompt = f"""El usuario necesita orientación sobre: "{topic}"

Contexto de la conversación: {context}

Proporciona orientación experta que incluya:
1. Explicación clara del concepto o proceso
2. Mejores prácticas relevantes
3. Consideraciones importantes a tener en cuenta
4. Ejemplos específicos si es apropiado
5. Pasos recomendados para proceder

Enfócate en ser práctico y accionable, adaptando el nivel de detalle técnico al contexto del usuario."""

        return await self.process(prompt, context)