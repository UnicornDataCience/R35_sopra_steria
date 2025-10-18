"""
Servicio para manejar la lógica de chat y comunicación con agentes
"""
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from api.models.schemas import ChatResponse, AgentType
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class ChatService:
    """Servicio para procesar mensajes de chat y coordinar agentes"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
    
    async def process_message(
        self,
        message: str,
        context: Dict[str, Any],
        preferred_agent: Optional[str] = None
    ) -> ChatResponse:
        """
        Procesa un mensaje del usuario y devuelve la respuesta del agente apropiado
        """
        try:
            logger.info("Procesando mensaje: %s", message[:100])
            
            # Si se especifica un agente preferido, usarlo
            if preferred_agent:
                response = await self._call_specific_agent(preferred_agent, message, context)
            else:
                # Usar el orquestador para determinar el agente apropiado
                response = await self._call_orchestrator(message, context)
            
            # Limpiar y formatear la respuesta
            cleaned_response = self._clean_response(response)
            
            # Generar sugerencias basadas en la respuesta
            suggestions = self._generate_suggestions(message, cleaned_response, context)
            
            return ChatResponse(
                response=cleaned_response.get("message", "No se recibió respuesta"),
                agent=cleaned_response.get("agent", "unknown"),
                context=self._update_context(context, cleaned_response),
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error("Error procesando mensaje: %s", e)
            return ChatResponse(
                response=f"Lo siento, ocurrió un error al procesar tu mensaje: {str(e)}",
                agent="system",
                context=context,
                suggestions=["Intenta reformular tu pregunta", "Verifica tu conexión"]
            )
    
    async def _call_orchestrator(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Llama al orquestador principal"""
        try:
            # Verificar si el orquestador tiene método asíncrono
            if hasattr(self.orchestrator, 'process_user_input'):
                if asyncio.iscoroutinefunction(self.orchestrator.process_user_input):
                    return await self.orchestrator.process_user_input(message, context)
                else:
                    # Si es síncrono, ejecutarlo en un hilo separado
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, 
                        self.orchestrator.process_user_input, 
                        message, 
                        context
                    )
            # Fallback para método síncrono
            elif hasattr(self.orchestrator, 'process_user_input_sync'):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    self.orchestrator.process_user_input_sync,
                    message,
                    context
                )
            else:
                raise Exception("Orquestador no tiene método de procesamiento válido")
                
        except Exception as e:
            logger.error("Error en orquestador: %s", e)
            return {
                "message": f"Error interno del sistema: {str(e)}",
                "agent": "system",
                "error": True
            }
    
    async def _call_specific_agent(self, agent_type: str, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Llama a un agente específico"""
        try:
            # Mapear tipos de agente a métodos específicos
            agent_methods = {
                "coordinator": self._call_coordinator,
                "analyzer": self._call_analyzer,
                "generator": self._call_generator,
                "validator": self._call_validator,
                "simulator": self._call_simulator,
                "evaluator": self._call_evaluator
            }
            
            if agent_type in agent_methods:
                return await agent_methods[agent_type](message, context)
            else:
                # Si no existe el agente específico, usar orquestador
                return await self._call_orchestrator(message, context)
                
        except Exception as e:
            logger.error("Error llamando agente %s: %s", agent_type, e)
            return {
                "message": f"Error en agente {agent_type}: {str(e)}",
                "agent": agent_type,
                "error": True
            }
    
    async def _call_coordinator(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Llama al agente coordinador"""
        if hasattr(self.orchestrator, 'agents') and 'coordinator' in self.orchestrator.agents:
            coordinator = self.orchestrator.agents['coordinator']
            return await self._execute_agent_method(coordinator, message, context)
        return await self._call_orchestrator(message, context)
    
    async def _call_analyzer(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Llama al agente analizador"""
        if hasattr(self.orchestrator, 'agents') and 'analyzer' in self.orchestrator.agents:
            analyzer = self.orchestrator.agents['analyzer']
            return await self._execute_agent_method(analyzer, message, context)
        return await self._call_orchestrator(message, context)
    
    async def _call_generator(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Llama al agente generador"""
        if hasattr(self.orchestrator, 'agents') and 'generator' in self.orchestrator.agents:
            generator = self.orchestrator.agents['generator']
            return await self._execute_agent_method(generator, message, context)
        return await self._call_orchestrator(message, context)
    
    async def _call_validator(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Llama al agente validador"""
        if hasattr(self.orchestrator, 'agents') and 'validator' in self.orchestrator.agents:
            validator = self.orchestrator.agents['validator']
            return await self._execute_agent_method(validator, message, context)
        return await self._call_orchestrator(message, context)
    
    async def _call_simulator(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Llama al agente simulador"""
        if hasattr(self.orchestrator, 'agents') and 'simulator' in self.orchestrator.agents:
            simulator = self.orchestrator.agents['simulator']
            return await self._execute_agent_method(simulator, message, context)
        return await self._call_orchestrator(message, context)
    
    async def _call_evaluator(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Llama al agente evaluador"""
        if hasattr(self.orchestrator, 'agents') and 'evaluator' in self.orchestrator.agents:
            evaluator = self.orchestrator.agents['evaluator']
            return await self._execute_agent_method(evaluator, message, context)
        return await self._call_orchestrator(message, context)
    
    async def _execute_agent_method(self, agent, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta el método de un agente, manejando tanto sync como async"""
        try:
            # Intentar método asíncrono
            if hasattr(agent, 'process') and asyncio.iscoroutinefunction(agent.process):
                return await agent.process(message, context)
            # Intentar método síncrono
            elif hasattr(agent, 'process_sync'):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, agent.process_sync, message, context)
            # Método process síncrono
            elif hasattr(agent, 'process'):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, agent.process, message, context)
            else:
                return {
                    "message": f"Agente {agent.__class__.__name__} no tiene método de procesamiento",
                    "agent": agent.__class__.__name__,
                    "error": True
                }
        except Exception as e:
            logger.error("Error ejecutando agente %s: %s", agent.__class__.__name__, e)
            return {
                "message": f"Error en agente {agent.__class__.__name__}: {str(e)}",
                "agent": agent.__class__.__name__,
                "error": True
            }
    
    def _clean_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Limpia y normaliza la respuesta del agente"""
        if not isinstance(response, dict):
            return {
                "message": str(response),
                "agent": "unknown",
                "error": False
            }
        
        # Limpiar mensaje de caracteres especiales si es necesario
        message = response.get("message", "")
        if isinstance(message, str):
            # Aquí puedes añadir la lógica de limpieza de caracteres Unicode
            # que tenías en clean_response_message del chat_llm.py
            pass
        
        return {
            "message": message,
            "agent": response.get("agent", "unknown"),
            "error": response.get("error", False),
            "payload": response.get("payload"),
            "metadata": response.get("metadata", {})
        }
    
    def _update_context(self, original_context: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        """Actualiza el contexto con información de la respuesta"""
        updated_context = original_context.copy()
        
        # Actualizar con payload si existe
        if response.get("payload"):
            updated_context.update(response["payload"])
        
        # Actualizar timestamp de última interacción
        updated_context["last_interaction"] = datetime.now().isoformat()
        updated_context["last_agent"] = response.get("agent")
        
        return updated_context
    
    def _generate_suggestions(self, original_message: str, response: Dict[str, Any], context: Dict[str, Any]) -> list:
        """Genera sugerencias basadas en el mensaje y respuesta"""
        suggestions = []
        
        # Sugerencias basadas en el agente que respondió
        agent = response.get("agent", "")
        
        if "analyzer" in agent.lower():
            suggestions.extend([
                "Genera datos sintéticos basados en este análisis",
                "¿Qué columnas son más importantes?",
                "Muestra estadísticas detalladas"
            ])
        elif "generator" in agent.lower():
            suggestions.extend([
                "Valida los datos generados",
                "Genera más muestras",
                "Compara con datos originales"
            ])
        elif "coordinator" in agent.lower():
            if context.get("dataset_uploaded"):
                suggestions.extend([
                    "Analiza estos datos",
                    "Genera datos sintéticos",
                    "Valida la calidad"
                ])
            else:
                suggestions.extend([
                    "¿Cómo subo un dataset?",
                    "¿Qué modelos puedo usar?",
                    "Ejemplos de uso"
                ])
        
        return suggestions[:5]  # Limitar a 5 sugerencias
