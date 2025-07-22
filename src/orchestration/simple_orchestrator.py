"""
Orquestador Simple - Fallback para LangGraph (ya no necesario - LangGraph funciona correctamente)
"""

import pandas as pd
import logging
from typing import Dict, Any, List

# Configurar logger
logger = logging.getLogger(__name__)

class SimpleOrchestrator:
    """Orquestador simple que reemplaza temporalmente LangGraph"""
    
    def __init__(self, agents: Dict[str, Any] = None):
        self.agents = agents or {}
        print("🏗️ Simple Orchestrator inicializado (fallback - LangGraph ya funciona correctamente)")

    async def process_user_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Procesa la entrada del usuario de forma simple"""
        context = context or {}
        
        try:
            # Si hay coordinador, usarlo
            if "coordinator" in self.agents:
                coordinator = self.agents["coordinator"]
                result = await coordinator.safe_process(user_input, context)
                return result
            else:
                # Respuesta genérica sin coordinador
                return {
                    "success": True,
                    "result": f"🤖 **Orquestador Simple**\n\nHe recibido tu solicitud: *\"{user_input[:100]}...\"*\n\n📋 **Estado:**\n• Agentes disponibles: {list(self.agents.keys())}\n• Modo: Fallback funcional\n• Proveedor LLM: Activo\n\n*Nota: LangGraph ya funciona correctamente. Este orquestador simple es solo un fallback.*",
                    "metadata": {
                        "orchestrator": "simple",
                        "agents_available": list(self.agents.keys()),
                        "context": context
                    }
                }
        except Exception as e:
            logger.error(f"Error en orquestador simple: {e}")
            return {
                "success": False,
                "result": f"❌ Error en procesamiento: {e}",
                "metadata": {
                    "orchestrator": "simple",
                    "error": str(e)
                }
            }

    async def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Método de compatibilidad con interface existente"""
        user_input = state.get("user_input", "")
        context = state.get("context", {})
        
        result = await self.process_user_input(user_input, context)
        
        return {
            "coordinator_response": result,
            "messages": [{"content": result.get("result", "Sin respuesta"), "agent": "simple_orchestrator"}]
        }

# Alias para compatibilidad
MedicalAgentsOrchestrator = SimpleOrchestrator
