"""
Router para el chat y conversación con agentes
"""
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from typing import Dict, Any
import uuid

from api.models.schemas import ChatRequest, ChatResponse, APIResponse
from api.core.dependencies import get_orchestrator, get_current_context
from api.services.chat_service import ChatService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Normalizador de respuestas del chat (dict o ChatResponse -> dict estable)
def _to_chat_response_dict(resp: Any) -> Dict[str, Any]:
    if hasattr(resp, "model_dump"):
        d = resp.model_dump()  # type: ignore
        return {
            "message": d.get("message") or d.get("response") or d.get("result") or "",
            "agent": d.get("agent") or "coordinator",
            "success": bool(d.get("success", True)),
            "metadata": d.get("metadata") or {},
            "timestamp": d.get("timestamp") or datetime.now().isoformat(),
        }
    if isinstance(resp, dict):
        return {
            "message": resp.get("message") or resp.get("response") or resp.get("result") or "",
            "agent": resp.get("agent") or "coordinator",
            "success": bool(resp.get("success", True)),
            "metadata": resp.get("metadata") or {},
            "timestamp": resp.get("timestamp") or datetime.now().isoformat(),
        }
    return {
        "message": str(resp),
        "agent": "coordinator",
        "success": True,
        "metadata": {},
        "timestamp": datetime.now().isoformat(),
    }

@router.post("/chat", response_model=APIResponse)
async def chat_with_agent(
    request: ChatRequest,
    orchestrator = Depends(get_orchestrator),
    context: Dict[str, Any] = Depends(get_current_context)
):
    """
    Endpoint principal para chat con agentes
    """
    try:
        logger.info("Chat request: %s", request.message[:100])
        
        # Combinar contexto de la sesión con el contexto del request (tolerante a None)
        full_context = {**context, **(request.context or {})}
        if getattr(request, 'session_id', None):
            full_context['session_id'] = request.session_id  # type: ignore[attr-defined]
        elif 'session_id' not in full_context:
            full_context['session_id'] = str(uuid.uuid4())
        
        # Usar el servicio de chat para procesar
        chat_service = ChatService(orchestrator)
        resp = await chat_service.process_message(
            message=request.message,
            context=full_context,
            preferred_agent=request.agent_type
        )
        cr = _to_chat_response_dict(resp)
        
        return APIResponse(
            success=True,
            message="Chat procesado exitosamente",
            data={
                "chat_response": cr,
                "session_id": full_context.get("session_id"),
                "agent_used": cr["agent"],
                "timestamp": cr["timestamp"],
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error en chat: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando chat: {str(e)}"
        )

@router.post("/message", response_model=APIResponse)
async def send_chat_message(
    request: ChatRequest,
    orchestrator = Depends(get_orchestrator),
    context: Dict[str, Any] = Depends(get_current_context)
):
    """
    Endpoint para enviar mensajes de chat (compatible con cliente Streamlit)
    """
    try:
        logger.info("Chat message received: %s", request.message[:100])
        
        # Combinar contexto de la sesión con el contexto del request
        full_context = {**context, **(request.context or {})}
        
        # Si hay session_id en el request, usarlo
        if getattr(request, 'session_id', None):
            full_context['session_id'] = request.session_id  # type: ignore[attr-defined]
        elif 'session_id' not in full_context:
            full_context['session_id'] = str(uuid.uuid4())
        
        # Usar el servicio de chat para procesar
        chat_service = ChatService(orchestrator)
        resp = await chat_service.process_message(
            message=request.message,
            context=full_context,
            preferred_agent=getattr(request, 'agent_type', None)
        )
        cr = _to_chat_response_dict(resp)
        
        return APIResponse(
            success=True,
            message="Mensaje procesado exitosamente",
            data={
                "chat_response": cr,
                "session_id": full_context.get('session_id'),
                "agent_used": cr["agent"],
                "timestamp": cr["timestamp"],
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error processing chat message: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )

@router.get("/history/{session_id}", response_model=APIResponse)
async def get_chat_history(
    session_id: str,
    context: Dict[str, Any] = Depends(get_current_context)
):
    """
    Obtener historial de chat para una sesión específica
    """
    try:
        # TODO: Implementar persistencia de historial
        # Por ahora retornar historial vacío
        
        return APIResponse(
            success=True,
            message="Historial obtenido",
            data=[],  # Lista vacía por ahora
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error getting chat history: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting history: {str(e)}"
        )

@router.post("/chat/agent/{agent_type}", response_model=APIResponse)
async def chat_with_specific_agent(
    agent_type: str,
    request: ChatRequest,
    orchestrator = Depends(get_orchestrator),
    context: Dict[str, Any] = Depends(get_current_context)
):
    """
    Chat directo con un agente específico
    """
    try:
        # Forzar el tipo de agente
        request.agent_type = agent_type
        
        chat_service = ChatService(orchestrator)
        resp = await chat_service.process_message(
            message=request.message,
            context={**context, **(request.context or {})},
            preferred_agent=agent_type
        )
        cr = _to_chat_response_dict(resp)
        
        return APIResponse(
            success=True,
            message=f"Respuesta del agente {agent_type}",
            data={"chat_response": cr},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error en chat con agente %s: %s", agent_type, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error en agente {agent_type}: {str(e)}"
        )

@router.get("/chat/suggestions", response_model=APIResponse)
async def get_chat_suggestions(
    context: Dict[str, Any] = Depends(get_current_context)
):
    """
    Obtener sugerencias de comandos basadas en el contexto actual
    """
    suggestions = []
    
    # Sugerencias basadas en el contexto
    if context.get("dataset_uploaded"):
        suggestions.extend([
            "Analiza este dataset",
            "Genera 500 registros sintéticos con CTGAN",
            "Valida la coherencia médica de los datos",
            "¿Qué patrones encuentras en estos datos?"
        ])
    else:
        suggestions.extend([
            "¿Cómo funciona la generación de datos sintéticos?",
            "¿Qué tipos de datasets médicos puedo usar?",
            "Explícame los diferentes modelos disponibles",
            "¿Qué es CTGAN y cuándo usarlo?"
        ])
    
    # Sugerencias médicas generales
    suggestions.extend([
        "¿Cuáles son los factores de riesgo cardiovascular?",
        "Información sobre diabetes tipo 2",
        "Análisis de datos COVID-19"
    ])
    
    return APIResponse(
        success=True,
        message="Sugerencias generadas",
        data={"suggestions": suggestions},
        timestamp=datetime.now().isoformat()
    )

@router.delete("/chat/session")
async def clear_chat_session():
    """
    Limpiar la sesión actual de chat
    """
    try:
        # Aquí implementarías la limpieza de sesión
        # Por ejemplo, limpiar contexto almacenado, historial, etc.
        
        return APIResponse(
            success=True,
            message="Sesión de chat limpiada",
            data={"new_session_id": str(uuid.uuid4())},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error limpiando sesión: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error limpiando sesión: {str(e)}"
        )
