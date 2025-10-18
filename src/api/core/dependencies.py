"""
Dependencias compartidas para la API - Versión simplificada
"""
import uuid
from typing import Dict, Any, Optional
from fastapi import HTTPException
from functools import lru_cache

from api.services.orchestrator_service import OrchestratorService
from api.services.dataset_service import DatasetService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Cache global para servicios
_orchestrator_service = None
_dataset_service = None
_context_storage = {}

@lru_cache()
def get_orchestrator():
    """
    Dependency para obtener el servicio de orquestación
    """
    global _orchestrator_service
    
    if _orchestrator_service is None:
        try:
            logger.info("Inicializando servicio de orquestación...")
            _orchestrator_service = OrchestratorService()
            logger.info("✅ Servicio de orquestación inicializado")
        except Exception as e:
            logger.error("Error inicializando orquestador: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Error inicializando orquestador: {str(e)}"
            )
    
    return _orchestrator_service

@lru_cache()
def get_dataset_service():
    """
    Dependency para obtener el servicio de datasets
    """
    global _dataset_service
    
    if _dataset_service is None:
        try:
            logger.info("Inicializando servicio de datasets...")
            _dataset_service = DatasetService()
            logger.info("✅ Servicio de datasets inicializado")
        except Exception as e:
            logger.error("Error inicializando servicio de datasets: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Error inicializando servicio de datasets: {str(e)}"
            )
    
    return _dataset_service

def get_current_context(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Dependency para obtener el contexto actual de la sesión
    """
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    if session_id not in _context_storage:
        _context_storage[session_id] = {
            "session_id": session_id,
            "created_at": uuid.uuid4().time,
            "datasets": {},
            "chat_history": []
        }
    
    return _context_storage[session_id]

def update_context(session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Actualiza el contexto de una sesión
    """
    if session_id in _context_storage:
        _context_storage[session_id].update(updates)
        return _context_storage[session_id]
    else:
        return get_current_context(session_id)

def clear_context(session_id: str) -> bool:
    """
    Limpia el contexto de una sesión específica
    """
    if session_id in _context_storage:
        del _context_storage[session_id]
        return True
    return False
