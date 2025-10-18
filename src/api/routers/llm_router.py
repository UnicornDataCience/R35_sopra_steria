"""
Router para gestión de proveedores LLM
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import os
from typing import Dict, Any

from api.models.schemas import APIResponse
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.get("/llm/providers", response_model=APIResponse)
async def list_llm_providers():
    """
    Lista todos los proveedores LLM disponibles y el activo
    """
    try:
        from src.config.llm_config import unified_llm_config
        
        status_info = unified_llm_config.status_info
        
        # Información detallada de cada proveedor
        providers_detail = {}
        for name, provider in unified_llm_config.providers.items():
            providers_detail[name] = {
                "available": provider.available,
                "model": getattr(provider, 'model', 'Unknown'),
                "name": provider.name
            }
        
        return APIResponse(
            success=True,
            message="Proveedores LLM listados",
            data={
                "active_provider": status_info["active_provider"],
                "available_providers": status_info["available_providers"],
                "providers_detail": providers_detail
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error listando proveedores LLM: %s", e)
        return APIResponse(
            success=False,
            message="Error listando proveedores",
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@router.post("/llm/switch/{provider_name}", response_model=APIResponse)
async def switch_llm_provider(provider_name: str):
    """
    Cambia el proveedor LLM activo
    
    Args:
        provider_name: Nombre del proveedor (groq, gemini, azure, ollama, grok)
    """
    try:
        from src.config.llm_config import unified_llm_config
        
        # Validar que el proveedor existe
        valid_providers = ["groq", "gemini", "azure", "ollama", "grok"]
        if provider_name.lower() not in valid_providers:
            return APIResponse(
                success=False,
                message=f"Proveedor inválido. Opciones: {', '.join(valid_providers)}",
                timestamp=datetime.now().isoformat()
            )
        
        # Obtener proveedor actual antes del cambio
        previous_provider = unified_llm_config.active_provider
        
        # Intentar cambiar el proveedor (sin test de conexión automático)
        provider = unified_llm_config.providers.get(provider_name.lower())
        
        if not provider:
            return APIResponse(
                success=False,
                message=f"Proveedor {provider_name} no encontrado",
                timestamp=datetime.now().isoformat()
            )
        
        if not provider.available:
            return APIResponse(
                success=False,
                message=f"Proveedor {provider_name} no está disponible. Verifica la configuración en .env",
                timestamp=datetime.now().isoformat()
            )
        
        # Cambiar el proveedor sin test (lazy)
        unified_llm_config.active_provider = provider_name.lower()
        
        # Actualizar variable de entorno para persistir en la sesión
        os.environ['LLM_PROVIDER'] = provider_name.lower()
        
        logger.info("✅ Proveedor LLM cambiado de %s a %s", previous_provider, provider_name.lower())
        
        return APIResponse(
            success=True,
            message=f"Proveedor cambiado exitosamente a {provider_name}",
            data={
                "previous_provider": previous_provider,
                "new_provider": provider_name.lower(),
                "model": provider.model,
                "status": unified_llm_config.status_info
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error cambiando proveedor LLM: %s", e)
        return APIResponse(
            success=False,
            message="Error cambiando proveedor",
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@router.get("/llm/status", response_model=APIResponse)
async def get_llm_status():
    """
    Obtiene el estado actual del proveedor LLM
    """
    try:
        from src.config.llm_config import unified_llm_config
        
        status_info = unified_llm_config.status_info
        
        return APIResponse(
            success=True,
            message="Estado del LLM obtenido",
            data=status_info,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error obteniendo estado LLM: %s", e)
        return APIResponse(
            success=False,
            message="Error obteniendo estado",
            error=str(e),
            timestamp=datetime.now().isoformat()
        )
