"""
Router para health checks y estado del sistema
"""
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timedelta
import psutil
import os
from typing import Dict, Any

from api.models.schemas import HealthResponse, APIResponse
from api.core.dependencies import get_orchestrator
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Variable global para trackear uptime
start_time = datetime.now()

@router.get("/health", response_model=APIResponse)
async def health_check(orchestrator = Depends(get_orchestrator)):
    """
    Health check completo del sistema
    """
    try:
        # Verificar estado del LLM
        llm_status, llm_provider = await _check_llm_status()
        
        # Verificar agentes
        agents_available = _check_agents_status(orchestrator)
        
        # Calcular uptime
        uptime = datetime.now() - start_time
        uptime_str = str(uptime).split('.')[0]  # Remover microsegundos
        
        # Estado general
        overall_status = "healthy" if llm_status == "connected" and agents_available else "degraded"
        
        health_data = HealthResponse(
            status=overall_status,
            llm_status=llm_status,
            llm_provider=llm_provider,
            agents_available=agents_available,
            version="1.0.0",
            uptime=uptime_str
        )
        
        return APIResponse(
            success=True,
            message="Health check completado",
            data={"health": health_data.model_dump()},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error en health check: %s", e)
        return APIResponse(
            success=False,
            message="Error en health check",
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@router.get("/health/llm", response_model=APIResponse)
async def llm_health_check():
    """
    Health check específico del LLM
    """
    try:
        status, provider = await _check_llm_status()
        
        return APIResponse(
            success=status == "connected",
            message=f"Estado LLM: {status}",
            data={
                "llm_status": status,
                "provider": provider,
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error en LLM health check: %s", e)
        return APIResponse(
            success=False,
            message="Error verificando LLM",
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@router.get("/health/system", response_model=APIResponse)
async def system_health_check():
    """
    Health check del sistema (recursos)
    """
    try:
        # Obtener información del sistema
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Calcular uptime
        uptime = datetime.now() - start_time
        
        system_info = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime_human": str(uptime).split('.')[0]
        }
        
        # Determinar estado basado en recursos
        status = "healthy"
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = "warning"
        if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
            status = "critical"
        
        return APIResponse(
            success=True,
            message=f"Sistema: {status}",
            data={
                "system_status": status,
                "resources": system_info
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error en system health check: %s", e)
        return APIResponse(
            success=False,
            message="Error verificando sistema",
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@router.get("/health/agents", response_model=APIResponse)
async def agents_health_check(orchestrator = Depends(get_orchestrator)):
    """
    Health check específico de agentes
    """
    try:
        agents_status = {}
        
        # Verificar cada agente individualmente
        if hasattr(orchestrator, 'agents'):
            for agent_name, agent in orchestrator.agents.items():
                try:
                    # Verificar si el agente tiene métodos necesarios
                    has_process = hasattr(agent, 'process') or hasattr(agent, 'process_sync')
                    has_config = hasattr(agent, 'config')
                    
                    agents_status[agent_name] = {
                        "available": has_process,
                        "configured": has_config,
                        "type": agent.__class__.__name__,
                        "status": "ready" if has_process else "not_ready"
                    }
                except Exception as e:
                    agents_status[agent_name] = {
                        "available": False,
                        "configured": False,
                        "status": "error",
                        "error": str(e)
                    }
        else:
            agents_status = {"orchestrator": {"status": "mock_mode", "available": True}}
        
        # Contar agentes disponibles
        available_count = sum(1 for status in agents_status.values() if status.get("available", False))
        total_count = len(agents_status)
        
        return APIResponse(
            success=True,
            message=f"Agentes verificados: {available_count}/{total_count} disponibles",
            data={
                "agents_summary": {
                    "available_count": available_count,
                    "total_count": total_count,
                    "availability_percentage": (available_count / total_count * 100) if total_count > 0 else 0
                },
                "agents_detail": agents_status
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error en agents health check: %s", e)
        return APIResponse(
            success=False,
            message="Error verificando agentes",
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

@router.get("/workflow/stats", response_model=APIResponse)
async def get_workflow_statistics(
    orchestrator = Depends(get_orchestrator)
):
    """
    Obtiene estadísticas del flujo de trabajo de agentes
    """
    try:
        stats = orchestrator.get_workflow_statistics()
        transitions = orchestrator.get_available_transitions()
        
        return APIResponse(
            success=True,
            message="Estadísticas del workflow obtenidas",
            data={
                "workflow_stats": stats,
                "agent_transitions": transitions,
                "orchestrator_status": orchestrator.get_status()
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error obteniendo estadísticas del workflow: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estadísticas: {str(e)}"
        )

@router.get("/agents/available", response_model=APIResponse)
async def get_available_agents(
    orchestrator = Depends(get_orchestrator)
):
    """
    Lista todos los agentes disponibles y sus capacidades
    """
    try:
        status = orchestrator.get_status()
        transitions = orchestrator.get_available_transitions()
        
        # Información detallada de cada agente
        agent_info = {
            "coordinator": {
                "name": "Coordinador",
                "description": "Punto de entrada y coordinación general",
                "capabilities": ["conversación", "coordinación", "detección de intenciones"]
            },
            "analyzer": {
                "name": "Analizador Clínico", 
                "description": "Análisis de datasets médicos",
                "capabilities": ["análisis estadístico", "detección de patrones", "caracterización de datos"]
            },
            "generator": {
                "name": "Generador Sintético",
                "description": "Generación de datos sintéticos médicos",
                "capabilities": ["CTGAN", "TVAE", "SDV", "preservación de privacidad"]
            },
            "validator": {
                "name": "Validador Médico",
                "description": "Validación de coherencia médica",
                "capabilities": ["validación clínica", "detección de anomalías", "coherencia temporal"]
            },
            "evaluator": {
                "name": "Evaluador de Utilidad",
                "description": "Evaluación de calidad de datos sintéticos",
                "capabilities": ["métricas de calidad", "fidelidad estadística", "utilidad para ML"]
            },
            "simulator": {
                "name": "Simulador de Pacientes",
                "description": "Simulación de evolución temporal de pacientes",
                "capabilities": ["evolución temporal", "progresión de enfermedades", "realismo clínico"]
            }
        }
        
        return APIResponse(
            success=True,
            message="Agentes disponibles obtenidos",
            data={
                "orchestrator_status": status,
                "available_agents": agent_info,
                "possible_transitions": transitions,
                "workflow_description": {
                    "entry_point": "coordinator",
                    "typical_flow": ["coordinator", "analyzer", "generator", "validator", "evaluator"],
                    "optional_flows": {
                        "temporal_analysis": ["analyzer", "simulator"],
                        "quality_check": ["generator", "evaluator"],
                        "medical_validation": ["generator", "validator", "simulator"]
                    }
                }
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error obteniendo agentes disponibles: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo agentes: {str(e)}"
        )

async def _check_llm_status() -> tuple[str, str]:
    """Verificar estado del LLM"""
    try:
        from src.config.llm_config import unified_llm_config
        
        provider_info = unified_llm_config.status_info
        provider = provider_info.get("active_provider", "none")
        
        # Intentar test de conexión
        try:
            connection_test = unified_llm_config.test_connection()
            status = "connected" if connection_test else "configured_not_connected"
        except Exception:
            status = "configured_not_connected"
        
        return status, provider
        
    except Exception as e:
        logger.error("Error verificando LLM: %s", e)
        return "not_configured", "none"

def _check_agents_status(orchestrator) -> bool:
    """Verificar si los agentes están disponibles"""
    try:
        if hasattr(orchestrator, 'agents'):
            # Si tiene agentes reales
            return len(orchestrator.agents) > 0
        else:
            # Si es mock orchestrator, también está "disponible"
            return True
    except Exception:
        return False

@router.get("/health/detailed", response_model=APIResponse)
async def detailed_health_check(orchestrator = Depends(get_orchestrator)):
    """
    Health check detallado combinando toda la información
    """
    try:
        # Recopilar información de todos los health checks
        llm_status, llm_provider = await _check_llm_status()
        agents_available = _check_agents_status(orchestrator)
        
        # Sistema
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Uptime
        uptime = datetime.now() - start_time
        
        detailed_info = {
            "timestamp": datetime.now().isoformat(),
            "uptime": str(uptime).split('.')[0],
            "llm": {
                "status": llm_status,
                "provider": llm_provider,
                "connected": llm_status == "connected"
            },
            "agents": {
                "available": agents_available,
                "type": "real" if hasattr(orchestrator, 'agents') else "mock"
            },
            "system": {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "status": "healthy" if cpu_percent < 80 and memory.percent < 80 else "warning"
            },
            "api": {
                "version": "1.0.0",
                "environment": os.getenv("ENVIRONMENT", "development")
            }
        }
        
        # Estado general
        overall_healthy = (
            llm_status == "connected" and
            agents_available and
            cpu_percent < 90 and
            memory.percent < 90
        )
        
        return APIResponse(
            success=overall_healthy,
            message="Health check detallado completado",
            data={"detailed_health": detailed_info},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error en detailed health check: %s", e)
        return APIResponse(
            success=False,
            message="Error en health check detallado",
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

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
        
        # Intentar cambiar el proveedor
        success = unified_llm_config.switch_provider(provider_name.lower())
        
        if success:
            # Actualizar variable de entorno para persistir
            os.environ['LLM_PROVIDER'] = provider_name.lower()
            
            return APIResponse(
                success=True,
                message=f"Proveedor cambiado exitosamente a {provider_name}",
                data={
                    "previous_provider": unified_llm_config.active_provider,
                    "new_provider": provider_name.lower(),
                    "status": unified_llm_config.status_info
                },
                timestamp=datetime.now().isoformat()
            )
        else:
            return APIResponse(
                success=False,
                message=f"No se pudo cambiar a {provider_name}. Verifica que esté configurado correctamente.",
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
