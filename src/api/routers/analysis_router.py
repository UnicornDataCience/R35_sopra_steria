"""
Router para análisis de datasets
"""
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone
from typing import Dict, Any

from api.models.schemas import AnalysisRequest, APIResponse
from api.core.dependencies import get_orchestrator, get_current_context
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

def _normalize_orchestrator_response(raw: Any) -> Dict[str, Any]:
    """
    Acepta dict u objeto y devuelve un dict con response, agent, suggestions.
    """
    if raw is None:
        return {"response": "", "agent": "unknown", "suggestions": []}
    if isinstance(raw, dict):
        return {
            "response": raw.get("response") or raw.get("content") or raw.get("message") or raw.get("text") or "",
            "agent": raw.get("agent") or raw.get("agent_name") or raw.get("source") or "unknown",
            "suggestions": raw.get("suggestions") or raw.get("next_actions") or raw.get("hints") or [],
        }
    # objeto
    return {
        "response": getattr(raw, "response", None) or getattr(raw, "content", None) or getattr(raw, "message", None) or getattr(raw, "text", None) or "",
        "agent": getattr(raw, "agent", None) or getattr(raw, "agent_name", None) or getattr(raw, "source", None) or "unknown",
        "suggestions": getattr(raw, "suggestions", None) or getattr(raw, "next_actions", None) or [],
    }

@router.post("/analyze", response_model=APIResponse)
async def analyze_dataset(
    request: AnalysisRequest,
    orchestrator = Depends(get_orchestrator),
    context: Dict[str, Any] = Depends(get_current_context)
):
    """
    Analiza un dataset cargado
    """
    try:
        logger.info("Solicitud de análisis para dataset: %s", request.dataset_id)
        
        # CRÍTICO: Cargar el dataframe del dataset
        from api.services.dataset_service import DatasetService
        dataset_service = DatasetService()
        
        try:
            df = dataset_service.get_dataframe(request.dataset_id)
            logger.info("✅ Dataframe cargado: %sx%s", df.shape[0], df.shape[1])
        except Exception as e:
            logger.error("❌ Error cargando dataframe: %s", e)
            raise HTTPException(status_code=404, detail=f"Dataset no encontrado: {str(e)}")
        
        # Preparar contexto para análisis CON el dataframe
        analysis_context = {
            **context,
            "dataset_id": request.dataset_id,
            "analysis_type": request.analysis_type,
            "action": "analyze",
            "dataframe": df  # ← CRÍTICO: pasar el dataframe
        }
        
        # Procesar con el orquestador
        response = await orchestrator.process_message(
            message=f"Analiza el dataset {request.dataset_id} usando análisis {request.analysis_type}",
            context=analysis_context,
            preferred_agent="analyzer"
        )

        norm = _normalize_orchestrator_response(response)
        
        return APIResponse(
            success=True,
            message="Análisis completado exitosamente",
            data={
                "analysis_response": {
                    "response": norm["response"],
                    "agent": norm["agent"],
                    "suggestions": norm["suggestions"]
                },
                "dataset_id": request.dataset_id,
                "analysis_type": request.analysis_type
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error("Error en análisis: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error en análisis: {str(e)}"
        )

@router.get("/analyze/{dataset_id}/summary", response_model=APIResponse)
async def get_analysis_summary(
    dataset_id: str,
    context: Dict[str, Any] = Depends(get_current_context)
):
    """
    Obtiene un resumen de análisis previos para un dataset
    """
    try:
        # Mock response por ahora
        return APIResponse(
            success=True,
            message="Resumen de análisis obtenido",
            data={
                "dataset_id": dataset_id,
                "summary": {
                    "total_analyses": 0,
                    "last_analysis": None,
                    "available_reports": []
                }
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error("Error obteniendo resumen de análisis: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo resumen: {str(e)}"
        )
