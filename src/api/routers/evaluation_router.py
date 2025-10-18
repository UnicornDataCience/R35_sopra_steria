"""
API Router for Data Evaluation
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from datetime import datetime, timezone

from api.models.schemas import APIResponse
from api.core.dependencies import get_orchestrator, get_dataset_service
from src.utils.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _normalize_orchestrator_response(raw: Any) -> Dict[str, Any]:
    """Acepta dict u objeto y devuelve un dict con response, agent, suggestions."""
    if raw is None:
        return {"response": "", "agent": "unknown", "suggestions": []}
    if isinstance(raw, dict):
        return {
            "response": raw.get("response") or raw.get("content") or raw.get("message") or raw.get("text") or "",
            "agent": raw.get("agent") or raw.get("agent_name") or raw.get("source") or "unknown",
            "suggestions": raw.get("suggestions") or raw.get("next_actions") or raw.get("hints") or [],
        }
    return {
        "response": getattr(raw, "response", None) or getattr(raw, "content", None) or getattr(raw, "message", None) or getattr(raw, "text", None) or "",
        "agent": getattr(raw, "agent", None) or getattr(raw, "agent_name", None) or getattr(raw, "source", None) or "unknown",
        "suggestions": getattr(raw, "suggestions", None) or getattr(raw, "next_actions", None) or [],
    }


@router.post("/evaluation", response_model=APIResponse)
async def evaluate_data(
    request: Dict[str, Any],
    orchestrator = Depends(get_orchestrator),
    dataset_service = Depends(get_dataset_service)
):
    """
    Endpoint to evaluate the quality of a dataset.
    """
    try:
        # 🔥 CARGAR el dataset antes de evaluar
        dataset_id = request.get("dataset_id")
        if not dataset_id:
            raise HTTPException(
                status_code=400,
                detail="dataset_id es requerido"
            )
        
        df = await dataset_service.load_dataset(dataset_id)
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {dataset_id} no encontrado o vacío"
            )
        
        logger.info("✅ Dataset cargado para evaluación: %sx%s", df.shape[0], df.shape[1])
        
        # 🆕 BUSCAR datos sintéticos generados recientemente para este dataset
        synthetic_df = None
        latest_generation_id = None
        
        from api.services.generation_service import GenerationService
        generation_service = GenerationService()
        
        logger.info("📋 Total de generaciones almacenadas: %d", len(generation_service.generations))
        for gen_id in generation_service.generations:
            logger.info("  - Generation ID: %s, Dataset: %s, Status: %s", 
                       gen_id, 
                       generation_service.generations[gen_id].get("dataset_id"),
                       generation_service.generations[gen_id].get("status"))
        
        # Buscar la generación más reciente completada para este dataset
        for gen_id, gen_data in generation_service.generations.items():
            if (gen_data.get("dataset_id") == dataset_id and 
                gen_data.get("status") == "completed" and 
                "synthetic_data" in gen_data):
                latest_generation_id = gen_id
                synthetic_df = gen_data.get("synthetic_data")
                logger.info("🔬 Encontrados datos sintéticos del generation_id: %s", gen_id)
                break  # Usar la más reciente
        
        # Construir mensaje y contexto para el orquestador
        if synthetic_df is not None and hasattr(synthetic_df, 'shape') and not synthetic_df.empty:
            logger.info("✅ Evaluando datos SINTÉTICOS (%sx%s) generados recientemente", 
                       synthetic_df.shape[0], synthetic_df.shape[1])
            eval_context = {
                **request,
                "action": "evaluate",
                "dataframe": df,  # Dataset original como referencia
                "synthetic_data": synthetic_df,  # Datos sintéticos a evaluar
                "synthetic_data_id": latest_generation_id,
                "evaluation_target": "synthetic"
            }
        else:
            logger.info("📊 No hay datos sintéticos recientes - evaluando dataset ORIGINAL")
            eval_context = {
                **request,
                "action": "evaluate",
                "dataframe": df,  # 🔥 CRÍTICO: Incluir el dataframe
                "evaluation_target": "original"
            }
        
        
        # Construir mensaje apropiado según el tipo de evaluación
        if synthetic_df is not None and not synthetic_df.empty:
            msg = f"Evalúa la calidad de los datos sintéticos generados para el dataset {dataset_id}"
        elif "original_dataset_id" in request and "synthetic_dataset_id" in request:
            msg = (
                f"Compara la calidad entre el dataset original {request['original_dataset_id']} "
                f"y el sintético {request['synthetic_dataset_id']}"
            )
        else:
            msg = f"Evalúa la calidad del dataset {dataset_id}"

        raw = await orchestrator.process_message(
            message=msg,
            context=eval_context,
            preferred_agent="evaluator"
        )
        norm = _normalize_orchestrator_response(raw)

        return APIResponse(
            success=True,
            message="Evaluación completada",
            data={
                "evaluation_response": {
                    "response": norm["response"],
                    "agent": norm["agent"],
                    "suggestions": norm["suggestions"],
                },
                "input": request,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
