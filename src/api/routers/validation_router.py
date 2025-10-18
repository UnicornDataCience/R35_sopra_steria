"""
Router para validación de datos sintéticos
"""
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timezone
from typing import Dict, Any

from api.models.schemas import ValidationRequest, ValidationResponse, APIResponse
from api.core.dependencies import get_orchestrator, get_dataset_service, get_current_context
from api.services.generation_service import GenerationService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _normalize_orchestrator_response(raw: Any) -> Dict[str, Any]:
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

@router.post("/validate", response_model=APIResponse)
async def validate_dataset(
    request: ValidationRequest,
    orchestrator = Depends(get_orchestrator),
    dataset_service = Depends(get_dataset_service),
    context: Dict[str, Any] = Depends(get_current_context)
):
    """
    Valida la coherencia médica de un dataset o datos sintéticos recientes
    """
    try:
        logger.info("Solicitud de validación para dataset: %s", request.dataset_id)
        
        # 🔥 CARGAR el dataset ORIGINAL
        df_original = await dataset_service.load_dataset(request.dataset_id)
        if df_original is None or df_original.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {request.dataset_id} no encontrado o vacío"
            )
        
        logger.info("✅ Dataset original cargado: %sx%s", df_original.shape[0], df_original.shape[1])
        
        # 🆕 BUSCAR datos sintéticos generados recientemente para este dataset
        synthetic_df = None
        latest_generation_id = None
        
        # Crear instancia del servicio para acceder a generaciones compartidas
        generation_service = GenerationService()
        
        logger.info("📋 Total de generaciones almacenadas: %d", len(generation_service.generations))
        for gen_id in generation_service.generations:
            logger.info("  - Generation ID: %s, Dataset: %s, Status: %s", 
                       gen_id, 
                       generation_service.generations[gen_id].get("dataset_id"),
                       generation_service.generations[gen_id].get("status"))
        
        # Buscar la generación más reciente completada para este dataset
        for gen_id, gen_data in generation_service.generations.items():
            if (gen_data.get("dataset_id") == request.dataset_id and 
                gen_data.get("status") == "completed" and 
                "synthetic_data" in gen_data):
                latest_generation_id = gen_id
                synthetic_df = gen_data.get("synthetic_data")
                logger.info("🔬 Encontrados datos sintéticos del generation_id: %s", gen_id)
                break  # Usar la más reciente
        
        # Preparar contexto para validación
        if synthetic_df is not None and hasattr(synthetic_df, 'shape') and not synthetic_df.empty:
            logger.info("✅ Validando datos SINTÉTICOS (%sx%s) generados recientemente", 
                       synthetic_df.shape[0], synthetic_df.shape[1])
            validation_context = {
                **context,
                "dataset_id": request.dataset_id,
                "dataframe": df_original,  # Dataset original como referencia
                "synthetic_data": synthetic_df,  # Datos sintéticos a validar
                "synthetic_data_id": latest_generation_id,
                "action": "validate",
                "validation_target": "synthetic"
            }
        else:
            logger.info("📊 No hay datos sintéticos recientes - validando dataset ORIGINAL")
            validation_context = {
                **context,
                "dataset_id": request.dataset_id,
                "dataframe": df_original,
                "synthetic_data_id": request.synthetic_data_id,
                "action": "validate",
                "validation_target": "original"
            }
        
        # Procesar con el orquestador
        response = await orchestrator.process_message(
            message=f"Valida la coherencia médica del dataset {request.dataset_id}",
            context=validation_context,
            preferred_agent="validator"
        )

        norm = _normalize_orchestrator_response(response)
        
        # Mock validation response por ahora
        validation_result = ValidationResponse(
            validation_id=f"val_{request.dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            overall_score=85.5,
            medical_coherence=88.0,
            statistical_similarity=83.0,
            privacy_score=95.0,
            issues_found=[
                "Algunas correlaciones podrían mejorarse",
                "Distribución de edades ligeramente desviada"
            ],
            recommendations=[
                "Revisar parámetros del modelo CTGAN",
                "Considerar usar TVAE para mejor preservación estadística"
            ]
        )
        
        return APIResponse(
            success=True,
            message="Validación completada exitosamente",
            data={
                "validation_result": validation_result.model_dump(),
                "agent_response": {
                    "response": norm["response"],
                    "agent": norm["agent"],
                    "suggestions": norm["suggestions"]
                }
            },
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error("Error en validación: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error en validación: {str(e)}"
        )

@router.get("/validate/{validation_id}", response_model=APIResponse)
async def get_validation_result(
    validation_id: str,
    context: Dict[str, Any] = Depends(get_current_context)
):
    """
    Obtiene los resultados de una validación específica
    """
    try:
        # Mock response por ahora
        return APIResponse(
            success=True,
            message="Resultado de validación obtenido",
            data={
                "validation_id": validation_id,
                "status": "completed",
                "results": {
                    "overall_score": 85.5,
                    "details": "Validación completada con éxito"
                }
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error obteniendo resultado de validación: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo resultado: {str(e)}"
        )
