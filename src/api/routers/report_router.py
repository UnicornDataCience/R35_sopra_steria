"""
Router para el informe consolidado de historial clínico sintético de cohorte.

Ejecuta el pipeline determinista (analyzer -> generator -> validator ->
evaluator -> simulator) una sola vez y devuelve un único informe HTML con EDA,
generación, validación, métricas, evolución temporal (con gráficos) y
tratamiento (reglas deterministas).
"""
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from api.models.schemas import APIResponse
from api.core.dependencies import get_orchestrator
from src.utils.logging_config import get_logger
from src.reporting.report_builder import build_cohort_report

logger = get_logger(__name__)
router = APIRouter()


@router.post("/report", response_model=APIResponse)
async def generate_cohort_report(
    request: Dict[str, Any],
    orchestrator=Depends(get_orchestrator),
):
    """Genera el informe consolidado de cohorte para un dataset cargado."""
    dataset_id = request.get("dataset_id")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id es requerido")

    from api.services.dataset_service import DatasetService

    dataset_service = DatasetService()
    try:
        df = dataset_service.get_dataframe(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Dataset no encontrado: {e}")

    if df is None or getattr(df, "empty", True):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} vacío o no encontrado")

    model_type = request.get("model_type", "auto")
    try:
        num_samples = int(request.get("num_samples", 200))
    except (TypeError, ValueError):
        num_samples = 200

    dataset_name = dataset_id
    try:
        info = await dataset_service.get_dataset_info(dataset_id)
        dataset_name = getattr(info, "filename", dataset_id) or dataset_id
    except Exception:
        pass

    logger.info("Generando informe de cohorte: dataset=%s modelo=%s n=%d", dataset_id, model_type, num_samples)

    pipeline_result = await orchestrator.process_clinical_history(
        {"dataframe": df, "model_type": model_type, "num_samples": num_samples}
    )
    if pipeline_result.get("error"):
        raise HTTPException(status_code=500, detail=pipeline_result["error"])

    report = build_cohort_report(pipeline_result, dataset_name=dataset_name)
    if report.get("error"):
        raise HTTPException(status_code=500, detail=report["error"])

    synthetic = pipeline_result.get("synthetic_data")
    n_rows = int(len(synthetic)) if synthetic is not None else 0

    return APIResponse(
        success=True,
        message="Informe de historial clínico sintético de cohorte generado",
        data={
            "report_html": report["html"],
            "report_path": report["path"],
            "dataset_name": dataset_name,
            "num_samples": n_rows,
            "agents": list((pipeline_result.get("steps") or {}).keys()),
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
