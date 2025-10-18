"""
Router para generación de datos sintéticos
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from datetime import datetime, timezone
import uuid
from typing import Dict, Any

from api.models.schemas import GenerationRequest, GenerationResponse, APIResponse, ModelType
from api.services.generation_service import GenerationService
from api.core.dependencies import get_orchestrator, get_dataset_service
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/generation/start", response_model=APIResponse)
async def start_generation(
    request: GenerationRequest,
    orchestrator = Depends(get_orchestrator),
    generation_service: GenerationService = Depends(lambda: GenerationService()),
    dataset_service = Depends(get_dataset_service)
):
    """
    Iniciar generación de datos sintéticos (síncrono)
    """
    try:
        logger.info("Iniciando generación: dataset=%s, modelo=%s, muestras=%d", 
                   request.dataset_id, request.model_type, request.num_samples)
        
        # Crear ID único para esta generación
        generation_id = str(uuid.uuid4())
        
        # Ejecutar generación de forma síncrona (esperar resultado)
        result = await generation_service.start_generation(
            generation_id=generation_id,
            dataset_id=request.dataset_id,
            model_type=request.model_type,
            num_samples=request.num_samples,
            selected_columns=request.selected_columns,
            parameters=request.parameters,
            orchestrator=orchestrator
        )
        
        # Obtener datos del resultado
        generation_data = generation_service.generations.get(generation_id, {})
        
        # Obtener datos sintéticos completos
        synthetic_df = generation_data.get("synthetic_data")
        synthetic_data_full = None
        
        if synthetic_df is not None:
            try:
                # Convertir DataFrame completo a dict para enviar al frontend
                synthetic_data_full = synthetic_df.to_dict(orient='records')
                logger.info("✅ Datos sintéticos completos preparados: %s registros", len(synthetic_data_full))
            except Exception as e:
                logger.warning("⚠️ No se pudieron convertir datos sintéticos completos: %s", e)
        
        # Obtener el nombre del dataset en lugar del ID
        dataset_name = request.dataset_id  # Por defecto usar el ID
        try:
            dataset_info = await dataset_service.get_dataset_info(request.dataset_id)
            if dataset_info:
                # dataset_info es un objeto DatasetInfo, acceder a su atributo
                dataset_name = dataset_info.filename if hasattr(dataset_info, 'filename') else request.dataset_id
                logger.info("✅ Nombre del dataset obtenido: %s", dataset_name)
            else:
                logger.warning("⚠️ No se obtuvo información del dataset")
        except Exception as e:
            logger.warning("⚠️ Error obteniendo nombre del dataset: %s", e)
        
        response_data = {
            "generation_id": generation_id,
            "status": generation_data.get("status", "completed"),
            "model_type": request.model_type,
            "num_samples": request.num_samples,
            "message": f"# Datos Sintéticos Generados\n\n"
                      f"**Modelo:** {request.model_type.upper()}\n"
                      f"**Registros generados:** {request.num_samples}\n\n"
                      f"✅ La generación se completó exitosamente.\n\n"
                      f"📁 **Dataset base:** {dataset_name}",
            "agent": "generator",
            "synthetic_data_preview": generation_data.get("synthetic_data_preview"),
            "synthetic_data": synthetic_data_full,  # 🆕 Datos completos para descarga
        }
        
        return APIResponse(
            success=True,
            message=f"Generación completada: {request.num_samples} registros con {request.model_type}",
            data=response_data,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error("❌ Error en generación: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error en generación: {str(e)}"
        )

@router.get("/generation/{generation_id}/status", response_model=APIResponse)
async def get_generation_status(
    generation_id: str,
    generation_service: GenerationService = Depends(lambda: GenerationService())
):
    """
    Obtener estado de una generación en curso
    """
    try:
        status_info = await generation_service.get_generation_status(generation_id)
        
        return APIResponse(
            success=True,
            message="Estado de generación obtenido",
            data={"generation_status": status_info},
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Generación {generation_id} no encontrada"
        )
    except Exception as e:
        logger.error("Error obteniendo estado de generación %s: %s", generation_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estado: {str(e)}"
        )

@router.get("/generation/{generation_id}/result", response_model=APIResponse)
async def get_generation_result(
    generation_id: str,
    preview_rows: int = 10,
    generation_service: GenerationService = Depends(lambda: GenerationService())
):
    """
    Obtener resultado de una generación completada
    """
    try:
        result = await generation_service.get_generation_result(generation_id, preview_rows)
        
        return APIResponse(
            success=True,
            message="Resultado de generación obtenido",
            data={"generation_result": result},
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Generación {generation_id} no encontrada"
        )
    except Exception as e:
        logger.error("Error obteniendo resultado de generación %s: %s", generation_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo resultado: {str(e)}"
        )

@router.get("/generation/{generation_id}/download", response_model=APIResponse)
async def download_generated_data(
    generation_id: str,
    format: str = "csv",  # csv, json, xlsx
    generation_service: GenerationService = Depends(lambda: GenerationService())
):
    """
    Descargar datos sintéticos generados
    """
    try:
        if format not in ["csv", "json", "xlsx"]:
            raise HTTPException(
                status_code=400,
                detail="Formato no soportado. Use: csv, json, xlsx"
            )
        
        download_info = await generation_service.prepare_download(generation_id, format)
        
        return APIResponse(
            success=True,
            message=f"Descarga preparada en formato {format}",
            data={"download_info": download_info},
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Generación {generation_id} no encontrada"
        )
    except Exception as e:
        logger.error("Error preparando descarga %s: %s", generation_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error preparando descarga: {str(e)}"
        )

@router.delete("/generation/{generation_id}", response_model=APIResponse)
async def delete_generation(
    generation_id: str,
    generation_service: GenerationService = Depends(lambda: GenerationService())
):
    """
    Eliminar una generación y sus archivos asociados
    """
    try:
        await generation_service.delete_generation(generation_id)
        
        return APIResponse(
            success=True,
            message=f"Generación {generation_id} eliminada exitosamente",
            data={"deleted_generation_id": generation_id},
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Generación {generation_id} no encontrada"
        )
    except Exception as e:
        logger.error("Error eliminando generación %s: %s", generation_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error eliminando generación: {str(e)}"
        )

@router.get("/generation", response_model=APIResponse)
async def list_generations(
    status_filter: str = None,  # "completed", "running", "failed", "all"
    generation_service: GenerationService = Depends(lambda: GenerationService())
):
    """
    Listar todas las generaciones
    """
    try:
        generations = await generation_service.list_generations(status_filter)
        
        return APIResponse(
            success=True,
            message=f"Se encontraron {len(generations)} generaciones",
            data={"generations": generations},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error listando generaciones: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error listando generaciones: {str(e)}"
        )

@router.post("/generation/{generation_id}/cancel", response_model=APIResponse)
async def cancel_generation(
    generation_id: str,
    generation_service: GenerationService = Depends(lambda: GenerationService())
):
    """
    Cancelar una generación en curso
    """
    try:
        await generation_service.cancel_generation(generation_id)
        
        return APIResponse(
            success=True,
            message=f"Generación {generation_id} cancelada",
            data={"cancelled_generation_id": generation_id},
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Generación {generation_id} no encontrada"
        )
    except Exception as e:
        logger.error("Error cancelando generación %s: %s", generation_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error cancelando generación: {str(e)}"
        )

@router.get("/generation/models/info", response_model=APIResponse)
async def get_models_info():
    """
    Obtener información sobre los modelos disponibles
    """
    models_info = {
        "ctgan": {
            "name": "CTGAN (Conditional Tabular GAN)",
            "description": "Red neuronal generativa adversarial especializada en datos tabulares",
            "pros": [
                "Excelente para datos mixtos (categóricos + numéricos)",
                "Maneja correlaciones complejas",
                "Rápido entrenamiento"
            ],
            "cons": [
                "Puede generar outliers",
                "Requiere ajuste de hiperparámetros"
            ],
            "best_for": "Datasets médicos con variables categóricas y numéricas mezcladas",
            "estimated_time_per_1000_samples": "2-5 minutos"
        },
        "tvae": {
            "name": "TVAE (Tabular Variational AutoEncoder)",
            "description": "Autoencoder variacional optimizado para datos tabulares",
            "pros": [
                "Preserva distribuciones estadísticas",
                "Menos propenso a outliers",
                "Estable y confiable"
            ],
            "cons": [
                "Puede ser conservador",
                "Menor diversidad en algunos casos"
            ],
            "best_for": "Cuando se requiere alta fidelidad estadística",
            "estimated_time_per_1000_samples": "3-7 minutos"
        },
        "sdv": {
            "name": "SDV (Synthetic Data Vault)",
            "description": "Suite completa de síntesis con múltiples algoritmos",
            "pros": [
                "Algoritmos múltiples integrados",
                "Optimizado para datos médicos",
                "Validación automática"
            ],
            "cons": [
                "Mayor complejidad computacional",
                "Tiempo de entrenamiento más largo"
            ],
            "best_for": "Proyectos que requieren máxima calidad y validación",
            "estimated_time_per_1000_samples": "5-15 minutos"
        }
    }
    
    return APIResponse(
        success=True,
        message="Información de modelos obtenida",
        data={"models": models_info},
        timestamp=datetime.now().isoformat()
    )
