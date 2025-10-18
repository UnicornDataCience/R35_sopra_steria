"""
Router para manejo de datasets
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from datetime import datetime
import uuid
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from api.models.schemas import DatasetResponse, DatasetInfo, APIResponse, DatasetType
from api.core.config import settings
from api.services.dataset_service import DatasetService
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Crear directorio de uploads si no existe
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

@router.post("/datasets/upload", response_model=APIResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_service: DatasetService = Depends(lambda: DatasetService())
):
    """
    Subir y procesar un nuevo dataset
    """
    try:
        # Validar archivo
        if not file.filename.lower().endswith(tuple(settings.ALLOWED_EXTENSIONS)):
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de archivo no permitido. Use: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Verificar tamaño
        file_content = await file.read()
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Archivo muy grande. Máximo: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Procesar dataset
        dataset_info = await dataset_service.upload_dataset(file.filename, file_content)
        
        return APIResponse(
            success=True,
            message="Dataset cargado exitosamente",
            data={"dataset_info": dataset_info.model_dump()},
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error subiendo dataset: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando archivo: {str(e)}"
        )

@router.get("/datasets/{dataset_id}", response_model=APIResponse)
async def get_dataset_info(
    dataset_id: str,
    dataset_service: DatasetService = Depends(lambda: DatasetService())
):
    """
    Obtener información de un dataset específico
    """
    try:
        dataset_response = await dataset_service.get_dataset_info(dataset_id)
        
        return APIResponse(
            success=True,
            message="Información del dataset obtenida",
            data={"dataset": dataset_response.model_dump()},
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} no encontrado"
        )
    except Exception as e:
        logger.error("Error obteniendo dataset %s: %s", dataset_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo dataset: {str(e)}"
        )

@router.get("/datasets/{dataset_id}/preview", response_model=APIResponse)
async def get_dataset_preview(
    dataset_id: str,
    rows: int = 10,
    dataset_service: DatasetService = Depends(lambda: DatasetService())
):
    """
    Obtener vista previa de un dataset
    """
    try:
        preview_data = await dataset_service.get_dataset_preview(dataset_id, rows)
        
        return APIResponse(
            success=True,
            message=f"Vista previa del dataset ({rows} filas)",
            data={"preview": preview_data},
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} no encontrado"
        )
    except Exception as e:
        logger.error("Error obteniendo vista previa %s: %s", dataset_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo vista previa: {str(e)}"
        )

@router.get("/datasets/{dataset_id}/columns", response_model=APIResponse)
async def get_dataset_columns(
    dataset_id: str,
    dataset_service: DatasetService = Depends(lambda: DatasetService())
):
    """
    Obtener información detallada de las columnas del dataset
    """
    try:
        columns_info = await dataset_service.get_columns_info(dataset_id)
        
        return APIResponse(
            success=True,
            message="Información de columnas obtenida",
            data={"columns": columns_info},
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} no encontrado"
        )
    except Exception as e:
        logger.error("Error obteniendo columnas %s: %s", dataset_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo columnas: {str(e)}"
        )

@router.get("/datasets/{dataset_id}/statistics", response_model=APIResponse)
async def get_dataset_statistics(
    dataset_id: str,
    dataset_service: DatasetService = Depends(lambda: DatasetService())
):
    """
    Obtener estadísticas detalladas del dataset
    """
    try:
        statistics = await dataset_service.get_dataset_statistics(dataset_id)
        
        return APIResponse(
            success=True,
            message="Estadísticas del dataset obtenidas",
            data={"statistics": statistics},
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} no encontrado"
        )
    except Exception as e:
        logger.error("Error obteniendo estadísticas %s: %s", dataset_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo estadísticas: {str(e)}"
        )

@router.delete("/datasets/{dataset_id}", response_model=APIResponse)
async def delete_dataset(
    dataset_id: str,
    dataset_service: DatasetService = Depends(lambda: DatasetService())
):
    """
    Eliminar un dataset
    """
    try:
        await dataset_service.delete_dataset(dataset_id)
        
        return APIResponse(
            success=True,
            message=f"Dataset {dataset_id} eliminado exitosamente",
            data={"deleted_dataset_id": dataset_id},
            timestamp=datetime.now().isoformat()
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset {dataset_id} no encontrado"
        )
    except Exception as e:
        logger.error("Error eliminando dataset %s: %s", dataset_id, e)
        raise HTTPException(
            status_code=500,
            detail=f"Error eliminando dataset: {str(e)}"
        )

@router.get("/datasets", response_model=APIResponse)
async def list_datasets(
    dataset_service: DatasetService = Depends(lambda: DatasetService())
):
    """
    Listar todos los datasets disponibles
    """
    try:
        datasets = await dataset_service.list_datasets()
        
        return APIResponse(
            success=True,
            message=f"Se encontraron {len(datasets)} datasets",
            data={"datasets": [ds.model_dump() for ds in datasets]},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error("Error listando datasets: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error listando datasets: {str(e)}"
        )
