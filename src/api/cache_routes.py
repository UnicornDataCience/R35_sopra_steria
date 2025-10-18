"""
Rutas API para gestión de caché.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import shutil
from src.utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/cache", tags=["cache"])


@router.post("/clear")
async def clear_cache():
    """
    Limpia toda la caché de análisis y simulaciones.
    
    Returns:
        Mensaje de confirmación
    """
    try:
        cache_paths = [
            Path("cache/analyzer/analyses"),
            Path("temp_generations/simulations"),
            Path("temp_generations/reports")
        ]
        
        cleared_count = 0
        for cache_path in cache_paths:
            if cache_path.exists():
                shutil.rmtree(cache_path)
                cache_path.mkdir(parents=True, exist_ok=True)
                cleared_count += 1
                logger.info(f"🗑️ Caché limpiada: {cache_path}")
        
        return {
            "success": True,
            "message": f"✅ Caché limpiada correctamente ({cleared_count} directorios)",
            "paths_cleared": [str(p) for p in cache_paths if p.exists()]
        }
    
    except Exception as e:
        logger.error(f"Error limpiando caché: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear/{cache_type}")
async def clear_specific_cache(cache_type: str):
    """
    Limpia un tipo específico de caché.
    
    Args:
        cache_type: Tipo de caché a limpiar ('analyses', 'simulations', 'reports')
    """
    cache_map = {
        "analyses": Path("cache/analyzer/analyses"),
        "simulations": Path("temp_generations/simulations"),
        "reports": Path("temp_generations/reports")
    }
    
    if cache_type not in cache_map:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de caché inválido. Opciones: {list(cache_map.keys())}"
        )
    
    try:
        cache_path = cache_map[cache_type]
        
        if cache_path.exists():
            shutil.rmtree(cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"🗑️ Caché '{cache_type}' limpiada: {cache_path}")
            
            return {
                "success": True,
                "message": f"✅ Caché '{cache_type}' limpiada correctamente",
                "path": str(cache_path)
            }
        else:
            return {
                "success": True,
                "message": f"⚠️ Caché '{cache_type}' no existía",
                "path": str(cache_path)
            }
    
    except Exception as e:
        logger.error(f"Error limpiando caché '{cache_type}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_cache_status():
    """
    Obtiene el estado actual de la caché.
    
    Returns:
        Información sobre el tamaño y contenido de la caché
    """
    try:
        cache_paths = {
            "analyses": Path("cache/analyzer/analyses"),
            "simulations": Path("temp_generations/simulations"),
            "reports": Path("temp_generations/reports")
        }
        
        status = {}
        total_size = 0
        
        for name, path in cache_paths.items():
            if path.exists():
                files = list(path.glob("**/*"))
                size = sum(f.stat().st_size for f in files if f.is_file())
                total_size += size
                
                status[name] = {
                    "exists": True,
                    "files": len([f for f in files if f.is_file()]),
                    "size_mb": round(size / (1024 * 1024), 2),
                    "path": str(path)
                }
            else:
                status[name] = {
                    "exists": False,
                    "files": 0,
                    "size_mb": 0,
                    "path": str(path)
                }
        
        return {
            "status": status,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "message": "✅ Estado de caché obtenido correctamente"
        }
    
    except Exception as e:
        logger.error(f"Error obteniendo estado de caché: {e}")
        raise HTTPException(status_code=500, detail=str(e))
