"""
Middleware para manejo de errores en la API
"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import traceback

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

async def error_handler_middleware(request: Request, call_next):
    """
    Middleware para capturar y manejar errores de manera consistente
    """
    try:
        response = await call_next(request)
        return response
        
    except HTTPException as e:
        logger.warning("HTTP Exception: %s - %s", e.status_code, e.detail)
        return JSONResponse(
            status_code=e.status_code,
            content={
                "error": "HTTP Error",
                "message": e.detail,
                "status_code": e.status_code,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )
        
    except Exception as e:
        logger.error("Unhandled exception: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred",
                "status_code": 500,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )
