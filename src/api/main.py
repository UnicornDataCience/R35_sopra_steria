"""
Patient-IA API Server
API RESTful para el sistema de agentes médicos de generación de datos sintéticos
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import sys
from pathlib import Path

# Añadir la ruta del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.routers import (
    chat_router,
    dataset_router,
    generation_router,
    analysis_router,
    validation_router,
    health_router,
    evaluation_router,
    simulation_router,
    llm_router,
    report_router,
    auth_router
)
from src.api.cache_routes import router as cache_router
from api.middleware.error_handler import error_handler_middleware
from api.core.config import settings
from api.core.auth import get_current_user
from src.utils.logging_config import get_logger

# Dependencia de autenticación aplicada a los routers protegidos.
_auth = [Depends(get_current_user)]

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejo del ciclo de vida de la aplicación"""
    # Startup
    logger.info("🚀 Iniciando Patient-IA API Server...")
    logger.info("📋 Configuración LLM: %s", settings.LLM_PROVIDER)
    
    # Crear directorio de uploads si no existe
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("🔄 Cerrando Patient-IA API Server...")

# Crear la aplicación FastAPI
app = FastAPI(
    title="Patient-IA API",
    description="API para generación de datos clínicos sintéticos con agentes especializados",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configurar CORS para el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware personalizado para manejo de errores
app.middleware("http")(error_handler_middleware)

# Incluir routers
# Públicos: health, estado de LLM y autenticación.
app.include_router(health_router.router, prefix="/api/v1", tags=["health"])
app.include_router(llm_router.router, prefix="/api/v1", tags=["llm"])
app.include_router(auth_router.router, prefix="/api/v1", tags=["auth"])

# Protegidos por JWT (Depends(get_current_user)).
app.include_router(chat_router.router, prefix="/api/v1", tags=["chat"], dependencies=_auth)
app.include_router(dataset_router.router, prefix="/api/v1", tags=["datasets"], dependencies=_auth)
app.include_router(analysis_router.router, prefix="/api/v1", tags=["analysis"], dependencies=_auth)
app.include_router(generation_router.router, prefix="/api/v1", tags=["generation"], dependencies=_auth)
app.include_router(validation_router.router, prefix="/api/v1", tags=["validation"], dependencies=_auth)
app.include_router(evaluation_router.router, prefix="/api/v1", tags=["evaluation"], dependencies=_auth)
app.include_router(simulation_router.router, prefix="/api/v1", tags=["simulation"], dependencies=_auth)
app.include_router(report_router.router, prefix="/api/v1", tags=["report"], dependencies=_auth)

# Servir el frontend estático desde /app (incluye index.html y assets)
CLIENT_DIR = None
try:
    # main.py está en src/api/main.py, entonces PROJECT_ROOT es dos niveles arriba
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    CLIENT_DIR = PROJECT_ROOT / "client"
    
    logger.info("📂 PROJECT_ROOT: %s", PROJECT_ROOT)
    logger.info("📂 CLIENT_DIR: %s", CLIENT_DIR)
    logger.info("📂 CLIENT_DIR exists: %s", CLIENT_DIR.exists())
    
    if CLIENT_DIR.exists() and (CLIENT_DIR / "index.html").exists():
        app.mount("/app", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")
        
        @app.get("/app")
        async def app_redirect():
            # Asegura la barra final para que rutas relativas funcionen (script.js, style.css)
            return RedirectResponse(url="/app/")

        logger.info("✅ Frontend montado correctamente en /app")
        logger.info("🌐 Accede a la UI en: http://%s:%s/app/", settings.HOST, settings.PORT)
    else:
        logger.warning("❌ Frontend no encontrado o index.html faltante en %s", CLIENT_DIR)
except Exception as e:
    logger.error("❌ Error montando frontend: %s", e)
    import traceback
    logger.error(traceback.format_exc())

@app.get("/")
async def root():
    """Endpoint raíz"""
    ui_path = None
    try:
        if CLIENT_DIR.exists():
            ui_path = "/app"
    except Exception:
        pass
    return {
        "message": "Patient-IA API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "ui": ui_path,
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
