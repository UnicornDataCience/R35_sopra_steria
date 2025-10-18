"""
Configuración central de la API
"""
import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Configuración del servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "*"  # Para desarrollo, en producción especificar dominios exactos
    ]
    
    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Límites de archivos
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".xlsx", ".xls"]
    
    # Directorio temporal para archivos
    UPLOAD_DIR: str = "temp_uploads"
    
    # Configuración de logging
    LOG_LEVEL: str = "INFO"
    
    # Timeouts
    LLM_TIMEOUT: int = 120  # segundos
    PROCESSING_TIMEOUT: int = 300  # segundos
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignora variables extra del .env

settings = Settings()
