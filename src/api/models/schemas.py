"""
Modelos Pydantic para las APIs
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

# Enums
class ModelType(str, Enum):
    CTGAN = "ctgan"
    TVAE = "tvae"
    SDV = "sdv"

class DatasetType(str, Enum):
    COVID = "covid"
    CARDIOLOGY = "cardiology"
    ONCOLOGY = "oncology"
    DIABETES = "diabetes"
    GENERAL = "general"

class AgentType(str, Enum):
    COORDINATOR = "coordinator"
    ANALYZER = "analyzer"
    GENERATOR = "generator"
    VALIDATOR = "validator"
    SIMULATOR = "simulator"
    EVALUATOR = "evaluator"

# Request Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="Mensaje del usuario")
    session_id: Optional[str] = Field(default=None, description="ID de la sesión de chat")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Contexto del chat")
    agent_type: Optional[AgentType] = Field(default=None, description="Agente específico a usar")

class GenerationRequest(BaseModel):
    dataset_id: str = Field(..., description="ID del dataset cargado")
    model_type: ModelType = Field(default=ModelType.CTGAN, description="Tipo de modelo a usar")
    num_samples: int = Field(default=100, ge=1, le=10000, description="Número de muestras a generar")
    selected_columns: Optional[List[str]] = Field(default=None, description="Columnas específicas a usar")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Parámetros adicionales del modelo")

class ValidationRequest(BaseModel):
    dataset_id: str = Field(..., description="ID del dataset a validar")
    synthetic_data_id: Optional[str] = Field(default=None, description="ID de datos sintéticos para comparar")

class AnalysisRequest(BaseModel):
    dataset_id: str = Field(..., description="ID del dataset a analizar")
    analysis_type: str = Field(default="comprehensive", description="Tipo de análisis")

# Response Models
class APIResponse(BaseModel):
    success: bool = Field(..., description="Si la operación fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Datos de respuesta")
    error: Optional[str] = Field(default=None, description="Mensaje de error si aplica")
    timestamp: str = Field(..., description="Timestamp de la respuesta")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Respuesta del agente")
    agent: str = Field(..., description="Agente que respondió")
    context: Dict[str, Any] = Field(default={}, description="Contexto actualizado")
    suggestions: Optional[List[str]] = Field(default=None, description="Sugerencias de acciones")

class DatasetInfo(BaseModel):
    id: str = Field(..., description="ID único del dataset")
    filename: str = Field(..., description="Nombre del archivo")
    rows: int = Field(..., description="Número de filas")
    columns: int = Field(..., description="Número de columnas")
    column_names: List[str] = Field(..., description="Nombres de las columnas")
    dtypes: Dict[str, str] = Field(..., description="Tipos de datos de las columnas")
    missing_values: Dict[str, int] = Field(..., description="Valores faltantes por columna")
    file_path: str = Field(..., description="Ruta del archivo")
    upload_time: datetime = Field(..., description="Tiempo de carga")
    file_size: int = Field(..., description="Tamaño del archivo en bytes")

class DatasetResponse(BaseModel):
    dataset_info: DatasetInfo
    preview: List[Dict[str, Any]] = Field(..., description="Vista previa de los datos")
    statistics: Dict[str, Any] = Field(..., description="Estadísticas básicas")

class GenerationResponse(BaseModel):
    generation_id: str = Field(..., description="ID único de la generación")
    status: str = Field(..., description="Estado de la generación")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progreso en porcentaje")
    synthetic_data_preview: Optional[List[Dict[str, Any]]] = Field(default=None, description="Vista previa de datos sintéticos")
    generation_info: Optional[Dict[str, Any]] = Field(default=None, description="Información de la generación")
    download_url: Optional[str] = Field(default=None, description="URL para descargar datos completos")

class ValidationResponse(BaseModel):
    validation_id: str = Field(..., description="ID único de la validación")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Puntuación general")
    medical_coherence: float = Field(..., ge=0.0, le=100.0, description="Coherencia médica")
    statistical_similarity: float = Field(..., ge=0.0, le=100.0, description="Similitud estadística")
    privacy_score: float = Field(..., ge=0.0, le=100.0, description="Puntuación de privacidad")
    issues_found: List[str] = Field(default=[], description="Problemas encontrados")
    recommendations: List[str] = Field(default=[], description="Recomendaciones")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado general del sistema")
    llm_status: str = Field(..., description="Estado del LLM")
    llm_provider: str = Field(..., description="Proveedor LLM activo")
    agents_available: bool = Field(..., description="Si los agentes están disponibles")
    version: str = Field(..., description="Versión de la API")
    uptime: str = Field(..., description="Tiempo de actividad")

# Error Models
class ErrorResponse(BaseModel):
    error: str = Field(..., description="Tipo de error")
    message: str = Field(..., description="Mensaje de error")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Detalles adicionales del error")
    timestamp: str = Field(..., description="Timestamp del error")
