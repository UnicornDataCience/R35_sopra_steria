"""
Servicio para generación de datos sintéticos
"""
import math
import uuid
import asyncio
from typing import Dict, Any, Optional, Union
from datetime import datetime

from api.models.schemas import GenerationResponse
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class GenerationService:
    """Servicio para generación de datos sintéticos"""
    
    # 🔥 Singleton: mantener generaciones entre instancias
    _shared_generations = {}
    
    def __init__(self):
        self.generations = GenerationService._shared_generations  # Usar storage compartido
    
    async def start_generation(
        self,
        generation_id: str,
        dataset_id: str,
        model_type: str = "ctgan",
        num_samples: int = 100,
        selected_columns: Optional[list] = None,
        parameters: Optional[Dict[str, Any]] = None,
        orchestrator = None
    ) -> GenerationResponse:
        """
        Inicia el proceso de generación de datos sintéticos
        """
        try:
            # Crear registro de generación
            generation_info = {
                "id": generation_id,
                "dataset_id": dataset_id,
                "model_type": model_type,
                "num_samples": num_samples,
                "selected_columns": selected_columns or [],
                "parameters": parameters or {},
                "status": "processing",
                "progress": 0.0,
                "created_at": datetime.now(),
                "synthetic_data": None
            }
            
            self.generations[generation_id] = generation_info
            
            logger.info("Iniciando generación sintética: %s con modelo %s", generation_id, model_type)
            
            # Si hay orchestrator, usar el agente generador real
            if orchestrator:
                try:
                    # Actualizar progreso
                    self.generations[generation_id]["status"] = "generating"
                    self.generations[generation_id]["progress"] = 0.3
                    
                    # 🔥 CRÍTICO: Cargar el dataset antes de generar
                    from api.services.dataset_service import DatasetService
                    dataset_service = DatasetService()
                    df = dataset_service.get_dataframe(dataset_id)
                    
                    if df is None or df.empty:
                        raise ValueError(f"Dataset {dataset_id} no encontrado o vacío")
                    
                    logger.info("Dataset cargado para generación: %sx%s", df.shape[0], df.shape[1])
                    
                    # Llamar al orchestrator para generar datos
                    context = {
                        "dataset_id": dataset_id,
                        "dataframe": df,  # 🔥 Agregar el dataframe al contexto
                        "model_type": model_type,
                        "num_samples": num_samples,
                        "selected_columns": selected_columns or [],
                        "parameters": parameters or {}
                    }
                    
                    # Convertir model_type a string si es un enum
                    model_name = getattr(model_type, 'value', str(model_type)).upper()
                    
                    result = await orchestrator.process_user_input(
                        f"Genera {num_samples} datos sintéticos con {model_name}",
                        context
                    )
                    
                    logger.info("Resultado del orchestrator keys: %s", list(result.keys()))
                    logger.info("Resultado tiene synthetic_data: %s", result.get("synthetic_data") is not None)
                    
                    # Extraer datos sintéticos del resultado
                    synthetic_df = result.get("synthetic_data")
                    
                    if synthetic_df is not None:
                        # Convertir a preview
                        preview_data = synthetic_df.head(10).to_dict(orient='records')
                        
                        self.generations[generation_id]["status"] = "completed"
                        self.generations[generation_id]["progress"] = 1.0
                        self.generations[generation_id]["synthetic_data"] = synthetic_df
                        self.generations[generation_id]["synthetic_data_preview"] = preview_data
                        
                        logger.info("✅ Generación completada: %s registros", len(synthetic_df))
                    else:
                        logger.warning("⚠️ Generación no retornó datos, usando mock")
                        self.generations[generation_id]["status"] = "completed"
                        self.generations[generation_id]["progress"] = 1.0
                        
                except Exception as e:
                    logger.error("❌ Error en generación con orchestrator: %s", e)
                    self.generations[generation_id]["status"] = "error"
                    self.generations[generation_id]["error"] = str(e)
            else:
                # Mock: completar inmediatamente
                logger.warning("⚠️ Sin orchestrator, usando modo mock")
                self.generations[generation_id]["status"] = "completed"
                self.generations[generation_id]["progress"] = 1.0
            
            return GenerationResponse(
                generation_id=generation_id,
                status=self.generations[generation_id]["status"],
                progress=self.generations[generation_id]["progress"],
                synthetic_data_preview=self.generations[generation_id].get("synthetic_data_preview"),
                generation_info={
                    "model_type": model_type,
                    "num_samples": num_samples,
                    "dataset_id": dataset_id
                }
            )
            
        except Exception as e:
            logger.error("Error iniciando generación: %s", e)
            if generation_id in self.generations:
                self.generations[generation_id]["status"] = "error"
                self.generations[generation_id]["error"] = str(e)
            raise Exception(f"Error iniciando generación: {str(e)}")
    
    async def get_generation_status(self, generation_id: str) -> Optional[GenerationResponse]:
        """
        Obtiene el estado de una generación
        """
        try:
            if generation_id not in self.generations:
                return None
            
            info = self.generations[generation_id]
            
            return GenerationResponse(
                generation_id=generation_id,
                status=info["status"],
                progress=info["progress"],
                synthetic_data_preview=info.get("synthetic_data_preview"),
                generation_info=info.get("generation_info", {}),
                download_url=info.get("download_url")
            )
            
        except Exception as e:
            logger.error("Error obteniendo estado de generación: %s", e)
            return None
        
    def estimate_generation_time(self, num_samples: int, model_type: str) -> int:
        """
        Estima minutos de generación según modelo y cantidad de muestras.
        """
        mt = str(model_type).lower()
        mt = (getattr(model_type, "value", None) or str(model_type)).lower()
        base_per_1000 = {
            "ctgan": 3,   # min por 1000 filas
            "tvae": 5,
            "sdv": 10
        }.get(mt, 5)
        return max(1, math.ceil((num_samples / 1000.0) * base_per_1000))
    
    async def cancel_generation(self, generation_id: str) -> bool:
        """
        Cancela una generación en proceso
        """
        try:
            if generation_id in self.generations:
                self.generations[generation_id]["status"] = "cancelled"
                logger.info("Generación cancelada: %s", generation_id)
                return True
            return False
            
        except Exception as e:
            logger.error("Error cancelando generación: %s", e)
            return False
