from typing import Dict, Any
import os
import random
import time
import numpy as np
import pandas as pd
from src.utils.logging_config import get_logger
from .base_agent import BaseLLMAgent, BaseAgentConfig
from ..generation.ctgan_generator import CTGANGenerator
from ..generation.tvae_generator import TVAEGenerator
from ..generation.sdv_generator import SDVGenerator
from ..generation.model_cache import get_model_cache
from ..generation.quality_metrics import get_quality_evaluator
from ..generation.early_stopping import EarlyStoppingConfig

logger = get_logger(__name__)

class SyntheticGeneratorAgent(BaseLLMAgent):
    """Agente especializado en generación de datos sintéticos con caché y métricas"""
    
    def __init__(self):
        config = BaseAgentConfig(
            name="Generador Sintético",
            description="Especialista en generación de datos clínicos sintéticos usando SDV y técnicas de ML avanzadas",
            system_prompt="Eres un agente experto en generación de datos sintéticos médicos. Tu misión es generar datos de alta calidad y responder de forma técnica y accesible."
        )
        super().__init__(config, tools=[])  # Explícitamente sin herramientas
        self.ctgan_generator = CTGANGenerator()
        self.tvae_generator = TVAEGenerator()
        self.sdv_generator = SDVGenerator()
        
        # Inicializar caché y evaluador de calidad
        self.cache = get_model_cache()
        self.quality_evaluator = get_quality_evaluator()
        
        # Métricas de rendimiento
        self.generation_count = 0
        self.cache_hits = 0
        self.total_generation_time = 0.0
        
        logger.info(f"SyntheticGeneratorAgent inicializado con caché habilitado: {self.cache.enabled}")

    def _choose_model_auto(self, df: pd.DataFrame) -> str:
        """Selecciona el modelo por heurística simple."""
        n_rows = len(df)
        n_cols = len(df.columns)
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        cat_cols = n_cols - num_cols
        cat_ratio = cat_cols / n_cols if n_cols else 0
        
        # Heurística básica
        if n_rows < 300:
            return "sdv"  # modelos más simples para pocos datos
        if cat_ratio >= 0.6:
            return "ctgan"
        return "tvae"

    def _set_global_seeds(self):
        """Fija semillas globales para reproducibilidad."""
        seed = int(os.getenv("GENERATOR_SEED", "42"))
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        return seed

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Punto de entrada principal para el agente generador con caché y métricas."""
        start_time = time.time()
        context = context or {}
        original_data = context.get("dataframe")
        
        if original_data is None:
            return {"message": "Error: No se encontró un dataset base para la generación.", "agent": self.name, "error": True}

        # Extraer parámetros de la solicitud del coordinador O directamente del contexto
        params = context.get("parameters", {})
        num_samples = params.get("num_samples") or context.get("num_samples", 100)
        model_type = params.get("model_type") or context.get("model_type", "auto")
        
        # Normalizar model_type a minúsculas
        if isinstance(model_type, str):
            model_type = model_type.lower()
        
        logger.info("⚙️ Parámetros de generación: model_type=%s, num_samples=%s", model_type, num_samples)

        # Selección por heurística si corresponde
        if model_type == "auto":
            model_type = self._choose_model_auto(original_data)
            logger.info("🤖 Selección automática de modelo: %s", model_type)

        # Semillas deterministas
        seed_used = self._set_global_seeds()

        try:
            synthetic_data = await self.generate_synthetic_data(original_data, num_samples, model_type, context)
            
            # Calcular métricas de calidad
            quality_metrics = None
            compute_quality = os.getenv('GENERATOR_COMPUTE_QUALITY_METRICS', 'true').lower() == 'true'
            
            if compute_quality and not synthetic_data.empty:
                logger.info("📊 Calculando métricas de calidad...")
                try:
                    quality_metrics = self.quality_evaluator.evaluate(original_data, synthetic_data)
                    logger.info(f"✅ Métricas de calidad: {quality_metrics}")
                    
                    # Advertir si la calidad es baja
                    quality_threshold = float(os.getenv('GENERATOR_QUALITY_THRESHOLD', '0.7'))
                    if quality_metrics.overall_quality < quality_threshold:
                        logger.warning(
                            f"⚠️ Calidad por debajo del umbral: "
                            f"{quality_metrics.overall_quality:.3f} < {quality_threshold}"
                        )
                except Exception as e:
                    logger.warning(f"Error calculando métricas de calidad: {e}")
            
            # Métricas de rendimiento
            elapsed_time = time.time() - start_time
            self.generation_count += 1
            self.total_generation_time += elapsed_time
            
            # Crear información de generación detallada
            generation_info = {
                "model_type": model_type,
                "num_samples": len(synthetic_data),
                "columns_used": len(synthetic_data.columns) if not synthetic_data.empty else 0,
                "selection_method": "MedicalColumnSelector" if context.get('selected_columns') else "Default",
                "timestamp": pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'),
                "seed": seed_used,
                "elapsed_time_seconds": round(elapsed_time, 2),
                "cache_hit": context.get('cache_hit', False),
                "quality_metrics": quality_metrics.to_dict() if quality_metrics else None,
                "agent_stats": {
                    "total_generations": self.generation_count,
                    "cache_hits": self.cache_hits,
                    "avg_generation_time": round(self.total_generation_time / self.generation_count, 2)
                }
            }
            
            logger.info(
                f"✅ Generación completada en {elapsed_time:.2f}s "
                f"(cache_hit={context.get('cache_hit', False)})"
            )
            
            return {
                "message": f"Se han generado exitosamente {len(synthetic_data)} registros sintéticos con el modelo {model_type.upper()}.",
                "agent": self.name,
                "synthetic_data": synthetic_data,
                "generation_info": generation_info
            }
        except Exception as e:
            logger.error(f"❌ Error durante la generación de datos: {e}", exc_info=True)
            return {"message": f"Error durante la generación de datos: {e}", "agent": self.name, "error": True}

    async def generate_synthetic_data(self, original_data: pd.DataFrame, num_samples: int, model_type: str, context: Dict[str, Any]) -> pd.DataFrame:
        """Genera datos sintéticos basados en el dataset original con caché."""
        import asyncio
        
        is_covid_dataset = context.get('universal_analysis', {}).get('dataset_type') == 'COVID-19'
        selected_columns = context.get('selected_columns')
        
        # Filtrar DataFrame si hay columnas seleccionadas
        if selected_columns:
            available_columns = [col for col in selected_columns if col in original_data.columns]
            if available_columns:
                logger.info("✂️ Usando %s columnas seleccionadas", len(available_columns))
                original_data = original_data[available_columns].copy()
            else:
                logger.warning("Ninguna columna seleccionada existe en el DataFrame. Usando dataset completo.")
        
        # Crear configuración para caché
        cache_config = {
            'num_samples': num_samples,
            'is_covid_dataset': is_covid_dataset,
            'selected_columns': sorted(selected_columns) if selected_columns else None,
            'model_type': model_type
        }
        
        # Intentar recuperar del caché
        cached_result = self.cache.get(original_data, model_type, cache_config)
        
        if cached_result is not None:
            model, cache_metadata = cached_result
            logger.info("🎯 Usando modelo del caché, generando samples...")
            context['cache_hit'] = True
            self.cache_hits += 1
            
            # Generar samples con modelo cacheado
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._generate_from_cached_model(model, num_samples, model_type)
                )
                return result
            except Exception as e:
                logger.warning(f"Error usando modelo cacheado: {e}. Reentrenando...")
                # Caer al flujo normal de entrenamiento
        
        # No hay caché o falló, entrenar modelo
        context['cache_hit'] = False
        logger.info(f"🔄 Entrenando modelo {model_type} desde cero...")
        
        # Función para ejecutar la generación en un executor
        def _run_generation():
            if model_type == 'ctgan':
                return self.ctgan_generator.generate(original_data, num_samples, is_covid_dataset, selected_columns)
            elif model_type == 'tvae':
                return self.tvae_generator.generate(original_data, num_samples, is_covid_dataset, selected_columns)
            elif model_type == 'sdv':
                return self.sdv_generator.generate(original_data, num_samples, is_covid_dataset, selected_columns)
            else:
                raise ValueError(f"Modelo '{model_type}' no soportado.")
        
        # Ejecutar con timeout de 10 minutos (600 segundos)
        try:
            logger.info(f"⏱️ Iniciando generación con modelo {model_type} (timeout: 600s)...")
            start_time = time.time()
            
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _run_generation),
                timeout=600.0
            )
            
            train_time = time.time() - start_time
            logger.info(f"✅ Generación completada en {train_time:.2f}s")
            
            # TODO: Cachear el modelo entrenado
            # Esto requeriría modificar los generadores para retornar (model, data)
            # Por ahora, solo retornamos los datos
            
            return result
        except asyncio.TimeoutError:
            logger.error("❌ Timeout: La generación tomó más de 10 minutos")
            raise TimeoutError(f"La generación con modelo {model_type} excedió el tiempo límite de 10 minutos")
        except Exception as e:
            logger.error(f"❌ Error durante la generación: {e}", exc_info=True)
            raise
    
    def _generate_from_cached_model(self, model: Any, num_samples: int, model_type: str) -> pd.DataFrame:
        """Genera samples usando un modelo cacheado."""
        try:
            # El modelo es un synthesizer de SDV ya entrenado
            return model.sample(num_samples)
        except Exception as e:
            logger.error(f"Error generando desde modelo cacheado: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas de rendimiento del generador.
        
        Returns:
            Diccionario con métricas acumuladas
        """
        cache_stats = self.cache.get_stats()
        
        return {
            'generation_count': self.generation_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / self.generation_count if self.generation_count > 0 else 0.0,
            'total_generation_time': round(self.total_generation_time, 2),
            'avg_generation_time': round(self.total_generation_time / self.generation_count, 2) if self.generation_count > 0 else 0.0,
            'cache_stats': cache_stats
        }
    
    def clear_cache(self) -> int:
        """
        Limpia el caché de modelos.
        
        Returns:
            Número de modelos eliminados
        """
        return self.cache.clear_all()
