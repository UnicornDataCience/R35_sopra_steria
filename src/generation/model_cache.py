"""
Gestión de caché para modelos entrenados del generador.

Este módulo proporciona funcionalidades para:
- Cachear modelos entrenados (CTGAN, TVAE, SDV) para reutilización
- Calcular hash de datasets y configuraciones para claves de caché
- Serializar/deserializar modelos de forma segura
- Gestionar TTL y limpieza de caché antiguo
"""

import os
import pickle
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelCache:
    """Gestor de caché para modelos entrenados de generación sintética."""
    
    def __init__(
        self,
        cache_dir: str = "temp_generations/model_cache",
        ttl_hours: int = 24,
        enabled: bool = True
    ):
        """
        Inicializa el gestor de caché.
        
        Args:
            cache_dir: Directorio para almacenar modelos cacheados
            ttl_hours: Tiempo de vida del caché en horas
            enabled: Si el caché está habilitado
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        self.enabled = enabled
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"ModelCache inicializado: dir={cache_dir}, ttl={ttl_hours}h, enabled={enabled}")
    
    def compute_cache_key(
        self,
        df: pd.DataFrame,
        model_type: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Calcula un hash único para un dataset + configuración.
        
        Args:
            df: DataFrame de entrada
            model_type: Tipo de modelo ('ctgan', 'tvae', 'sdv')
            config: Configuración del modelo (epochs, batch_size, etc.)
            
        Returns:
            Hash hexadecimal de 16 caracteres
        """
        # Normalizar config para excluir valores None
        normalized_config = {k: v for k, v in config.items() if v is not None}
        
        components = [
            f"shape:{df.shape}",
            f"columns:{sorted(df.columns.tolist())}",
            f"dtypes:{sorted(df.dtypes.astype(str).to_dict().items())}",
            # NO usar sample de datos ya que cambia con cada generación aleatoria
            # En su lugar, usar estadísticas básicas que sean estables
            f"means:{df.select_dtypes(include=[np.number]).mean().to_dict()}",
            f"model:{model_type}",
            f"config:{sorted(normalized_config.items())}"
        ]
        
        hash_input = '|'.join(str(c) for c in components).encode('utf-8')
        cache_key = hashlib.sha256(hash_input).hexdigest()[:16]
        
        logger.debug(f"Cache key generado: {cache_key} para model={model_type}, shape={df.shape}")
        return cache_key
    
    def _get_model_path(self, cache_key: str, model_type: str) -> Path:
        """Retorna path del archivo del modelo."""
        return self.cache_dir / f"{cache_key}_{model_type}.pkl"
    
    def _get_metadata_path(self, cache_key: str, model_type: str) -> Path:
        """Retorna path del archivo de metadatos."""
        return self.cache_dir / f"{cache_key}_{model_type}_meta.json"
    
    def _is_cache_valid(self, metadata_path: Path) -> bool:
        """Verifica si el caché es válido basado en TTL."""
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            created_at = datetime.fromisoformat(metadata['created_at'])
            expires_at = created_at + timedelta(hours=self.ttl_hours)
            
            is_valid = datetime.now() < expires_at
            
            if not is_valid:
                logger.debug(f"Caché expirado: created={created_at}, expires={expires_at}")
            
            return is_valid
        except Exception as e:
            logger.warning(f"Error validando caché: {e}")
            return False
    
    def get(
        self,
        df: pd.DataFrame,
        model_type: str,
        config: Dict[str, Any]
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Intenta recuperar un modelo del caché.
        
        Args:
            df: DataFrame original
            model_type: Tipo de modelo
            config: Configuración del modelo
            
        Returns:
            Tuple (modelo, metadata) si existe y es válido, None si no
        """
        if not self.enabled:
            logger.debug("Caché deshabilitado, retornando None")
            return None
        
        cache_key = self.compute_cache_key(df, model_type, config)
        model_path = self._get_model_path(cache_key, model_type)
        metadata_path = self._get_metadata_path(cache_key, model_type)
        
        # Verificar si existe y es válido
        if not model_path.exists() or not self._is_cache_valid(metadata_path):
            logger.info(f"❌ Cache MISS para key={cache_key}, model={model_type}")
            return None
        
        try:
            # Cargar modelo
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Cargar metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"✅ Cache HIT para key={cache_key}, model={model_type}, age={self._get_age(metadata)}")
            return (model, metadata)
        
        except Exception as e:
            logger.error(f"Error cargando modelo del caché: {e}", exc_info=True)
            # Limpiar caché corrupto
            self._delete_cache_files(cache_key, model_type)
            return None
    
    def put(
        self,
        df: pd.DataFrame,
        model_type: str,
        config: Dict[str, Any],
        model: Any,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Guarda un modelo entrenado en el caché.
        
        Args:
            df: DataFrame original
            model_type: Tipo de modelo
            config: Configuración usada
            model: Modelo entrenado
            metrics: Métricas de entrenamiento
            
        Returns:
            True si se guardó exitosamente
        """
        if not self.enabled:
            logger.debug("Caché deshabilitado, no guardando modelo")
            return False
        
        cache_key = self.compute_cache_key(df, model_type, config)
        model_path = self._get_model_path(cache_key, model_type)
        metadata_path = self._get_metadata_path(cache_key, model_type)
        
        try:
            # Serializar modelo
            with open(model_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Guardar metadata
            metadata = {
                'cache_key': cache_key,
                'model_type': model_type,
                'config': config,
                'metrics': metrics,
                'dataset_shape': df.shape,
                'dataset_columns': df.columns.tolist(),
                'created_at': datetime.now().isoformat(),
                'model_size_bytes': model_path.stat().st_size
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            size_mb = metadata['model_size_bytes'] / (1024 * 1024)
            logger.info(f"💾 Modelo cacheado: key={cache_key}, model={model_type}, size={size_mb:.2f}MB")
            return True
        
        except Exception as e:
            logger.error(f"Error guardando modelo en caché: {e}", exc_info=True)
            # Limpiar archivos parciales
            self._delete_cache_files(cache_key, model_type)
            return False
    
    def _delete_cache_files(self, cache_key: str, model_type: str):
        """Elimina archivos de caché de un modelo."""
        model_path = self._get_model_path(cache_key, model_type)
        metadata_path = self._get_metadata_path(cache_key, model_type)
        
        for path in [model_path, metadata_path]:
            if path.exists():
                try:
                    path.unlink()
                    logger.debug(f"Eliminado archivo de caché: {path}")
                except Exception as e:
                    logger.warning(f"Error eliminando {path}: {e}")
    
    def _get_age(self, metadata: Dict[str, Any]) -> str:
        """Calcula la antigüedad del caché en formato legible."""
        created_at = datetime.fromisoformat(metadata['created_at'])
        age_seconds = (datetime.now() - created_at).total_seconds()
        
        if age_seconds < 60:
            return f"{int(age_seconds)}s"
        elif age_seconds < 3600:
            return f"{int(age_seconds / 60)}m"
        else:
            return f"{age_seconds / 3600:.1f}h"
    
    def clean_expired(self) -> int:
        """
        Limpia modelos expirados del caché.
        
        Returns:
            Número de modelos eliminados
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0
        
        deleted = 0
        for metadata_path in self.cache_dir.glob("*_meta.json"):
            if not self._is_cache_valid(metadata_path):
                # Extraer cache_key y model_type del nombre del archivo
                base_name = metadata_path.stem.replace('_meta', '')
                parts = base_name.rsplit('_', 1)
                if len(parts) == 2:
                    cache_key, model_type = parts
                    self._delete_cache_files(cache_key, model_type)
                    deleted += 1
        
        if deleted > 0:
            logger.info(f"🧹 Limpieza de caché: {deleted} modelos expirados eliminados")
        
        return deleted
    
    def clear_all(self) -> int:
        """
        Limpia todo el caché.
        
        Returns:
            Número de archivos eliminados
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0
        
        deleted = 0
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Error eliminando {file_path}: {e}")
        
        logger.info(f"🧹 Caché completamente limpiado: {deleted} archivos eliminados")
        return deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del caché.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.enabled or not self.cache_dir.exists():
            return {
                'enabled': False,
                'total_models': 0,
                'total_size_mb': 0
            }
        
        model_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in model_files)
        
        # Contar por tipo de modelo
        model_types = {}
        for f in model_files:
            model_type = f.stem.split('_')[-1]
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        return {
            'enabled': True,
            'cache_dir': str(self.cache_dir),
            'total_models': len(model_files),
            'total_size_mb': total_size / (1024 * 1024),
            'models_by_type': model_types,
            'ttl_hours': self.ttl_hours
        }


# Instancia global del caché (singleton)
_cache_instance: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """
    Retorna la instancia singleton del caché.
    Configuración desde variables de entorno.
    """
    global _cache_instance
    
    if _cache_instance is None:
        cache_dir = os.getenv('GENERATOR_CACHE_DIR', 'temp_generations/model_cache')
        ttl_hours = int(os.getenv('GENERATOR_CACHE_TTL_HOURS', '24'))
        enabled = os.getenv('GENERATOR_CACHE_ENABLED', 'true').lower() == 'true'
        
        _cache_instance = ModelCache(
            cache_dir=cache_dir,
            ttl_hours=ttl_hours,
            enabled=enabled
        )
    
    return _cache_instance
