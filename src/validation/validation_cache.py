"""
Sistema de caché para validaciones médicas.
Evita re-validar datos ya validados guardando resultados por hash.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class ValidationCache:
    """Gestor de caché para validaciones médicas."""
    
    def __init__(self, cache_dir: str = "cache/validator", ttl_hours: int = 24):
        """
        Inicializar caché de validaciones.
        
        Args:
            cache_dir: Directorio para almacenar caché
            ttl_hours: Tiempo de vida del caché en horas
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.hits = 0
        self.misses = 0
        
    def _compute_validation_hash(self, 
                                  df: pd.DataFrame, 
                                  is_covid: bool,
                                  validation_mode: str) -> str:
        """
        Calcular hash único para una validación.
        
        Args:
            df: DataFrame a validar
            is_covid: Si es dataset COVID
            validation_mode: Modo de validación ("sintéticos" o "originales")
            
        Returns:
            Hash hexadecimal de 16 caracteres
        """
        # Componentes del hash
        components = [
            str(df.shape),
            str(sorted(df.columns.tolist())),
            str(df.dtypes.to_dict()),
            str(is_covid),
            validation_mode,
            # Muestra de datos (primeras y últimas filas)
            df.head(5).to_json(),
            df.tail(5).to_json()
        ]
        
        content = "|".join(components)
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()[:16]
    
    def get(self, 
            df: pd.DataFrame, 
            is_covid: bool,
            validation_mode: str) -> Optional[Dict[str, Any]]:
        """
        Recuperar validación del caché si existe y es válida.
        
        Args:
            df: DataFrame a validar
            is_covid: Si es dataset COVID
            validation_mode: Modo de validación
            
        Returns:
            Resultados de validación o None si no existe/expirado
        """
        cache_hash = self._compute_validation_hash(df, is_covid, validation_mode)
        cache_file = self.cache_dir / f"validation_{cache_hash}.json"
        
        if not cache_file.exists():
            self.misses += 1
            logger.debug(f"❌ Cache MISS: {cache_hash}")
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Verificar TTL
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time > self.ttl:
                logger.debug(f"⏰ Cache EXPIRED: {cache_hash}")
                cache_file.unlink()
                self.misses += 1
                return None
            
            self.hits += 1
            logger.info(f"✅ Cache HIT: {cache_hash} (age: {datetime.now() - cached_time})")
            return cached_data['results']
            
        except Exception as e:
            logger.warning(f"Error loading cache {cache_hash}: {e}")
            self.misses += 1
            return None
    
    def put(self, 
            df: pd.DataFrame, 
            is_covid: bool,
            validation_mode: str,
            results: Dict[str, Any]) -> str:
        """
        Guardar validación en caché.
        
        Args:
            df: DataFrame validado
            is_covid: Si es dataset COVID
            validation_mode: Modo de validación
            results: Resultados de la validación
            
        Returns:
            Hash del caché guardado
        """
        cache_hash = self._compute_validation_hash(df, is_covid, validation_mode)
        cache_file = self.cache_dir / f"validation_{cache_hash}.json"
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'hash': cache_hash,
            'is_covid': is_covid,
            'validation_mode': validation_mode,
            'shape': list(df.shape),
            'results': results
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"💾 Cache SAVED: {cache_hash}")
            return cache_hash
        except Exception as e:
            logger.error(f"Error saving cache {cache_hash}: {e}")
            return ""
    
    def clear(self):
        """Limpiar todo el caché."""
        count = 0
        for cache_file in self.cache_dir.glob("validation_*.json"):
            cache_file.unlink()
            count += 1
        logger.info(f"🗑️ Cleared {count} validation cache entries")
        self.hits = 0
        self.misses = 0
    
    def clean_expired(self):
        """Limpiar entradas expiradas del caché."""
        count = 0
        for cache_file in self.cache_dir.glob("validation_*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                cached_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now() - cached_time > self.ttl:
                    cache_file.unlink()
                    count += 1
            except Exception:
                cache_file.unlink()
                count += 1
        
        if count > 0:
            logger.info(f"🗑️ Cleaned {count} expired validation cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        cache_files = list(self.cache_dir.glob("validation_*.json"))
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate_pct': hit_rate,
            'cache_entries': len(cache_files),
            'cache_dir': str(self.cache_dir)
        }


# Singleton global
_validation_cache: Optional[ValidationCache] = None

def get_validation_cache() -> ValidationCache:
    """Obtener instancia global del caché de validaciones."""
    global _validation_cache
    if _validation_cache is None:
        # Leer configuración desde variables de entorno
        cache_dir = os.getenv('VALIDATION_CACHE_DIR', 'cache/validator')
        ttl_hours = int(os.getenv('VALIDATION_CACHE_TTL_HOURS', '24'))
        _validation_cache = ValidationCache(cache_dir=cache_dir, ttl_hours=ttl_hours)
        logger.info(f"📦 Validation cache initialized: {cache_dir}, TTL: {ttl_hours}h")
    return _validation_cache
