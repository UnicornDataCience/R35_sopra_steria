"""
Sistema de caché para evaluaciones de datos sintéticos.
Evita re-evaluar mismos pares de datasets (original vs sintético).
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class EvaluationCache:
    """Gestor de caché para evaluaciones."""
    
    def __init__(self, cache_dir: str = "cache/evaluator", ttl_hours: int = 48):
        """
        Inicializar caché de evaluaciones.
        
        Args:
            cache_dir: Directorio para almacenar caché
            ttl_hours: Tiempo de vida del caché en horas (48h por defecto)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.hits = 0
        self.misses = 0
        
    def _compute_evaluation_hash(self, 
                                  df_original: pd.DataFrame, 
                                  df_synthetic: pd.DataFrame) -> str:
        """
        Calcular hash único para una evaluación.
        
        Args:
            df_original: DataFrame original
            df_synthetic: DataFrame sintético
            
        Returns:
            Hash hexadecimal de 16 caracteres
        """
        # Componentes del hash
        components = [
            # Original
            str(df_original.shape),
            str(sorted(df_original.columns.tolist())),
            df_original.head(5).to_json(),
            df_original.tail(5).to_json(),
            # Sintético
            str(df_synthetic.shape),
            str(sorted(df_synthetic.columns.tolist())),
            df_synthetic.head(5).to_json(),
            df_synthetic.tail(5).to_json()
        ]
        
        content = "|".join(components)
        hash_obj = hashlib.sha256(content.encode())
        return hash_obj.hexdigest()[:16]
    
    def get(self, 
            df_original: pd.DataFrame, 
            df_synthetic: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Recuperar evaluación del caché si existe y es válida.
        
        Args:
            df_original: DataFrame original
            df_synthetic: DataFrame sintético
            
        Returns:
            Resultados de evaluación o None si no existe/expirado
        """
        cache_hash = self._compute_evaluation_hash(df_original, df_synthetic)
        cache_file = self.cache_dir / f"evaluation_{cache_hash}.json"
        
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
            df_original: pd.DataFrame, 
            df_synthetic: pd.DataFrame,
            results: Dict[str, Any]) -> str:
        """
        Guardar evaluación en caché.
        
        Args:
            df_original: DataFrame original
            df_synthetic: DataFrame sintético
            results: Resultados de la evaluación
            
        Returns:
            Hash del caché guardado
        """
        cache_hash = self._compute_evaluation_hash(df_original, df_synthetic)
        cache_file = self.cache_dir / f"evaluation_{cache_hash}.json"
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'hash': cache_hash,
            'original_shape': list(df_original.shape),
            'synthetic_shape': list(df_synthetic.shape),
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
        for cache_file in self.cache_dir.glob("evaluation_*.json"):
            cache_file.unlink()
            count += 1
        logger.info(f"🗑️ Cleared {count} evaluation cache entries")
        self.hits = 0
        self.misses = 0
    
    def clean_expired(self):
        """Limpiar entradas expiradas del caché."""
        count = 0
        for cache_file in self.cache_dir.glob("evaluation_*.json"):
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
            logger.info(f"🗑️ Cleaned {count} expired evaluation cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        cache_files = list(self.cache_dir.glob("evaluation_*.json"))
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate_pct': hit_rate,
            'cache_entries': len(cache_files),
            'cache_dir': str(self.cache_dir)
        }


# Singleton global
_evaluation_cache: Optional[EvaluationCache] = None

def get_evaluation_cache() -> EvaluationCache:
    """Obtener instancia global del caché de evaluaciones."""
    global _evaluation_cache
    if _evaluation_cache is None:
        cache_dir = os.getenv('EVALUATION_CACHE_DIR', 'cache/evaluator')
        ttl_hours = int(os.getenv('EVALUATION_CACHE_TTL_HOURS', '48'))
        _evaluation_cache = EvaluationCache(cache_dir=cache_dir, ttl_hours=ttl_hours)
        logger.info(f"📦 Evaluation cache initialized: {cache_dir}, TTL: {ttl_hours}h")
    return _evaluation_cache
