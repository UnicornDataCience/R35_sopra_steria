"""
Utilidades para Optimización del Sistema Patient-IA

Este módulo contiene funciones reutilizables para:
- Hashing y versionado de datasets
- Caché de modelos y resultados
- Métricas de performance
- Logging mejorado

Autor: Sistema Patient-IA
Versión: 1.0
Fecha: 2025-10-15
"""

import hashlib
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from functools import wraps

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# ============================================================================
# HASHING Y VERSIONADO
# ============================================================================

def get_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Calcular hash único y reproducible de un DataFrame.
    
    Args:
        df: DataFrame a hashear
        
    Returns:
        Hash hexadecimal de 16 caracteres
        
    Example:
        >>> hash_val = get_dataframe_hash(df)
        >>> print(hash_val)  # 'a3f5e8c9d2b1f4a7'
    """
    try:
        # Combinar shape, columnas y sample de datos
        content = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'head_sample': df.head(5).to_dict(),
            'tail_sample': df.tail(5).to_dict()
        }
        
        content_str = json.dumps(content, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(content_str.encode())
        return hash_obj.hexdigest()[:16]
    except Exception as e:
        logger.warning(f"Error calculating DataFrame hash: {e}")
        # Fallback: usar solo shape
        fallback = f"{df.shape[0]}x{df.shape[1]}"
        return hashlib.sha256(fallback.encode()).hexdigest()[:16]


def version_dataframe(df: pd.DataFrame, metadata: Optional[Dict] = None) -> tuple:
    """
    Versionar DataFrame con hash y metadata.
    
    Args:
        df: DataFrame a versionar
        metadata: Metadata adicional (opcional)
        
    Returns:
        Tuple (hash, version_info_dict)
        
    Example:
        >>> hash_val, info = version_dataframe(df, {'experiment': 'test_1'})
        >>> print(f"Version: {hash_val}")
    """
    hash_val = get_dataframe_hash(df)
    
    version_info = {
        'hash': hash_val,
        'timestamp': datetime.now().isoformat(),
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_mb': df.memory_usage(deep=True).sum() / 1e6,
        'null_count': int(df.isna().sum().sum()),
        'metadata': metadata or {}
    }
    
    # Guardar versión si no existe
    version_dir = Path('data/versions')
    version_dir.mkdir(parents=True, exist_ok=True)
    
    version_file = version_dir / f"dataset_{hash_val}.json"
    if not version_file.exists():
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        logger.info(f"📦 Dataset versioned: {hash_val}")
    
    return hash_val, version_info


# ============================================================================
# CACHÉ DE RESULTADOS
# ============================================================================

class CacheManager:
    """
    Gestor de caché para modelos y resultados.
    
    Example:
        >>> cache = CacheManager()
        >>> cache.save('my_model', model_obj, 'ctgan')
        >>> loaded = cache.load('my_model', 'ctgan')
    """
    
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CacheManager initialized: {self.cache_dir}")
    
    def _get_cache_path(self, key: str, category: str) -> Path:
        """Construir path de caché"""
        category_dir = self.cache_dir / category
        category_dir.mkdir(exist_ok=True)
        return category_dir / f"{key}.pkl"
    
    def save(self, key: str, obj: Any, category: str = 'general') -> bool:
        """
        Guardar objeto en caché.
        
        Args:
            key: Clave única (puede ser hash)
            obj: Objeto a guardar
            category: Categoría de caché (ej: 'models', 'analysis', 'results')
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            cache_path = self._get_cache_path(key, category)
            with open(cache_path, 'wb') as f:
                pickle.dump(obj, f)
            
            size_mb = cache_path.stat().st_size / 1e6
            logger.info(f"💾 Cached [{category}] {key}: {size_mb:.2f} MB")
            return True
        except Exception as e:
            logger.error(f"Error caching [{category}] {key}: {e}")
            return False
    
    def load(self, key: str, category: str = 'general') -> Optional[Any]:
        """
        Cargar objeto desde caché.
        
        Args:
            key: Clave única
            category: Categoría de caché
            
        Returns:
            Objeto cacheado o None si no existe
        """
        try:
            cache_path = self._get_cache_path(key, category)
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'rb') as f:
                obj = pickle.load(f)
            
            logger.info(f"📦 Loaded from cache [{category}] {key}")
            return obj
        except Exception as e:
            logger.warning(f"Error loading from cache [{category}] {key}: {e}")
            return None
    
    def exists(self, key: str, category: str = 'general') -> bool:
        """Verificar si existe en caché"""
        cache_path = self._get_cache_path(key, category)
        return cache_path.exists()
    
    def clear(self, category: Optional[str] = None):
        """
        Limpiar caché.
        
        Args:
            category: Categoría específica a limpiar, o None para todo
        """
        if category:
            category_dir = self.cache_dir / category
            if category_dir.exists():
                for file in category_dir.glob('*.pkl'):
                    file.unlink()
                logger.info(f"🗑️ Cleared cache category: {category}")
        else:
            for file in self.cache_dir.rglob('*.pkl'):
                file.unlink()
            logger.info("🗑️ Cleared all cache")


# ============================================================================
# MÉTRICAS DE PERFORMANCE
# ============================================================================

def time_it(func):
    """
    Decorador para medir tiempo de ejecución.
    
    Example:
        >>> @time_it
        >>> def my_function():
        >>>     time.sleep(1)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        func_name = func.__name__
        logger.info(f"⏱️ {func_name} executed in {elapsed:.2f}s")
        
        return result
    
    return wrapper


class PerformanceTracker:
    """
    Rastreador de métricas de performance.
    
    Example:
        >>> tracker = PerformanceTracker('analysis')
        >>> tracker.start()
        >>> # ... operación ...
        >>> tracker.stop()
        >>> tracker.log_summary()
    """
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def start(self):
        """Iniciar tracking"""
        self.start_time = time.time()
        logger.debug(f"▶️ Started: {self.operation_name}")
    
    def stop(self):
        """Detener tracking"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.metrics['duration_seconds'] = elapsed
        logger.debug(f"⏹️ Stopped: {self.operation_name} ({elapsed:.2f}s)")
    
    def add_metric(self, key: str, value: Any):
        """Agregar métrica adicional"""
        self.metrics[key] = value
    
    def log_summary(self):
        """Logging de resumen"""
        if 'duration_seconds' not in self.metrics:
            self.stop()
        
        logger.info(f"📊 Performance [{self.operation_name}]:")
        for key, value in self.metrics.items():
            logger.info(f"  - {key}: {value}")
    
    def get_metrics(self) -> Dict:
        """Obtener métricas como diccionario"""
        return self.metrics.copy()


# ============================================================================
# BACKUP Y RESTORE
# ============================================================================

def create_backup(file_path: str, backup_dir: str = 'backups') -> Optional[str]:
    """
    Crear backup de un archivo antes de modificarlo.
    
    Args:
        file_path: Path del archivo a respaldar
        backup_dir: Directorio de backups
        
    Returns:
        Path del backup o None si falla
        
    Example:
        >>> backup_path = create_backup('src/agents/analyzer_agent.py')
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File not found for backup: {file_path}")
            return None
        
        # Crear directorio de backups
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Nombre de backup con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        # Copiar archivo
        import shutil
        shutil.copy2(file_path, backup_path)
        
        logger.info(f"💾 Backup created: {backup_path}")
        return str(backup_path)
    
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        return None


# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

class ExperimentTracker:
    """
    Rastreador de experimentos para investigación.
    
    Example:
        >>> tracker = ExperimentTracker('covid_ctgan_500')
        >>> tracker.log_parameter('model', 'CTGAN')
        >>> tracker.log_metric('f1_score', 0.85)
        >>> tracker.finish()
    """
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.log = {
            'name': experiment_name,
            'start_time': self.start_time.isoformat(),
            'parameters': {},
            'datasets': {},
            'results': {},
            'metrics': {},
            'performance': {}
        }
    
    def log_parameter(self, key: str, value: Any):
        """Registrar parámetro del experimento"""
        self.log['parameters'][key] = value
        logger.debug(f"📝 Logged parameter: {key}={value}")
    
    def log_dataset(self, name: str, df: pd.DataFrame):
        """Registrar dataset usado"""
        hash_val, version_info = version_dataframe(df, {'experiment': self.experiment_name})
        self.log['datasets'][name] = {
            'hash': hash_val,
            'shape': df.shape,
            'version_file': f"data/versions/dataset_{hash_val}.json"
        }
        logger.debug(f"📊 Logged dataset: {name} (hash: {hash_val})")
    
    def log_metric(self, key: str, value: Any):
        """Registrar métrica de resultado"""
        self.log['metrics'][key] = value
        logger.debug(f"📈 Logged metric: {key}={value}")
    
    def log_result(self, key: str, value: Any):
        """Registrar resultado del experimento"""
        self.log['results'][key] = value
        logger.debug(f"✅ Logged result: {key}")
    
    def log_performance(self, key: str, value: Any):
        """Registrar métrica de performance"""
        self.log['performance'][key] = value
    
    def finish(self) -> str:
        """Finalizar y guardar experimento"""
        self.log['end_time'] = datetime.now().isoformat()
        self.log['duration_seconds'] = (datetime.now() - self.start_time).total_seconds()
        
        # Guardar log
        exp_dir = Path('experiments')
        exp_dir.mkdir(exist_ok=True)
        
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        exp_file = exp_dir / f"{self.experiment_name}_{timestamp}.json"
        
        with open(exp_file, 'w') as f:
            json.dump(self.log, f, indent=2, default=str)
        
        logger.info(f"📝 Experiment logged: {exp_file}")
        return str(exp_file)


# ============================================================================
# VALIDACIÓN DE ESTADO
# ============================================================================

def validate_system_state() -> Dict[str, bool]:
    """
    Validar que el sistema está en buen estado.
    
    Returns:
        Diccionario con resultados de validación
        
    Example:
        >>> state = validate_system_state()
        >>> if all(state.values()):
        >>>     print("✅ Sistema OK")
    """
    checks = {}
    
    # 1. Verificar estructura de directorios
    required_dirs = [
        'src/agents',
        'src/api',
        'src/generation',
        'src/validation',
        'src/evaluation',
        'src/simulation',
        'client',
        'data',
        'logs'
    ]
    
    for dir_path in required_dirs:
        checks[f'dir_{dir_path}'] = Path(dir_path).exists()
    
    # 2. Verificar archivos críticos
    required_files = [
        'src/agents/base_agent.py',
        'src/agents/coordinator_agent.py',
        'src/agents/analyzer_agent.py',
        'client/index.html',
        'client/script.js',
        'run_api.py'
    ]
    
    for file_path in required_files:
        checks[f'file_{file_path}'] = Path(file_path).exists()
    
    # 3. Verificar imports críticos
    try:
        import pandas
        import numpy
        from langchain import prompts
        checks['import_critical'] = True
    except ImportError as e:
        checks['import_critical'] = False
        logger.error(f"Import error: {e}")
    
    # Log resumen
    total = len(checks)
    passed = sum(checks.values())
    logger.info(f"🔍 System validation: {passed}/{total} checks passed")
    
    if passed < total:
        logger.warning("⚠️ Some validation checks failed:")
        for key, value in checks.items():
            if not value:
                logger.warning(f"  ❌ {key}")
    
    return checks


# ============================================================================
# UTILIDADES DE LOGGING MEJORADO
# ============================================================================

def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Logging detallado de información de DataFrame.
    
    Args:
        df: DataFrame a loggear
        name: Nombre descriptivo
    """
    logger.info(f"📊 {name} Info:")
    logger.info(f"  - Shape: {df.shape} (rows × cols)")
    logger.info(f"  - Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    logger.info(f"  - Columns: {df.columns.tolist()}")
    logger.info(f"  - Nulls: {df.isna().sum().sum()} ({df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%)")
    logger.info(f"  - Dtypes: {df.dtypes.value_counts().to_dict()}")


# ============================================================================
# EXPORTAR TODO
# ============================================================================

__all__ = [
    'get_dataframe_hash',
    'version_dataframe',
    'CacheManager',
    'time_it',
    'PerformanceTracker',
    'create_backup',
    'ExperimentTracker',
    'validate_system_state',
    'log_dataframe_info'
]
