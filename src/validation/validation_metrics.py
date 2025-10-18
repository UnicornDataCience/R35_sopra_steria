"""
Métricas detalladas para validaciones médicas.
Tracking de performance y calidad de validaciones.
"""
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class ValidationMetrics:
    """Métricas de una validación individual."""
    
    # Identificación
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    validation_mode: str = "sintéticos"
    is_covid: bool = False
    
    # Tamaño del dataset
    num_rows: int = 0
    num_columns: int = 0
    
    # Scores
    overall_score: float = 0.0
    clinical_coherence: float = 0.0
    data_quality: float = 0.0
    
    # Issues
    num_issues: int = 0
    issues: List[str] = field(default_factory=list)
    
    # Performance
    validation_time_ms: float = 0.0
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'timestamp': self.timestamp,
            'validation_mode': self.validation_mode,
            'is_covid': self.is_covid,
            'dataset_size': f"{self.num_rows}x{self.num_columns}",
            'scores': {
                'overall': round(self.overall_score, 3),
                'clinical_coherence': round(self.clinical_coherence, 3),
                'data_quality': round(self.data_quality, 3)
            },
            'issues': {
                'count': self.num_issues,
                'list': self.issues
            },
            'performance': {
                'validation_time_ms': round(self.validation_time_ms, 2),
                'cache_hit': self.cache_hit
            }
        }


class ValidatorPerformanceTracker:
    """Tracker de performance del validador."""
    
    def __init__(self):
        self.total_validations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_validation_time_ms = 0.0
        self.total_rows_validated = 0
        self.total_issues_found = 0
        self.validations_history: List[ValidationMetrics] = []
        
    def record(self, metrics: ValidationMetrics):
        """Registrar una validación."""
        self.total_validations += 1
        self.total_validation_time_ms += metrics.validation_time_ms
        self.total_rows_validated += metrics.num_rows
        self.total_issues_found += metrics.num_issues
        
        if metrics.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Mantener solo últimas 100 validaciones
        self.validations_history.append(metrics)
        if len(self.validations_history) > 100:
            self.validations_history.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas acumuladas."""
        avg_time = (self.total_validation_time_ms / self.total_validations 
                   if self.total_validations > 0 else 0)
        
        cache_hit_rate = (self.cache_hits / self.total_validations * 100 
                         if self.total_validations > 0 else 0)
        
        avg_rows = (self.total_rows_validated / self.total_validations 
                   if self.total_validations > 0 else 0)
        
        # Calcular throughput (filas/segundo)
        total_time_seconds = self.total_validation_time_ms / 1000
        throughput = (self.total_rows_validated / total_time_seconds 
                     if total_time_seconds > 0 else 0)
        
        return {
            'total_validations': self.total_validations,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate_pct': round(cache_hit_rate, 1),
            'total_rows_validated': self.total_rows_validated,
            'total_issues_found': self.total_issues_found,
            'performance': {
                'avg_validation_time_ms': round(avg_time, 2),
                'total_validation_time_ms': round(self.total_validation_time_ms, 2),
                'avg_rows_per_validation': round(avg_rows, 0),
                'throughput_rows_per_second': round(throughput, 0)
            }
        }
    
    def get_recent_validations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Obtener últimas N validaciones."""
        recent = self.validations_history[-n:]
        return [v.to_dict() for v in recent]
    
    def log_summary(self):
        """Loggear resumen de métricas."""
        stats = self.get_stats()
        logger.info("=" * 60)
        logger.info("📊 VALIDATOR PERFORMANCE SUMMARY")
        logger.info(f"Total validations: {stats['total_validations']}")
        logger.info(f"Cache hit rate: {stats['cache_hit_rate_pct']}%")
        logger.info(f"Avg validation time: {stats['performance']['avg_validation_time_ms']}ms")
        logger.info(f"Throughput: {stats['performance']['throughput_rows_per_second']} rows/s")
        logger.info(f"Total issues found: {stats['total_issues_found']}")
        logger.info("=" * 60)


# Singleton global
_validator_tracker: ValidatorPerformanceTracker = None

def get_validator_tracker() -> ValidatorPerformanceTracker:
    """Obtener instancia global del tracker."""
    global _validator_tracker
    if _validator_tracker is None:
        _validator_tracker = ValidatorPerformanceTracker()
        logger.info("📊 Validator performance tracker initialized")
    return _validator_tracker
