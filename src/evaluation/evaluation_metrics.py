"""
Métricas detalladas para evaluaciones de datos sintéticos.
Tracking de performance y calidad de evaluaciones.
"""
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class EvaluationMetrics:
    """Métricas de una evaluación individual."""
    
    # Identificación
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Tamaño de datasets
    original_rows: int = 0
    original_cols: int = 0
    synthetic_rows: int = 0
    synthetic_cols: int = 0
    
    # Scores principales
    final_quality_score: float = 0.0
    fidelity_score: float = 0.0
    ml_utility_score: float = 0.0
    privacy_score: float = 0.0
    quality_tier: str = "N/A"
    
    # Métricas detalladas
    correlation_preservation: float = 0.0
    distribution_similarity: float = 0.0
    unique_value_coverage: float = 0.0
    f1_preservation: float = 0.0
    
    # Performance
    total_time_ms: float = 0.0
    fidelity_time_ms: float = 0.0
    ml_utility_time_ms: float = 0.0
    privacy_time_ms: float = 0.0
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'timestamp': self.timestamp,
            'dataset_sizes': {
                'original': f"{self.original_rows}x{self.original_cols}",
                'synthetic': f"{self.synthetic_rows}x{self.synthetic_cols}"
            },
            'scores': {
                'final_quality': round(self.final_quality_score, 3),
                'fidelity': round(self.fidelity_score, 3),
                'ml_utility': round(self.ml_utility_score, 3),
                'privacy': round(self.privacy_score, 3),
                'tier': self.quality_tier
            },
            'detailed_metrics': {
                'correlation_preservation': round(self.correlation_preservation, 3),
                'distribution_similarity': round(self.distribution_similarity, 3),
                'unique_value_coverage': round(self.unique_value_coverage, 3),
                'f1_preservation': round(self.f1_preservation, 3)
            },
            'performance': {
                'total_time_ms': round(self.total_time_ms, 2),
                'fidelity_time_ms': round(self.fidelity_time_ms, 2),
                'ml_utility_time_ms': round(self.ml_utility_time_ms, 2),
                'privacy_time_ms': round(self.privacy_time_ms, 2),
                'cache_hit': self.cache_hit
            }
        }


class EvaluatorPerformanceTracker:
    """Tracker de performance del evaluador."""
    
    def __init__(self):
        self.total_evaluations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_evaluation_time_ms = 0.0
        self.total_rows_evaluated = 0
        self.evaluations_history: List[EvaluationMetrics] = []
        
        # Acumuladores de scores
        self.total_quality_score = 0.0
        self.total_fidelity_score = 0.0
        self.total_ml_utility_score = 0.0
        self.total_privacy_score = 0.0
        
    def record(self, metrics: EvaluationMetrics):
        """Registrar una evaluación."""
        self.total_evaluations += 1
        self.total_evaluation_time_ms += metrics.total_time_ms
        self.total_rows_evaluated += metrics.synthetic_rows
        
        # Acumular scores
        self.total_quality_score += metrics.final_quality_score
        self.total_fidelity_score += metrics.fidelity_score
        self.total_ml_utility_score += metrics.ml_utility_score
        self.total_privacy_score += metrics.privacy_score
        
        if metrics.cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Mantener solo últimas 50 evaluaciones
        self.evaluations_history.append(metrics)
        if len(self.evaluations_history) > 50:
            self.evaluations_history.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas acumuladas."""
        avg_time = (self.total_evaluation_time_ms / self.total_evaluations 
                   if self.total_evaluations > 0 else 0)
        
        cache_hit_rate = (self.cache_hits / self.total_evaluations * 100 
                         if self.total_evaluations > 0 else 0)
        
        avg_rows = (self.total_rows_evaluated / self.total_evaluations 
                   if self.total_evaluations > 0 else 0)
        
        # Calcular throughput (filas/segundo)
        total_time_seconds = self.total_evaluation_time_ms / 1000
        throughput = (self.total_rows_evaluated / total_time_seconds 
                     if total_time_seconds > 0 else 0)
        
        # Promedios de scores
        avg_quality = (self.total_quality_score / self.total_evaluations 
                      if self.total_evaluations > 0 else 0)
        avg_fidelity = (self.total_fidelity_score / self.total_evaluations 
                       if self.total_evaluations > 0 else 0)
        avg_ml_utility = (self.total_ml_utility_score / self.total_evaluations 
                         if self.total_evaluations > 0 else 0)
        avg_privacy = (self.total_privacy_score / self.total_evaluations 
                      if self.total_evaluations > 0 else 0)
        
        return {
            'total_evaluations': self.total_evaluations,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate_pct': round(cache_hit_rate, 1),
            'total_rows_evaluated': self.total_rows_evaluated,
            'average_scores': {
                'quality': round(avg_quality, 3),
                'fidelity': round(avg_fidelity, 3),
                'ml_utility': round(avg_ml_utility, 3),
                'privacy': round(avg_privacy, 3)
            },
            'performance': {
                'avg_evaluation_time_ms': round(avg_time, 2),
                'total_evaluation_time_ms': round(self.total_evaluation_time_ms, 2),
                'avg_rows_per_evaluation': round(avg_rows, 0),
                'throughput_rows_per_second': round(throughput, 0)
            }
        }
    
    def get_recent_evaluations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Obtener últimas N evaluaciones."""
        recent = self.evaluations_history[-n:]
        return [e.to_dict() for e in recent]
    
    def log_summary(self):
        """Loggear resumen de métricas."""
        stats = self.get_stats()
        logger.info("=" * 60)
        logger.info("📊 EVALUATOR PERFORMANCE SUMMARY")
        logger.info(f"Total evaluations: {stats['total_evaluations']}")
        logger.info(f"Cache hit rate: {stats['cache_hit_rate_pct']}%")
        logger.info(f"Avg quality score: {stats['average_scores']['quality']:.3f}")
        logger.info(f"Avg evaluation time: {stats['performance']['avg_evaluation_time_ms']}ms")
        logger.info(f"Throughput: {stats['performance']['throughput_rows_per_second']} rows/s")
        logger.info("=" * 60)


# Singleton global
_evaluator_tracker: EvaluatorPerformanceTracker = None

def get_evaluator_tracker() -> EvaluatorPerformanceTracker:
    """Obtener instancia global del tracker."""
    global _evaluator_tracker
    if _evaluator_tracker is None:
        _evaluator_tracker = EvaluatorPerformanceTracker()
        logger.info("📊 Evaluator performance tracker initialized")
    return _evaluator_tracker
