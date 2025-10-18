"""
Métricas de calidad para datos sintéticos generados.

Este módulo proporciona funcionalidades para evaluar:
- Similitud estadística entre datos reales y sintéticos
- Preservación de correlaciones
- Fidelidad de distribuciones marginales
- Privacy score (distancia a registros reales)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import cdist
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Estructura para almacenar métricas de calidad de datos sintéticos."""
    
    # Métricas principales (0-1, donde 1 es mejor)
    statistical_similarity: float  # Similitud estadística global
    correlation_preservation: float  # Preservación de correlaciones
    distribution_fidelity: float  # Fidelidad de distribuciones marginales
    privacy_score: float  # Privacy (distancia a registros reales)
    overall_quality: float  # Calidad global ponderada
    
    # Métricas detalladas
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        return {
            'statistical_similarity': float(self.statistical_similarity),
            'correlation_preservation': float(self.correlation_preservation),
            'distribution_fidelity': float(self.distribution_fidelity),
            'privacy_score': float(self.privacy_score),
            'overall_quality': float(self.overall_quality),
            'details': self.details
        }
    
    def __repr__(self) -> str:
        return (
            f"QualityMetrics(overall={self.overall_quality:.3f}, "
            f"stats={self.statistical_similarity:.3f}, "
            f"corr={self.correlation_preservation:.3f}, "
            f"dist={self.distribution_fidelity:.3f}, "
            f"privacy={self.privacy_score:.3f})"
        )


class QualityEvaluator:
    """Evaluador de calidad para datos sintéticos."""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        max_samples_for_privacy: int = 1000
    ):
        """
        Inicializa el evaluador.
        
        Args:
            weights: Pesos para cada métrica en el score global
            max_samples_for_privacy: Máximo de muestras para cálculo de privacy (performance)
        """
        self.weights = weights or {
            'statistical_similarity': 0.3,
            'correlation_preservation': 0.3,
            'distribution_fidelity': 0.25,
            'privacy_score': 0.15
        }
        self.max_samples_for_privacy = max_samples_for_privacy
        
        logger.info(f"QualityEvaluator inicializado con pesos: {self.weights}")
    
    def evaluate(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame
    ) -> QualityMetrics:
        """
        Evalúa la calidad de datos sintéticos comparados con reales.
        
        Args:
            real_df: DataFrame con datos reales
            synthetic_df: DataFrame con datos sintéticos
            
        Returns:
            QualityMetrics con todas las métricas calculadas
        """
        logger.info(f"Evaluando calidad: real_shape={real_df.shape}, synthetic_shape={synthetic_df.shape}")
        
        try:
            # Validaciones básicas
            if real_df.empty or synthetic_df.empty:
                logger.warning("DataFrame vacío detectado, retornando métricas por defecto")
                return self._default_metrics("Empty DataFrame")
            
            # Alinear columnas (usar solo columnas comunes)
            common_cols = list(set(real_df.columns) & set(synthetic_df.columns))
            if not common_cols:
                logger.warning("No hay columnas comunes entre datasets")
                return self._default_metrics("No common columns")
            
            real_aligned = real_df[common_cols].copy()
            synthetic_aligned = synthetic_df[common_cols].copy()
            
            # Excluir columnas ID/string que no son útiles para métricas numéricas
            # Identificar columnas ID por patrón de nombre o por valores únicos
            id_cols = []
            for col in real_aligned.columns:
                if 'id' in col.lower() or 'patient' in col.lower():
                    id_cols.append(col)
                elif real_aligned[col].dtype == 'object' and real_aligned[col].nunique() > len(real_aligned) * 0.9:
                    # Columna con >90% valores únicos, probablemente ID
                    id_cols.append(col)
            
            if id_cols:
                logger.debug(f"Excluyendo columnas ID de métricas: {id_cols}")
                real_aligned = real_aligned.drop(columns=id_cols)
                synthetic_aligned = synthetic_aligned.drop(columns=id_cols)
            
            # Calcular cada métrica
            details = {}
            
            # 1. Similitud estadística (basada en KL divergence)
            stat_sim, stat_details = self._compute_statistical_similarity(real_aligned, synthetic_aligned)
            details['statistical'] = stat_details
            
            # 2. Preservación de correlaciones
            corr_pres, corr_details = self._compute_correlation_preservation(real_aligned, synthetic_aligned)
            details['correlation'] = corr_details
            
            # 3. Fidelidad de distribuciones (Kolmogorov-Smirnov)
            dist_fid, dist_details = self._compute_distribution_fidelity(real_aligned, synthetic_aligned)
            details['distribution'] = dist_details
            
            # 4. Privacy score (distancia a registros reales)
            privacy, privacy_details = self._compute_privacy_score(real_aligned, synthetic_aligned)
            details['privacy'] = privacy_details
            
            # 5. Score global ponderado
            overall = (
                stat_sim * self.weights['statistical_similarity'] +
                corr_pres * self.weights['correlation_preservation'] +
                dist_fid * self.weights['distribution_fidelity'] +
                privacy * self.weights['privacy_score']
            )
            
            metrics = QualityMetrics(
                statistical_similarity=stat_sim,
                correlation_preservation=corr_pres,
                distribution_fidelity=dist_fid,
                privacy_score=privacy,
                overall_quality=overall,
                details=details
            )
            
            logger.info(f"Métricas calculadas: {metrics}")
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculando métricas de calidad: {e}", exc_info=True)
            return self._default_metrics(f"Error: {str(e)}")
    
    def _default_metrics(self, reason: str) -> QualityMetrics:
        """Retorna métricas por defecto en caso de error."""
        return QualityMetrics(
            statistical_similarity=0.0,
            correlation_preservation=0.0,
            distribution_fidelity=0.0,
            privacy_score=1.0,  # Conservador: asumimos privacidad perfecta
            overall_quality=0.0,
            details={'error': reason}
        )
    
    def _compute_statistical_similarity(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame
    ) -> tuple[float, Dict[str, Any]]:
        """
        Calcula similitud estadística basada en KL divergence.
        
        Returns:
            (score, details) donde score está en [0, 1] (1 = más similar)
        """
        try:
            numeric_cols = real_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return 1.0, {'message': 'No numeric columns'}
            
            kl_divergences = []
            
            for col in numeric_cols:
                real_vals = real_df[col].dropna().values
                synth_vals = synthetic_df[col].dropna().values
                
                if len(real_vals) < 10 or len(synth_vals) < 10:
                    continue
                
                # Histogramas con bins automáticos
                bins = min(30, len(real_vals) // 10)
                hist_range = (
                    min(real_vals.min(), synth_vals.min()),
                    max(real_vals.max(), synth_vals.max())
                )
                
                real_hist, _ = np.histogram(real_vals, bins=bins, range=hist_range, density=True)
                synth_hist, _ = np.histogram(synth_vals, bins=bins, range=hist_range, density=True)
                
                # Añadir pequeño epsilon para evitar división por 0
                real_hist = real_hist + 1e-10
                synth_hist = synth_hist + 1e-10
                
                # Normalizar
                real_hist = real_hist / real_hist.sum()
                synth_hist = synth_hist / synth_hist.sum()
                
                # KL divergence
                kl_div = np.sum(real_hist * np.log(real_hist / synth_hist))
                kl_divergences.append(kl_div)
            
            if not kl_divergences:
                return 1.0, {'message': 'Insufficient data'}
            
            # Convertir KL divergence a score [0, 1]
            # KL=0 -> score=1, KL>1 -> score~0
            avg_kl = np.mean(kl_divergences)
            score = np.exp(-avg_kl)  # Mapeo exponencial
            
            details = {
                'avg_kl_divergence': float(avg_kl),
                'num_columns_evaluated': len(kl_divergences),
                'kl_by_column': {col: float(kl) for col, kl in zip(numeric_cols[:len(kl_divergences)], kl_divergences)}
            }
            
            return float(score), details
        
        except Exception as e:
            logger.warning(f"Error en statistical_similarity: {e}")
            return 0.5, {'error': str(e)}
    
    def _compute_correlation_preservation(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame
    ) -> tuple[float, Dict[str, Any]]:
        """
        Calcula preservación de correlaciones (Frobenius norm).
        
        Returns:
            (score, details) donde score está en [0, 1] (1 = mejor preservación)
        """
        try:
            numeric_cols = list(real_df.select_dtypes(include=[np.number]).columns)
            
            if len(numeric_cols) < 2:
                return 1.0, {'message': 'Insufficient numeric columns for correlation'}
            
            # Matrices de correlación
            real_corr = real_df[numeric_cols].corr()
            synth_corr = synthetic_df[numeric_cols].corr()
            
            # Manejar NaN
            real_corr = real_corr.fillna(0)
            synth_corr = synth_corr.fillna(0)
            
            # Frobenius norm de la diferencia
            diff_matrix = real_corr - synth_corr
            frobenius_norm = np.linalg.norm(diff_matrix.values, 'fro')
            
            # Normalizar por tamaño de matriz
            max_possible_norm = np.sqrt(2 * len(numeric_cols) ** 2)  # Max teórico
            normalized_diff = frobenius_norm / max_possible_norm
            
            # Convertir a score [0, 1]
            score = 1.0 - min(normalized_diff, 1.0)
            
            details = {
                'frobenius_norm': float(frobenius_norm),
                'normalized_diff': float(normalized_diff),
                'num_columns': len(numeric_cols),
                'avg_abs_correlation_diff': float(np.abs(diff_matrix.values).mean())
            }
            
            return float(score), details
        
        except Exception as e:
            logger.warning(f"Error en correlation_preservation: {e}")
            return 0.5, {'error': str(e)}
    
    def _compute_distribution_fidelity(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame
    ) -> tuple[float, Dict[str, Any]]:
        """
        Calcula fidelidad de distribuciones usando Kolmogorov-Smirnov.
        
        Returns:
            (score, details) donde score está en [0, 1] (1 = distribuciones idénticas)
        """
        try:
            numeric_cols = real_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return 1.0, {'message': 'No numeric columns'}
            
            ks_statistics = []
            p_values = []
            columns_passed = []
            
            for col in numeric_cols:
                real_vals = real_df[col].dropna().values
                synth_vals = synthetic_df[col].dropna().values
                
                if len(real_vals) < 10 or len(synth_vals) < 10:
                    continue
                
                # Test Kolmogorov-Smirnov
                ks_stat, p_value = stats.ks_2samp(real_vals, synth_vals)
                ks_statistics.append(ks_stat)
                p_values.append(p_value)
                
                # Logging mejorado por columna
                if p_value > 0.05:
                    columns_passed.append(col)
                    logger.debug(f"✅ KS Test [{col}]: ks={ks_stat:.4f}, p={p_value:.4f} (similar)")
                else:
                    logger.debug(f"⚠️ KS Test [{col}]: ks={ks_stat:.4f}, p={p_value:.4f} (diferente)")
            
            if not ks_statistics:
                return 1.0, {'message': 'Insufficient data'}
            
            # KS statistic: 0 = idénticas, 1 = completamente diferentes
            avg_ks = np.mean(ks_statistics)
            score = 1.0 - avg_ks
            
            # Logging del resumen
            logger.info(f"📊 KS Test Summary: {len(columns_passed)}/{len(ks_statistics)} columnas pasan (p>0.05)")
            logger.info(f"📊 Distribution Fidelity Score: {score:.3f} (avg_ks={avg_ks:.4f})")
            
            details = {
                'avg_ks_statistic': float(avg_ks),
                'avg_p_value': float(np.mean(p_values)),
                'num_columns_evaluated': len(ks_statistics),
                'columns_similar_at_5pct': int(sum(p > 0.05 for p in p_values)),
                'pass_rate': len(columns_passed) / len(ks_statistics) if ks_statistics else 0
            }
            
            return float(score), details
        
        except Exception as e:
            logger.warning(f"Error en distribution_fidelity: {e}")
            return 0.5, {'error': str(e)}
    
    def _compute_privacy_score(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame
    ) -> tuple[float, Dict[str, Any]]:
        """
        Calcula privacy score basado en distancia mínima a registros reales.
        Score alto = mayor privacy (datos sintéticos menos similares a individuos reales).
        
        Returns:
            (score, details) donde score está en [0, 1] (1 = máxima privacy)
        """
        try:
            # Usar solo columnas numéricas para cálculo de distancia
            numeric_cols = list(real_df.select_dtypes(include=[np.number]).columns)
            
            if len(numeric_cols) == 0:
                return 1.0, {'message': 'No numeric columns for privacy calculation'}
            
            # Submuestrear para performance
            real_sample = real_df[numeric_cols].sample(
                n=min(self.max_samples_for_privacy, len(real_df)),
                random_state=42
            ).values
            
            synth_sample = synthetic_df[numeric_cols].sample(
                n=min(self.max_samples_for_privacy, len(synthetic_df)),
                random_state=42
            ).values
            
            # Normalizar datos (Z-score)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            real_scaled = scaler.fit_transform(real_sample)
            synth_scaled = scaler.transform(synth_sample)
            
            # Calcular distancias mínimas (cada sintético -> real más cercano)
            # Usar euclidean distance
            distances = cdist(synth_scaled, real_scaled, metric='euclidean')
            min_distances = distances.min(axis=1)
            
            # Métrica: promedio de distancias mínimas
            avg_min_distance = np.mean(min_distances)
            
            # Convertir a score [0, 1]
            # Distancia pequeña -> baja privacy (score bajo)
            # Distancia grande -> alta privacy (score alto)
            # Usamos función sigmoidal para mapear
            score = 1.0 / (1.0 + np.exp(-avg_min_distance + 2))  # Shifted sigmoid
            
            # Calcular DCR (Distance to Closest Record) percentiles
            dcr_percentiles = np.percentile(min_distances, [5, 50, 95])
            
            details = {
                'avg_min_distance': float(avg_min_distance),
                'dcr_5th_percentile': float(dcr_percentiles[0]),
                'dcr_median': float(dcr_percentiles[1]),
                'dcr_95th_percentile': float(dcr_percentiles[2]),
                'samples_evaluated': len(synth_sample)
            }
            
            return float(score), details
        
        except Exception as e:
            logger.warning(f"Error en privacy_score: {e}")
            return 0.8, {'error': str(e)}  # Conservador


# Instancia global del evaluador (singleton)
_evaluator_instance: Optional[QualityEvaluator] = None


def get_quality_evaluator() -> QualityEvaluator:
    """Retorna la instancia singleton del evaluador."""
    global _evaluator_instance
    
    if _evaluator_instance is None:
        _evaluator_instance = QualityEvaluator()
    
    return _evaluator_instance
