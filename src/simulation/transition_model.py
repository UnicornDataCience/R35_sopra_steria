"""
Modelo de transición aprendido para simulaciones realistas
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TransitionStats:
    """Estadísticas de transición para un parámetro clínico"""
    mean_change: float
    std_change: float
    min_value: float
    max_value: float
    improvement_rate: float
    deterioration_rate: float


class LearnedTransitionModel:
    """
    Modelo que aprende patrones de transición de datos reales
    para generar simulaciones más realistas
    """
    
    def __init__(self, cache_dir: str = "cache/simulation"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transitions: Dict[str, TransitionStats] = {}
        self.disease_type: Optional[str] = None
        self.is_fitted = False
        
    def _get_cache_key(self, data: pd.DataFrame, disease_type: str) -> str:
        """Genera clave única para cachear el modelo"""
        # Usar hash de columnas + tamaño + disease_type
        cols_str = "_".join(sorted(data.columns))
        key_str = f"{cols_str}_{len(data)}_{disease_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Ruta del archivo de cache"""
        return self.cache_dir / f"transition_model_{cache_key}.json"
    
    def fit(self, data: pd.DataFrame, disease_type: str = "general", force_refit: bool = False) -> None:
        """
        Aprende patrones de transición de los datos reales
        
        Args:
            data: DataFrame con datos históricos de pacientes
            disease_type: Tipo de enfermedad para contexto
            force_refit: Forzar reentrenamiento incluso si existe cache
        """
        cache_key = self._get_cache_key(data, disease_type)
        cache_path = self._get_cache_path(cache_key)
        
        # Intentar cargar desde cache
        if not force_refit and cache_path.exists():
            logger.info(f"📦 Cargando modelo de transición desde cache: {cache_key}")
            try:
                self._load_from_cache(cache_path)
                return
            except Exception as e:
                logger.warning(f"⚠️ Error cargando cache, reentrenando: {e}")
        
        logger.info(f"🎓 Entrenando modelo de transición para {disease_type} con {len(data)} registros")
        self.disease_type = disease_type
        
        # Analizar parámetros clave según el tipo de enfermedad
        if disease_type == "covid19":
            params_to_learn = {
                'oxygen_saturation': ('oxygen_saturation', 'SAT_O2'),
                'temperature': ('temperature', 'TEMP'),
                'pcr_result': ('pcr_result', 'PCR')
            }
        else:
            # Para otros tipos, detectar automáticamente columnas numéricas
            params_to_learn = {col: (col, col) for col in data.select_dtypes(include=[np.number]).columns}
        
        for param_name, (col_name, _) in params_to_learn.items():
            if col_name not in data.columns:
                continue
                
            values = data[col_name].dropna()
            if len(values) < 2:
                continue
            
            # Calcular estadísticas de cambio (si hay datos temporales)
            # Por ahora, usar estadísticas agregadas
            mean_val = values.mean()
            std_val = values.std()
            min_val = values.min()
            max_val = values.max()
            
            # Estimar tasas de mejora/deterioro basadas en percentiles
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)
            
            # Para parámetros donde mayor es mejor (ej: saturación)
            if param_name in ['oxygen_saturation', 'SAT_O2']:
                improvement_rate = 0.7 if mean_val > 92 else 0.4
                deterioration_rate = 0.3 if mean_val > 92 else 0.6
            # Para parámetros donde menor es mejor (ej: PCR, temperatura)
            elif param_name in ['pcr_result', 'PCR', 'temperature', 'TEMP']:
                improvement_rate = 0.6 if mean_val < 10 else 0.3
                deterioration_rate = 0.4 if mean_val < 10 else 0.7
            else:
                improvement_rate = 0.5
                deterioration_rate = 0.5
            
            # Estimar cambio promedio basado en variabilidad
            mean_change = std_val * 0.1  # Cambio gradual
            std_change = std_val * 0.2
            
            self.transitions[param_name] = TransitionStats(
                mean_change=float(mean_change),
                std_change=float(std_change),
                min_value=float(min_val),
                max_value=float(max_val),
                improvement_rate=float(improvement_rate),
                deterioration_rate=float(deterioration_rate)
            )
            
            logger.debug(f"  ✓ {param_name}: mean_Δ={mean_change:.2f}, std_Δ={std_change:.2f}")
        
        self.is_fitted = True
        
        # Guardar en cache
        self._save_to_cache(cache_path)
        logger.info(f"💾 Modelo guardado en cache: {cache_key}")
    
    def predict_next_value(
        self,
        current_value: float,
        param_name: str,
        visit_number: int,
        severity: str = "moderate"
    ) -> Tuple[float, bool]:
        """
        Predice el siguiente valor basado en el modelo aprendido
        
        Args:
            current_value: Valor actual del parámetro
            param_name: Nombre del parámetro
            visit_number: Número de visita (para modelar progresión temporal)
            severity: Severidad del caso ("mild", "moderate", "severe")
            
        Returns:
            Tuple[nuevo_valor, es_mejora]
        """
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado. Llama a fit() primero.")
        
        if param_name not in self.transitions:
            # Si no conocemos el parámetro, hacer cambio aleatorio pequeño
            logger.warning(f"⚠️ Parámetro {param_name} no encontrado en modelo, usando fallback")
            change = np.random.normal(0, abs(current_value) * 0.05)
            new_value = current_value + change
            return new_value, change < 0  # Asumimos que menor es mejor por defecto
        
        stats = self.transitions[param_name]
        
        # Determinar si hay mejora o deterioro basado en probabilidades aprendidas
        # La probabilidad de mejora aumenta con el tiempo (generalmente)
        time_factor = min(1.0, visit_number / 10.0)  # Normalizar a [0, 1]
        
        # Ajustar probabilidades según severidad
        severity_multipliers = {
            "mild": 1.2,      # Más probabilidad de mejora
            "moderate": 1.0,
            "severe": 0.7     # Menos probabilidad de mejora
        }
        multiplier = severity_multipliers.get(severity, 1.0)
        
        improvement_prob = stats.improvement_rate * time_factor * multiplier
        is_improvement = np.random.random() < improvement_prob
        
        # Generar cambio basado en estadísticas aprendidas
        if is_improvement:
            # Cambio positivo (hacia mejora)
            if param_name in ['oxygen_saturation', 'SAT_O2']:
                # Mayor saturación = mejor
                change = abs(np.random.normal(stats.mean_change, stats.std_change))
            elif param_name in ['pcr_result', 'PCR', 'temperature', 'TEMP']:
                # Menor valor = mejor
                change = -abs(np.random.normal(stats.mean_change, stats.std_change))
            else:
                change = np.random.normal(stats.mean_change, stats.std_change)
        else:
            # Deterioro
            if param_name in ['oxygen_saturation', 'SAT_O2']:
                change = -abs(np.random.normal(stats.mean_change, stats.std_change))
            elif param_name in ['pcr_result', 'PCR', 'temperature', 'TEMP']:
                change = abs(np.random.normal(stats.mean_change, stats.std_change))
            else:
                change = -np.random.normal(stats.mean_change, stats.std_change)
        
        # Aplicar cambio con límites
        new_value = current_value + change
        new_value = max(stats.min_value, min(stats.max_value, new_value))
        
        return float(new_value), is_improvement
    
    def _save_to_cache(self, cache_path: Path) -> None:
        """Guarda el modelo en cache"""
        data = {
            'disease_type': self.disease_type,
            'transitions': {
                name: {
                    'mean_change': stats.mean_change,
                    'std_change': stats.std_change,
                    'min_value': stats.min_value,
                    'max_value': stats.max_value,
                    'improvement_rate': stats.improvement_rate,
                    'deterioration_rate': stats.deterioration_rate
                }
                for name, stats in self.transitions.items()
            }
        }
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_from_cache(self, cache_path: Path) -> None:
        """Carga el modelo desde cache"""
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        self.disease_type = data['disease_type']
        self.transitions = {
            name: TransitionStats(**stats_dict)
            for name, stats_dict in data['transitions'].items()
        }
        self.is_fitted = True
        logger.info(f"✅ Modelo cargado con {len(self.transitions)} parámetros")
