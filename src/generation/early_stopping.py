"""
Early stopping para entrenamiento de modelos generativos.

Este módulo proporciona criterios de parada temprana para:
- Detectar convergencia (loss no mejora)
- Limitar tiempo de entrenamiento
- Alcanzar calidad objetivo
"""

import time
import os
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Configuración de early stopping."""
    
    patience: int = 5  # Epochs sin mejora antes de parar
    min_delta: float = 0.001  # Mejora mínima significativa en loss
    max_time_seconds: Optional[int] = 300  # Tiempo máximo de entrenamiento (5 min)
    target_quality: Optional[float] = None  # Calidad objetivo (si aplica)
    restore_best_weights: bool = True  # Restaurar mejores pesos al parar
    
    @classmethod
    def from_env(cls) -> 'EarlyStoppingConfig':
        """Crea configuración desde variables de entorno."""
        enabled = os.getenv('GENERATOR_EARLY_STOPPING', 'true').lower() == 'true'
        
        if not enabled:
            return cls(patience=999999, max_time_seconds=None)  # Deshabilitado
        
        return cls(
            patience=int(os.getenv('GENERATOR_PATIENCE', '5')),
            min_delta=float(os.getenv('GENERATOR_MIN_DELTA', '0.001')),
            max_time_seconds=int(os.getenv('GENERATOR_MAX_TRAIN_TIME', '300')),
            target_quality=float(q) if (q := os.getenv('GENERATOR_TARGET_QUALITY')) else None
        )


class EarlyStoppingMonitor:
    """Monitor de early stopping para entrenamiento."""
    
    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        """
        Inicializa el monitor.
        
        Args:
            config: Configuración de early stopping (usa defaults si es None)
        """
        self.config = config or EarlyStoppingConfig.from_env()
        
        # Estado interno
        self.best_loss: Optional[float] = None
        self.best_epoch: int = 0
        self.wait: int = 0  # Epochs sin mejora
        self.stopped_epoch: int = 0
        self.start_time: float = time.time()
        self.should_stop: bool = False
        self.stop_reason: Optional[str] = None
        
        # Métricas de progreso
        self.loss_history: list = []
        self.epoch_times: list = []
        
        logger.info(
            f"EarlyStoppingMonitor inicializado: "
            f"patience={self.config.patience}, "
            f"min_delta={self.config.min_delta}, "
            f"max_time={self.config.max_time_seconds}s, "
            f"target_quality={self.config.target_quality}"
        )
    
    def on_epoch_end(
        self,
        epoch: int,
        loss: float,
        quality: Optional[float] = None
    ) -> bool:
        """
        Callback al final de cada epoch.
        
        Args:
            epoch: Número de epoch actual
            loss: Loss del epoch
            quality: Calidad del modelo (si está disponible)
            
        Returns:
            True si debe continuar entrenamiento, False si debe parar
        """
        epoch_time = time.time() - self.start_time
        self.loss_history.append(loss)
        self.epoch_times.append(epoch_time)
        
        # Verificar tiempo máximo
        if self.config.max_time_seconds and epoch_time > self.config.max_time_seconds:
            self.should_stop = True
            self.stopped_epoch = epoch
            self.stop_reason = f"Timeout: {epoch_time:.1f}s > {self.config.max_time_seconds}s"
            logger.warning(f"⏱️ {self.stop_reason}")
            return False
        
        # Verificar calidad objetivo (si aplica)
        if quality is not None and self.config.target_quality:
            if quality >= self.config.target_quality:
                self.should_stop = True
                self.stopped_epoch = epoch
                self.stop_reason = f"Target quality reached: {quality:.3f} >= {self.config.target_quality:.3f}"
                logger.info(f"🎯 {self.stop_reason}")
                return False
        
        # Verificar mejora de loss
        if self.best_loss is None:
            self.best_loss = loss
            self.best_epoch = epoch
            logger.debug(f"Epoch {epoch}: loss={loss:.4f} (inicial)")
            return True
        
        # ¿Hay mejora significativa?
        if loss < (self.best_loss - self.config.min_delta):
            self.best_loss = loss
            self.best_epoch = epoch
            self.wait = 0
            logger.debug(f"Epoch {epoch}: loss={loss:.4f} (mejora, wait=0)")
            return True
        
        # No hay mejora
        self.wait += 1
        logger.debug(f"Epoch {epoch}: loss={loss:.4f} (sin mejora, wait={self.wait}/{self.config.patience})")
        
        # ¿Alcanzó patience?
        if self.wait >= self.config.patience:
            self.should_stop = True
            self.stopped_epoch = epoch
            self.stop_reason = (
                f"No improvement for {self.config.patience} epochs "
                f"(best_loss={self.best_loss:.4f} at epoch {self.best_epoch})"
            )
            logger.info(f"⏹️ {self.stop_reason}")
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del entrenamiento."""
        elapsed_time = time.time() - self.start_time
        
        return {
            'stopped_early': self.should_stop,
            'stop_reason': self.stop_reason,
            'stopped_epoch': self.stopped_epoch if self.should_stop else None,
            'best_epoch': self.best_epoch,
            'best_loss': float(self.best_loss) if self.best_loss else None,
            'total_epochs': len(self.loss_history),
            'elapsed_time_seconds': elapsed_time,
            'avg_epoch_time': elapsed_time / len(self.epoch_times) if self.epoch_times else 0,
            'loss_history': [float(l) for l in self.loss_history]
        }
    
    def reset(self):
        """Reinicia el estado del monitor."""
        self.best_loss = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.start_time = time.time()
        self.should_stop = False
        self.stop_reason = None
        self.loss_history = []
        self.epoch_times = []
        logger.debug("EarlyStoppingMonitor reiniciado")


class DynamicEpochsCalculator:
    """Calcula número óptimo de epochs basado en dataset."""
    
    @staticmethod
    def calculate(
        n_rows: int,
        n_cols: int,
        model_type: str
    ) -> int:
        """
        Calcula epochs óptimos para un dataset y modelo.
        
        Args:
            n_rows: Número de filas
            n_cols: Número de columnas
            model_type: Tipo de modelo ('ctgan', 'tvae', 'sdv')
            
        Returns:
            Número de epochs recomendado
        """
        min_epochs = int(os.getenv('GENERATOR_MIN_EPOCHS', '50'))
        max_epochs = int(os.getenv('GENERATOR_MAX_EPOCHS', '500'))
        
        # Heurísticas por tipo de modelo
        if model_type == 'sdv':
            # SDV es más simple, necesita menos epochs
            base_epochs = 50
        elif model_type == 'tvae':
            # TVAE converge relativamente rápido
            base_epochs = 100
        elif model_type == 'ctgan':
            # CTGAN necesita más epochs para estabilizar
            base_epochs = 200
        else:
            base_epochs = 100
        
        # Ajustar por tamaño de dataset
        # Más datos -> necesita más epochs (pero con límite)
        size_factor = 1.0 + (n_rows / 5000)  # +1 epoch por cada 5000 filas
        size_factor = min(size_factor, 2.0)  # Max 2x del base
        
        # Ajustar por número de columnas
        # Más columnas -> más complejo -> más epochs
        col_factor = 1.0 + (n_cols / 50)  # +1 epoch por cada 50 columnas
        col_factor = min(col_factor, 1.5)  # Max 1.5x del base
        
        # Calcular epochs finales
        calculated_epochs = int(base_epochs * size_factor * col_factor)
        
        # Aplicar límites
        final_epochs = max(min_epochs, min(calculated_epochs, max_epochs))
        
        logger.info(
            f"Epochs calculados para {model_type}: "
            f"base={base_epochs}, size_factor={size_factor:.2f}, "
            f"col_factor={col_factor:.2f}, final={final_epochs}"
        )
        
        return final_epochs


def get_optimal_batch_size(n_rows: int, model_type: str) -> int:
    """
    Calcula batch size óptimo basado en dataset.
    
    Args:
        n_rows: Número de filas
        model_type: Tipo de modelo
        
    Returns:
        Batch size recomendado
    """
    # Batch size típicos por modelo
    default_batches = {
        'ctgan': 500,
        'tvae': 500,
        'sdv': 100
    }
    
    base_batch = default_batches.get(model_type, 500)
    
    # Ajustar por tamaño de dataset
    if n_rows < 500:
        # Dataset pequeño -> batch pequeño
        return min(base_batch, max(32, n_rows // 4))
    elif n_rows < 2000:
        # Dataset mediano -> batch mediano
        return min(base_batch, 256)
    else:
        # Dataset grande -> usar batch size por defecto
        return base_batch


def create_progress_callback(
    monitor: EarlyStoppingMonitor,
    log_interval: int = 10
) -> Callable[[int, float], bool]:
    """
    Crea un callback de progreso compatible con SDV/PyTorch.
    
    Args:
        monitor: Monitor de early stopping
        log_interval: Intervalo de epochs para logging detallado
        
    Returns:
        Función callback(epoch, loss) -> should_continue
    """
    def callback(epoch: int, loss: float) -> bool:
        # Log detallado cada N epochs
        if epoch % log_interval == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}: loss={loss:.4f}")
        
        # Delegar a monitor
        should_continue = monitor.on_epoch_end(epoch, loss)
        
        return should_continue
    
    return callback
