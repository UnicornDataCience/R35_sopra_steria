# Fase 3: Optimización del Generador (Plan)

## 📋 Estado Actual del Generador

### Arquitectura
- **Agente principal**: `generator_agent.py` - Orquesta la generación
- **Generadores específicos**: 
  - `ctgan_generator.py` - CTGAN (GAN condicional)
  - `tvae_generator.py` - TVAE (Variational Autoencoder)
  - `sdv_generator.py` - SDV tradicional

### Funcionalidad Actual
- Selección automática de modelo por heurística
- Filtrado de columnas por `MedicalColumnSelector`
- Timeout de 10 minutos para generación
- Seeds reproducibles
- Normalización de datos legacy para COVID-19

### Problemas Identificados
1. **Sin caché de modelos**: Se entrena desde cero cada vez (5-10 min)
2. **Sin métricas**: No se mide calidad, tiempo, memoria
3. **Logging básico**: Falta trazabilidad detallada
4. **Sin early stopping**: Modelos pueden sobre-entrenar
5. **Sin paralelización**: Proceso secuencial bloqueante
6. **Sin memoización**: Configuraciones repetidas no se cachean
7. **Entrenamiento largo**: CTGAN/TVAE pueden tardar mucho
8. **Sin validación de calidad**: No se verifica similitud con datos originales

## 🎯 Objetivos de la Fase 3

### 1. Caché de Modelos Entrenados
- **Objetivo**: Reducir tiempo de generación de 5-10 min a < 30 seg en llamadas repetidas
- **Implementación**:
  - Hash del DataFrame + configuración como clave
  - Serialización de modelos entrenados (pickle/joblib)
  - TTL configurable (24h por defecto)
  - Almacenamiento en `temp_generations/model_cache/`
  - Invalidación automática si cambian los datos

### 2. Métricas de Generación
- **Métricas de rendimiento**:
  - Tiempo de entrenamiento del modelo
  - Tiempo de generación de samples
  - Memoria utilizada
  - Tamaño del modelo serializado
- **Métricas de calidad** (validación post-generación):
  - Similitud estadística (KL divergence, Wasserstein distance)
  - Correlaciones preservadas
  - Distribuciones marginales
  - Privacy metrics (distancia a registros reales)

### 3. Logging Estructurado
- **Eventos a logear**:
  - Inicio/fin de entrenamiento con timestamp
  - Hit/miss de caché
  - Parámetros de configuración usados
  - Métricas de calidad calculadas
  - Errores con stack trace completo
  - Warnings de convergencia

### 4. Early Stopping Inteligente
- **Criterios**:
  - Loss no mejora por N epochs
  - Tiempo máximo de entrenamiento
  - Calidad mínima alcanzada
- **Beneficios**:
  - Reduce sobre-entrenamiento
  - Ahorra tiempo en datasets complejos
  - Mejora reproducibilidad

### 5. Optimización de Performance
- **Paralelización**:
  - Procesamiento de batches en paralelo (si es posible)
  - Uso eficiente de ThreadPoolExecutor
- **Reducción de epochs**:
  - Ajustar epochs dinámicamente por tamaño de dataset
  - Usar early stopping como criterio principal
- **Optimización de memoria**:
  - Limpieza de objetos grandes tras uso
  - Streaming de datos grandes

### 6. Memoización de Configuraciones
- **Qué cachear**:
  - Metadatos detectados (SingleTableMetadata)
  - Configuraciones óptimas por tipo de dataset
  - Resultados de heurísticas de selección de modelo
- **Implementación**:
  - Decorador `@lru_cache` para funciones puras
  - Caché persistente para metadatos costosos

## 📐 Diseño de Implementación

### Estructura de Caché
```
temp_generations/
  model_cache/
    {hash}_{model_type}.pkl      # Modelo entrenado serializado
    {hash}_{model_type}_meta.json # Metadatos (config, timestamp, metrics)
    {hash}_metadata.pkl           # SingleTableMetadata
```

### Hash de Dataset
```python
def _compute_dataset_hash(df: pd.DataFrame, config: Dict) -> str:
    """Hash basado en: shape, columnas, dtypes, sample de datos, config"""
    components = [
        str(df.shape),
        str(sorted(df.columns.tolist())),
        str(sorted(df.dtypes.to_dict().items())),
        str(df.head(5).values.tobytes()),  # Muestra de datos
        str(sorted(config.items()))
    ]
    return hashlib.sha256('|'.join(components).encode()).hexdigest()[:16]
```

### Métricas de Calidad
```python
class QualityMetrics:
    - statistical_similarity: float  # 0-1, basado en KL divergence
    - correlation_preservation: float  # 0-1, Frobenius norm de diff matrices
    - distribution_fidelity: float  # 0-1, Kolmogorov-Smirnov
    - privacy_score: float  # 0-1, distancia mínima a registros reales
    - overall_quality: float  # 0-1, promedio ponderado
```

### Early Stopping
```python
class EarlyStoppingConfig:
    patience: int = 5  # Epochs sin mejora
    min_delta: float = 0.001  # Mejora mínima significativa
    max_time_seconds: int = 300  # 5 minutos máx por defecto
    target_quality: float = 0.85  # Calidad objetivo
```

## 🔄 Plan de Implementación

### Paso 1: Backup y Preparación
- [x] Backup de `generator_agent.py` y generadores específicos
- [ ] Crear `src/generation/model_cache.py` para gestión de caché
- [ ] Crear `src/generation/quality_metrics.py` para validación
- [ ] Crear `src/generation/early_stopping.py` para criterios de parada

### Paso 2: Implementar Caché de Modelos
- [ ] Implementar hash de datasets
- [ ] Implementar serialización/deserialización de modelos
- [ ] Integrar caché en `generator_agent.py`
- [ ] Añadir logs de hit/miss de caché
- [ ] Variable de entorno `GENERATOR_CACHE_ENABLED` (default: true)

### Paso 3: Añadir Métricas de Performance
- [ ] Timer de entrenamiento/generación
- [ ] Memory profiler
- [ ] Tamaño de modelo serializado
- [ ] Logs estructurados con métricas

### Paso 4: Implementar Métricas de Calidad
- [ ] Similitud estadística (KL, Wasserstein)
- [ ] Preservación de correlaciones
- [ ] Fidelidad de distribuciones
- [ ] Privacy score (distancia DCR)
- [ ] Integrar en pipeline post-generación

### Paso 5: Early Stopping
- [ ] Callback para monitoreo de loss
- [ ] Criterios configurables (patience, time, quality)
- [ ] Integrar en CTGAN y TVAE
- [ ] Logs de razón de parada

### Paso 6: Optimizaciones de Performance
- [ ] Ajuste dinámico de epochs
- [ ] Limpieza de memoria post-generación
- [ ] Optimización de ThreadPoolExecutor
- [ ] Memoización de metadatos

### Paso 7: Testing y Validación
- [ ] Test de caché (hit/miss)
- [ ] Test de métricas de calidad
- [ ] Test de early stopping
- [ ] Test de integración completa
- [ ] Validación con dataset grande (>10k rows)

### Paso 8: Documentación
- [ ] Actualizar docstrings
- [ ] Crear guía de configuración de caché
- [ ] Documentar métricas de calidad
- [ ] Crear `FASE3_GENERADOR_COMPLETADA.md`

## ⚙️ Variables de Entorno (Nuevas)

```bash
# Caché de modelos
GENERATOR_CACHE_ENABLED=true
GENERATOR_CACHE_DIR=temp_generations/model_cache
GENERATOR_CACHE_TTL_HOURS=24

# Early stopping
GENERATOR_EARLY_STOPPING=true
GENERATOR_PATIENCE=5
GENERATOR_MAX_TRAIN_TIME=300  # 5 min
GENERATOR_TARGET_QUALITY=0.85

# Performance
GENERATOR_DYNAMIC_EPOCHS=true
GENERATOR_MIN_EPOCHS=50
GENERATOR_MAX_EPOCHS=500

# Métricas
GENERATOR_COMPUTE_QUALITY_METRICS=true
GENERATOR_QUALITY_THRESHOLD=0.7  # Advertir si calidad < threshold
```

## 🎯 Criterios de Éxito

1. **Caché funcional**: 
   - Hit en 2da generación con mismos datos
   - Reducción de tiempo ≥ 80%
2. **Métricas completas**:
   - Tiempo, memoria, calidad en logs
   - Overall quality ≥ 0.7 en test dataset
3. **Early stopping efectivo**:
   - Para antes de max_epochs si converge
   - Ahorra ≥ 30% tiempo en promedio
4. **No regresiones**:
   - Todos los tests anteriores pasan
   - Calidad de datos generados ≥ baseline actual
5. **Documentación completa**:
   - Plan y resultados documentados
   - Guía de uso para nuevos parámetros

## 📝 Notas de Implementación

- Mantener compatibilidad con código existente
- Caché opcional (puede deshabilitarse)
- Métricas de calidad no deben bloquear generación si fallan
- Early stopping conservador (no ahorrar tiempo a costa de calidad)
- Logs estructurados para facilitar debugging
- Tests exhaustivos antes de integrar en orchestrator

## 🚀 Próximos Pasos

1. Crear módulos de soporte (`model_cache.py`, `quality_metrics.py`, `early_stopping.py`)
2. Modificar `generator_agent.py` para integrar caché
3. Actualizar generadores específicos (CTGAN, TVAE, SDV)
4. Implementar tests de validación
5. Documentar y validar con usuario

---

**Fecha de inicio**: [Pendiente]
**Responsable**: Copilot + Usuario
**Estado**: 📋 En planificación
