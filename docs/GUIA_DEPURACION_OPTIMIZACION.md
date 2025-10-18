# 🔬 Guía de Depuración y Optimización para Investigación

> Recomendaciones técnicas para optimizar experimentos y garantizar rigor científico  
> Última actualización: 15 de Octubre, 2025

---

## 📋 Índice

1. [Depuración de Resultados](#depuración-de-resultados)
2. [Optimización de Métodos](#optimización-de-métodos)
3. [Garantía de Reproducibilidad](#garantía-de-reproducibilidad)
4. [Métricas y Evaluación](#métricas-y-evaluación)
5. [Validación Científica](#validación-científica)
6. [Checklist de Experimentos](#checklist-de-experimentos)

---

## 🐛 Depuración de Resultados

### 1. **Logging Exhaustivo**

#### Configuración Recomendada

```python
# src/utils/logging_config.py
LOG_LEVEL = "DEBUG"  # Para investigación, usar DEBUG
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/experiment_{timestamp}.log"
```

#### Qué Loggear

**Para cada experimento**:
```python
logger.info("=" * 80)
logger.info("INICIO DE EXPERIMENTO")
logger.info("Timestamp: %s", datetime.now().isoformat())
logger.info("Dataset: %s (hash: %s)", dataset_name, dataset_hash)
logger.info("Parámetros: %s", json.dumps(params, indent=2))
logger.info("Seed: %s", seed)
logger.info("=" * 80)
```

**Durante el procesamiento**:
```python
# En cada agente
logger.debug("Input shape: %s", df.shape)
logger.debug("Columns: %s", df.columns.tolist())
logger.debug("Memory usage: %.2f MB", df.memory_usage(deep=True).sum() / 1e6)
logger.debug("Processing time: %.2f seconds", elapsed_time)
```

**Al finalizar**:
```python
logger.info("Resultados: %s", json.dumps(results, indent=2))
logger.info("Métricas: %s", json.dumps(metrics, indent=2))
logger.info("Tiempo total: %.2f segundos", total_time)
logger.info("=" * 80)
```

#### Análisis de Logs

```bash
# Filtrar por nivel de error
cat logs/experiment_*.log | grep "ERROR"

# Buscar métricas específicas
cat logs/experiment_*.log | grep "F1 Score"

# Extraer tiempos de ejecución
cat logs/experiment_*.log | grep "Processing time"
```

---

### 2. **Validación de Entrada/Salida**

#### Assertions de Calidad

```python
def validate_dataframe(df: pd.DataFrame, stage: str):
    """Validar DataFrame en cada etapa del pipeline"""
    assert not df.empty, f"{stage}: DataFrame vacío"
    assert len(df) > 10, f"{stage}: Muy pocos registros ({len(df)})"
    assert not df.columns.empty, f"{stage}: Sin columnas"
    
    # Verificar tipos de datos
    for col in df.columns:
        dtype = df[col].dtype
        logger.debug(f"{stage} - {col}: {dtype} (nulls: {df[col].isna().sum()})")
    
    # Verificar valores nulos excesivos
    null_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
    assert null_pct < 0.5, f"{stage}: Demasiados nulos ({null_pct:.1%})"
    
    logger.info(f"{stage}: Validación OK - Shape: {df.shape}, Nulls: {null_pct:.1%}")
```

#### Uso en Pipeline

```python
# Después de cargar dataset
validate_dataframe(df_original, "LOAD")

# Después de análisis
validate_dataframe(df_analyzed, "ANALYSIS")

# Después de generación
validate_dataframe(df_synthetic, "GENERATION")
```

---

### 3. **Comparación con Baselines**

#### Guardar Resultados de Baseline

```python
# experiments/baselines.json
{
    "dataset_covid_original": {
        "timestamp": "2025-10-15T20:00:00",
        "metrics": {
            "rows": 1000,
            "columns": 25,
            "nulls_pct": 0.05,
            "mean_age": 45.3,
            "std_age": 15.2
        }
    }
}
```

#### Comparar con Experimento Actual

```python
def compare_with_baseline(current_metrics, baseline_metrics):
    """Comparar métricas actuales con baseline"""
    diff = {}
    for key in baseline_metrics:
        if key in current_metrics:
            baseline_val = baseline_metrics[key]
            current_val = current_metrics[key]
            
            if isinstance(baseline_val, (int, float)):
                pct_diff = ((current_val - baseline_val) / baseline_val) * 100
                diff[key] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "diff_pct": pct_diff
                }
                
                # Log si hay diferencias significativas (> 10%)
                if abs(pct_diff) > 10:
                    logger.warning(f"⚠️ {key}: {pct_diff:+.1f}% vs baseline")
    
    return diff
```

---

## ⚙️ Optimización de Métodos

### 1. **Generación de Datos Sintéticos**

#### A. Hyperparameter Tuning para CTGAN

**Parámetros a optimizar**:
```python
param_grid = {
    'epochs': [100, 200, 300, 500],
    'batch_size': [250, 500, 1000],
    'generator_dim': [(128, 128), (256, 256), (512, 512)],
    'discriminator_dim': [(128, 128), (256, 256), (512, 512)],
    'generator_lr': [1e-4, 2e-4, 5e-4],
    'discriminator_lr': [1e-4, 2e-4, 5e-4]
}
```

**Grid Search**:
```python
from sklearn.model_selection import ParameterGrid

best_score = 0
best_params = None

for params in ParameterGrid(param_grid):
    logger.info(f"Testing params: {params}")
    
    # Entrenar CTGAN con estos parámetros
    generator = CTGANGenerator(**params)
    synthetic_data = generator.fit_generate(original_data, num_samples=500)
    
    # Evaluar calidad
    score = evaluate_quality(original_data, synthetic_data)
    logger.info(f"Score: {score:.3f}")
    
    if score > best_score:
        best_score = score
        best_params = params
        logger.info(f"✅ New best: {best_params}, score: {best_score:.3f}")

# Guardar mejores parámetros
with open('best_params_ctgan.json', 'w') as f:
    json.dump(best_params, f, indent=2)
```

#### B. Early Stopping

```python
class CTGANWithEarlyStopping(CTGANGenerator):
    def __init__(self, patience=10, min_delta=0.001, **kwargs):
        super().__init__(**kwargs)
        self.patience = patience
        self.min_delta = min_delta
    
    def fit(self, data):
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            loss = self._train_epoch(data)
            
            # Early stopping check
            if loss < best_loss - self.min_delta:
                best_loss = loss
                patience_counter = 0
                self._save_checkpoint()
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                self._load_checkpoint()
                break
```

#### C. Caché de Modelos

```python
import hashlib
import pickle

def get_dataset_hash(df: pd.DataFrame) -> str:
    """Calcular hash único del dataset"""
    # Usar shape + primeras/últimas filas + columnas
    content = f"{df.shape}_{df.head().to_json()}_{df.tail().to_json()}_{df.columns.tolist()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def get_cached_model(df: pd.DataFrame, model_type: str):
    """Cargar modelo cacheado si existe"""
    dataset_hash = get_dataset_hash(df)
    cache_path = f"cache/model_{model_type}_{dataset_hash}.pkl"
    
    if os.path.exists(cache_path):
        logger.info(f"📦 Loading cached model: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    return None

def cache_model(model, df: pd.DataFrame, model_type: str):
    """Guardar modelo en caché"""
    dataset_hash = get_dataset_hash(df)
    cache_path = f"cache/model_{model_type}_{dataset_hash}.pkl"
    
    os.makedirs('cache', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"💾 Saved model to cache: {cache_path}")
```

**Uso**:
```python
# Intentar cargar de caché
cached_generator = get_cached_model(df_original, "ctgan")

if cached_generator:
    synthetic_data = cached_generator.sample(num_samples)
else:
    # Entrenar nuevo
    generator = CTGANGenerator()
    generator.fit(df_original)
    synthetic_data = generator.sample(num_samples)
    
    # Guardar en caché
    cache_model(generator, df_original, "ctgan")
```

---

### 2. **Validación Médica**

#### A. Reglas Configurables

**Archivo**: `config/validation_rules.yaml`

```yaml
covid19:
  pcr_result:
    type: categorical
    values: ['Positive', 'Negative']
    required: true
  
  severity:
    type: categorical
    values: ['Low', 'Medium', 'High', 'Critical']
    required: true
  
  temperature:
    type: numeric
    min: 35.0
    max: 42.0
    unit: "°C"
  
  spo2:
    type: numeric
    min: 70
    max: 100
    unit: "%"
  
  correlations:
    - if: {severity: "Critical"}
      then: {spo2: {max: 90}}
    
    - if: {severity: "Low"}
      then: {temperature: {max: 38.0}}
```

**Cargar reglas**:
```python
import yaml

def load_validation_rules(disease_type: str):
    with open('config/validation_rules.yaml') as f:
        rules = yaml.safe_load(f)
    return rules.get(disease_type, {})

def validate_with_rules(patient: dict, rules: dict) -> list:
    """Validar paciente con reglas configurables"""
    issues = []
    
    for field, field_rules in rules.items():
        if field == 'correlations':
            continue
        
        value = patient.get(field)
        
        # Validar requeridos
        if field_rules.get('required') and value is None:
            issues.append(f"Campo requerido faltante: {field}")
            continue
        
        # Validar tipo categorical
        if field_rules['type'] == 'categorical':
            if value not in field_rules['values']:
                issues.append(f"{field}='{value}' no está en {field_rules['values']}")
        
        # Validar tipo numeric
        elif field_rules['type'] == 'numeric':
            if value < field_rules['min'] or value > field_rules['max']:
                issues.append(f"{field}={value} fuera de rango [{field_rules['min']}, {field_rules['max']}]")
    
    # Validar correlaciones
    for corr_rule in rules.get('correlations', []):
        if_cond = corr_rule['if']
        then_cond = corr_rule['then']
        
        # Verificar si se cumple la condición IF
        if_match = all(patient.get(k) == v for k, v in if_cond.items())
        
        if if_match:
            # Verificar que se cumpla THEN
            for field, constraints in then_cond.items():
                value = patient.get(field)
                if 'max' in constraints and value > constraints['max']:
                    issues.append(f"Correlación violada: Si {if_cond}, entonces {field} <= {constraints['max']}")
    
    return issues
```

#### B. Paralelización de Validación

```python
from multiprocessing import Pool, cpu_count

def validate_patient_parallel(patient_dict):
    """Validar un paciente (función para paralelizar)"""
    return validate_patient_case(patient_dict, is_covid=True)

def validate_dataframe_parallel(df: pd.DataFrame, n_jobs=-1):
    """Validar DataFrame en paralelo"""
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    logger.info(f"🔧 Validando {len(df)} registros con {n_jobs} workers")
    
    # Convertir DataFrame a lista de dicts
    patients = df.to_dict('records')
    
    # Validar en paralelo
    with Pool(n_jobs) as pool:
        results = pool.map(validate_patient_parallel, patients)
    
    # Agregar resultados
    all_valid = [r['is_valid'] for r in results]
    all_issues = [issue for r in results for issue in r['issues']]
    
    return {
        'total': len(df),
        'valid': sum(all_valid),
        'invalid': len(all_valid) - sum(all_valid),
        'issues': all_issues
    }
```

---

### 3. **Evaluación de Utilidad**

#### A. Cross-Validation Robusta

```python
from sklearn.model_selection import StratifiedKFold

def evaluate_ml_utility_cv(df_original, df_synthetic, target_col, n_splits=5):
    """Evaluar utilidad ML con cross-validation"""
    
    # Preparar datos
    X_orig = df_original.drop(columns=[target_col])
    y_orig = df_original[target_col]
    
    X_synt = df_synthetic.drop(columns=[target_col])
    y_synt = df_synthetic[target_col]
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    f1_scores_orig = []
    f1_scores_synt = []
    
    for train_idx, test_idx in skf.split(X_orig, y_orig):
        X_train, X_test = X_orig.iloc[train_idx], X_orig.iloc[test_idx]
        y_train, y_test = y_orig.iloc[train_idx], y_orig.iloc[test_idx]
        
        # Modelo baseline (entrenado en original)
        model_orig = RandomForestClassifier(n_estimators=100, random_state=42)
        model_orig.fit(X_train, y_train)
        y_pred_orig = model_orig.predict(X_test)
        f1_orig = f1_score(y_test, y_pred_orig, average='weighted')
        f1_scores_orig.append(f1_orig)
        
        # Modelo sintético (entrenado en sintético, evaluado en original)
        model_synt = RandomForestClassifier(n_estimators=100, random_state=42)
        model_synt.fit(X_synt, y_synt)
        y_pred_synt = model_synt.predict(X_test)
        f1_synt = f1_score(y_test, y_pred_synt, average='weighted')
        f1_scores_synt.append(f1_synt)
    
    # Estadísticas
    f1_orig_mean = np.mean(f1_scores_orig)
    f1_orig_std = np.std(f1_scores_orig)
    f1_synt_mean = np.mean(f1_scores_synt)
    f1_synt_std = np.std(f1_scores_synt)
    
    preservation = f1_synt_mean / f1_orig_mean if f1_orig_mean > 0 else 0
    
    return {
        'f1_original_mean': f1_orig_mean,
        'f1_original_std': f1_orig_std,
        'f1_synthetic_mean': f1_synt_mean,
        'f1_synthetic_std': f1_synt_std,
        'f1_preservation': preservation,
        'confidence_interval_95': (
            f1_synt_mean - 1.96 * f1_synt_std,
            f1_synt_mean + 1.96 * f1_synt_std
        )
    }
```

#### B. Statistical Significance Testing

```python
from scipy import stats

def test_significance(f1_scores_orig, f1_scores_synt):
    """Probar si la diferencia es estadísticamente significativa"""
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(f1_scores_orig, f1_scores_synt)
    
    # Efecto de tamaño (Cohen's d)
    mean_diff = np.mean(f1_scores_orig) - np.mean(f1_scores_synt)
    pooled_std = np.sqrt((np.std(f1_scores_orig)**2 + np.std(f1_scores_synt)**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
    }
```

---

## 🔒 Garantía de Reproducibilidad

### 1. **Control Total de Semillas**

```python
def set_all_seeds(seed: int = 42):
    """Fijar todas las semillas para reproducibilidad completa"""
    import random
    import numpy as np
    import os
    
    # Python
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # Environment variable para SDV
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"🔒 All seeds set to {seed}")
```

### 2. **Versionado de Datasets**

```python
import hashlib
import json
from datetime import datetime

def version_dataset(df: pd.DataFrame, metadata: dict = None):
    """Versionar dataset con hash y metadata"""
    
    # Calcular hash del contenido
    content_hash = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()[:16]
    
    # Crear metadata
    version_info = {
        'hash': content_hash,
        'timestamp': datetime.now().isoformat(),
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_mb': df.memory_usage(deep=True).sum() / 1e6,
        'metadata': metadata or {}
    }
    
    # Guardar versión
    version_file = f"data/versions/dataset_{content_hash}.json"
    os.makedirs('data/versions', exist_ok=True)
    
    with open(version_file, 'w') as f:
        json.dump(version_info, f, indent=2)
    
    logger.info(f"📦 Dataset versioned: {content_hash}")
    return content_hash, version_info
```

### 3. **Registro de Experimentos**

```python
class ExperimentTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.log = {
            'name': experiment_name,
            'start_time': self.start_time.isoformat(),
            'parameters': {},
            'datasets': {},
            'results': {},
            'metrics': {}
        }
    
    def log_parameter(self, key: str, value):
        self.log['parameters'][key] = value
    
    def log_dataset(self, name: str, df: pd.DataFrame):
        hash_val, version_info = version_dataset(df, {'experiment': self.experiment_name})
        self.log['datasets'][name] = {
            'hash': hash_val,
            'shape': df.shape,
            'version_file': f"data/versions/dataset_{hash_val}.json"
        }
    
    def log_metric(self, key: str, value):
        self.log['metrics'][key] = value
    
    def log_result(self, key: str, value):
        self.log['results'][key] = value
    
    def finish(self):
        self.log['end_time'] = datetime.now().isoformat()
        self.log['duration_seconds'] = (datetime.now() - self.start_time).total_seconds()
        
        # Guardar log
        exp_file = f"experiments/{self.experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('experiments', exist_ok=True)
        
        with open(exp_file, 'w') as f:
            json.dump(self.log, f, indent=2)
        
        logger.info(f"📝 Experiment logged: {exp_file}")
        return exp_file

# Uso
tracker = ExperimentTracker("covid_ctgan_500")
tracker.log_parameter("model", "CTGAN")
tracker.log_parameter("epochs", 300)
tracker.log_parameter("num_samples", 500)
tracker.log_dataset("original", df_original)
tracker.log_dataset("synthetic", df_synthetic)
tracker.log_metric("f1_score", 0.85)
tracker.log_result("generation_time", 120.5)
tracker.finish()
```

---

## ✅ Checklist de Experimentos

### Antes de Ejecutar

- [ ] **Semillas fijadas**: Verificar `set_all_seeds(42)`
- [ ] **Dataset versionado**: Calcular hash y guardar metadata
- [ ] **Parámetros documentados**: Registrar todos los parámetros en log
- [ ] **Baseline establecido**: Tener métricas de referencia
- [ ] **Espacio en disco**: Verificar espacio para resultados (> 1GB libre)

### Durante Ejecución

- [ ] **Logging activo**: Verificar que logs se escriben correctamente
- [ ] **Monitoreo de recursos**: CPU, memoria, GPU
- [ ] **Checkpoints**: Guardar estados intermedios
- [ ] **Validación incremental**: Verificar calidad en cada etapa

### Después de Ejecutar

- [ ] **Resultados guardados**: DataFrame, modelos, métricas
- [ ] **Logs analizados**: Revisar errores y warnings
- [ ] **Comparación con baseline**: Calcular diferencias
- [ ] **Visualizaciones generadas**: Plots de distribuciones, correlaciones
- [ ] **Informe generado**: Markdown con resultados
- [ ] **Código commiteado**: Git commit con mensaje descriptivo

---

## 📊 Template de Reporte de Experimento

```markdown
# Experimento: [Nombre]

## Metadata
- **Fecha**: 2025-10-15
- **Investigador**: [Nombre]
- **Objetivo**: [Describir objetivo del experimento]
- **Hipótesis**: [Hipótesis a probar]

## Configuración
- **Dataset**: COVID-19 (hash: abc123, n=1000, cols=25)
- **Modelo**: CTGAN
- **Parámetros**:
  - epochs: 300
  - batch_size: 500
  - seed: 42

## Resultados

### Generación
- Tiempo: 120.5s
- Registros generados: 500
- Memoria usada: 45.2 MB

### Validación
- Coherencia clínica: 87.3%
- Errores de esquema: 2.1%
- Issues encontrados: 12

### Evaluación
- **Fidelidad**: 89.2%
  - Correlaciones: 91.5%
  - Distribuciones: 88.3%
  - Valores únicos: 87.8%

- **Utilidad ML**: 82.7%
  - F1 original: 0.91 ± 0.03
  - F1 sintético: 0.75 ± 0.05
  - F1 preservation: 82.4%
  - p-value: 0.032 (significativo)

- **Privacidad**: 95.6%
  - DCR medio: 0.23
  - Registros en riesgo: 22 (4.4%)

- **Score Final**: 87.1% (Excellent)

## Análisis
[Interpretar resultados, comparar con baseline, discutir implicaciones]

## Conclusiones
[Conclusiones principales del experimento]

## Próximos Pasos
- [ ] Probar con TVAE
- [ ] Aumentar epochs a 500
- [ ] Validar con expertos médicos
```

---

## 📞 Contacto y Soporte

Para dudas sobre depuración y optimización:
1. Revisar logs en `logs/`
2. Consultar documentación en `ARQUITECTURA_AGENTES_DETALLADA.md`
3. Revisar ejemplos en `notebooks/`

---

**Autor**: Sistema Patient-IA  
**Versión**: 1.0  
**Fecha**: 15 de Octubre, 2025
