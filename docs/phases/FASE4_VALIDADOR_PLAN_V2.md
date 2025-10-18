# Fase 4: Optimización del Agente Validador (Plan Completo v2)

## 📋 Objetivo
Optimizar el agente Validador para validación más eficiente, configurable, escalable y explicable de datos sintéticos y originales, implementando **todas las mejoras propuestas en la arquitectura** del sistema.

---

## 🎯 Objetivos Específicos (Alineados con ARQUITECTURA_AGENTES_DETALLADA.md)

### 1. Sistema de Caché de Validaciones ✅
**Referencia**: Arquitectura - Optimizaciones Validador #1  
**Prioridad**: ALTA

**Implementación**:
- Caché por hash de (datos + reglas + esquema)
- Evitar re-validar datos ya validados
- Invalidación automática si cambian reglas, esquema o datos
- Métricas de hit/miss ratio del caché
- TTL configurable por variable de entorno

**Archivos**:
- `src/validation/validation_cache.py` (nuevo)
- Modificar `src/agents/validator_agent.py`

**Métricas Esperadas**:
- Hit en 2da validación con mismos datos
- Reducción de tiempo ≥ 90% con caché hit
- Speedup esperado: 10-30x en llamadas repetidas

---

### 2. Reglas Configurables (YAML/JSON) ✅
**Referencia**: Arquitectura - Optimizaciones Validador #1  
**Prioridad**: ALTA

**Implementación**:
- Externalizar reglas clínicas a `src/config/clinical_rules.yaml`
- Externalizar esquemas JSON a `src/config/json_schemas/`
- Reglas específicas por tipo de dataset:
  - `patient_covid.json` (COVID-19)
  - `patient_diabetes.json` (Diabetes)
  - `patient_cardiology.json` (Cardiología)
  - `patient_generic.json` (Genérico)
- Hot-reload de reglas sin reiniciar servidor
- Facilitar personalización sin cambiar código

**Estructura de Reglas YAML**:
```yaml
# src/config/clinical_rules.yaml
dataset_type: covid19

rules:
  - name: "PCR Result Validation"
    field: "PCR_Result"
    type: "categorical"
    allowed_values: ["Positive", "Negative"]
    severity: "critical"
    message: "PCR Result must be Positive or Negative"
    
  - name: "Temperature Range"
    field: "Temperature"
    type: "numeric"
    min: 35.0
    max: 42.0
    severity: "critical"
    message: "Temperature must be between 35°C and 42°C"
    
  - name: "SpO2-Severity Correlation"
    type: "correlation"
    condition: "if Severity == 'Critical' then SpO2 < 90"
    severity: "warning"
    message: "Critical severity should have SpO2 < 90%"
```

**Archivos**:
- `src/validation/rules_parser.py` (nuevo)
- `src/validation/schema_loader.py` (nuevo)
- `src/config/clinical_rules.yaml` (nuevo)
- `src/config/json_schemas/*.json` (nuevos)
- `src/config/validation_config.yaml` (configuración general, nuevo)

---

### 3. Validación Paralela (Multiprocessing) ✅
**Referencia**: Arquitectura - Optimizaciones Validador #2  
**Prioridad**: MEDIA

**Implementación**:
- Validar registros en paralelo usando `multiprocessing`
- Chunking inteligente de datasets grandes (>10k registros)
- Pool de workers configurable (default: 4)
- Progress tracking en tiempo real
- Merge inteligente de resultados

**Estrategia**:
```python
# Para datasets grandes
if len(df) > 10000:
    chunks = split_dataframe(df, num_workers=4)
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(validate_chunk, chunks)
    return merge_validation_results(results)
```

**Archivos**:
- `src/validation/parallel_validator.py` (nuevo)
- Modificar `src/agents/validator_agent.py`

**Métricas Esperadas**:
- Speedup de 2-3x en datasets >10k registros
- Overhead mínimo en datasets pequeños

---

### 4. Reglas ML para Detección de Anomalías ✅
**Referencia**: Arquitectura - Optimizaciones Validador #3  
**Prioridad**: MEDIA

**Implementación**:
- Entrenar **Isolation Forest** o **One-Class SVM** para detectar registros "sospechosos"
- Complementar reglas estáticas con detección ML
- Modelo entrenado en datos originales, aplicado a sintéticos
- Score de anomalía por registro (0-1)
- Threshold configurable para clasificar anomalía

**Flujo**:
1. Entrenar modelo en datos **originales** (aprender distribución normal)
2. Aplicar a datos **sintéticos** (detectar outliers)
3. Asignar score de anomalía a cada registro
4. Flaggear registros con score > threshold

**Archivos**:
- `src/validation/anomaly_detector.py` (nuevo)
- Modificar `src/agents/validator_agent.py`

**Métricas**:
- Precision/Recall de anomalías vs validación manual
- % de anomalías detectadas

---

### 5. Explicabilidad (SHAP/Feature Analysis) ✅
**Referencia**: Arquitectura - Optimizaciones Validador #4  
**Prioridad**: MEDIA-ALTA

**Implementación**:
- **SHAP values** para explicar por qué un registro es inválido
- Feature importance para reglas que fallan
- Reportes de validación más claros y accionables
- Visualización de anomalías (opcional)

**Ejemplo de Salida**:
```
❌ Registro 42 INVÁLIDO:
  - SpO2 = 55% (fuera de rango 70-100%) [Crítico]
  - Severity = "Low" pero SpO2 < 90% (inconsistencia) [Warning]
  
📊 SHAP Analysis:
  - SpO2: -0.45 (mayor contribución a invalidez)
  - Severity: -0.20 (contribución media)
  - Age: +0.05 (contribución mínima)
```

**Archivos**:
- `src/validation/explainer.py` (nuevo)
- Modificar `src/agents/validator_agent.py`

---

### 6. Validación Incremental ✅
**Referencia**: Arquitectura - Optimizaciones Validador #5  
**Prioridad**: BAJA

**Implementación**:
- Solo validar registros nuevos/modificados
- Tracking de registros ya validados por hash
- Útil para datasets que cambian iterativamente
- Delta validation mode

**Caso de Uso**:
- Dataset inicial: 10,000 registros → validar todos
- Dataset actualizado: +500 registros → validar solo 500 nuevos

**Archivos**:
- `src/validation/incremental_validator.py` (nuevo)
- Modificar `src/agents/validator_agent.py`

---

### 7. Métricas Avanzadas y Logging ✅
**Prioridad**: ALTA

**Implementación**:
- Tracking detallado de tiempo de validación (por fase)
- Logging estructurado de issues encontrados
- Métricas de calidad y coherencia detalladas:
  - **Clinical Coherence** (0-1):
    - Signos vitales coherentes (40%)
    - Correlaciones demográficas (30%)
    - Validez médica (30%)
  - **Data Quality** (0-1):
    - Errores de esquema (50%)
    - Valores fuera de rango (30%)
    - Datos faltantes (20%)
  - **Overall Score** (0-1): `(clinical_coherence * 0.6) + (data_quality * 0.4)`
- Performance tracking (validaciones/segundo, throughput)

**Archivos**:
- `src/validation/validation_metrics.py` (nuevo)
- Modificar `src/agents/validator_agent.py`

---

### 8. Benchmarking con Expertos ✅
**Referencia**: Arquitectura - Optimizaciones Validador #6  
**Prioridad**: BAJA (Futuro)

**Implementación**:
- Framework para comparar validaciones con expertos humanos
- Métricas de concordancia (Cohen's Kappa, F1)
- Dataset de validación ground-truth
- Mejora iterativa del sistema basado en feedback de expertos

**Archivos**:
- `src/validation/benchmark.py` (nuevo)
- `data/validation_ground_truth.csv` (ground truth de expertos)

---

## 📂 Archivos Involucrados

### Archivos Principales a Modificar
- ✏️ `src/agents/validator_agent.py` (agente principal)
- ✏️ `src/validation/clinical_rules.py` (migrar a YAML + parser)
- ✏️ `src/validation/json_schema.py` (migrar a JSON + loader)

### Nuevos Módulos a Crear (8 módulos)
1. ✨ `src/validation/validation_cache.py` (caché de validaciones)
2. ✨ `src/validation/rules_parser.py` (parser de reglas YAML)
3. ✨ `src/validation/schema_loader.py` (loader de esquemas JSON)
4. ✨ `src/validation/validation_metrics.py` (métricas detalladas)
5. ✨ `src/validation/parallel_validator.py` (validación paralela)
6. ✨ `src/validation/explainer.py` (explicabilidad con SHAP)
7. ✨ `src/validation/incremental_validator.py` (validación incremental)
8. ✨ `src/validation/anomaly_detector.py` (detector ML de anomalías)
9. ✨ `src/validation/benchmark.py` (comparación con expertos)

### Archivos de Configuración a Crear
- ✨ `src/config/clinical_rules.yaml` (reglas COVID-19, diabetes, genérico)
- ✨ `src/config/json_schemas/patient_covid.json` (esquema COVID-19)
- ✨ `src/config/json_schemas/patient_diabetes.json` (esquema Diabetes)
- ✨ `src/config/json_schemas/patient_cardiology.json` (esquema Cardiología)
- ✨ `src/config/json_schemas/patient_generic.json` (esquema genérico)
- ✨ `src/config/validation_config.yaml` (configuración general)

### Tests a Crear
- ✨ `tests/test_validator_cache.py`
- ✨ `tests/test_validator_rules_parser.py`
- ✨ `tests/test_validator_parallel.py`
- ✨ `tests/test_validator_anomaly_detector.py`
- ✨ `tests/test_validator_explainer.py`
- ✨ `tests/test_validator_complete.py` (test de integración completo)

---

## 🔄 Plan de Implementación (Fases Secuenciales)

### **Fase 4.1: Backup y Análisis** ⏳
**Duración**: 30 min

- [x] Backup de `validator_agent.py`
- [ ] Analizar estructura actual del validador
- [ ] Identificar todas las reglas de validación existentes
- [ ] Mapear dependencias entre reglas
- [ ] Leer `clinical_rules.py` y `json_schema.py` completos

---

### **Fase 4.2: Reglas Configurables (YAML/JSON)** 🎯
**Duración**: 2-3 horas  
**Prioridad**: ALTA (fundamento para otras mejoras)

**Sub-tareas**:
1. Crear estructura de `src/config/clinical_rules.yaml`
   - Migrar reglas COVID-19 existentes
   - Añadir reglas diabetes, cardiología, genéricas
2. Crear `src/validation/rules_parser.py`
   - Parser de YAML a objetos Python
   - Validación de sintaxis de reglas
   - Hot-reload de archivos
3. Crear esquemas JSON en `src/config/json_schemas/`
   - `patient_covid.json`
   - `patient_generic.json`
4. Crear `src/validation/schema_loader.py`
   - Loader de esquemas JSON
   - Validación de esquemas
5. Modificar `validator_agent.py` para usar rules_parser y schema_loader
6. Test: `tests/test_validator_rules_parser.py`

**Criterio de Éxito**:
- Reglas cargadas desde YAML sin errores
- Validación funciona igual que antes
- Hot-reload funcional

---

### **Fase 4.3: Métricas Avanzadas y Logging** 📊
**Duración**: 1-2 horas  
**Prioridad**: ALTA

**Sub-tareas**:
1. Crear `src/validation/validation_metrics.py`
   - Clase `ValidationMetrics`
   - Tracking de tiempo, errores, warnings
   - Cálculo de Clinical Coherence y Data Quality
2. Modificar `validator_agent.py` para integrar métricas
3. Logging estructurado con `structlog`
4. Test: validar métricas en `test_validator_complete.py`

**Criterio de Éxito**:
- Métricas completas en logs
- Clinical Coherence y Data Quality calculados correctamente
- Performance tracking funcional

---

### **Fase 4.4: Sistema de Caché** 💾
**Duración**: 2 horas  
**Prioridad**: ALTA

**Sub-tareas**:
1. Crear `src/validation/validation_cache.py`
   - Hash de (datos + reglas + esquema)
   - Get/Set/Invalidate
   - TTL configurable
2. Modificar `validator_agent.py` para usar caché
3. Variables de entorno (VALIDATOR_CACHE_*)
4. Test: `tests/test_validator_cache.py`

**Criterio de Éxito**:
- Hit en 2da validación con mismos datos
- Reducción de tiempo ≥ 90% con caché hit
- Invalidación correcta al cambiar reglas

---

### **Fase 4.5: Validación Paralela** ⚡
**Duración**: 2-3 horas  
**Prioridad**: MEDIA

**Sub-tareas**:
1. Crear `src/validation/parallel_validator.py`
   - Chunking de datasets
   - ProcessPoolExecutor
   - Merge de resultados
2. Modificar `validator_agent.py` para usar validación paralela
3. Variables de entorno (VALIDATOR_PARALLEL_*, VALIDATOR_MAX_WORKERS)
4. Test: `tests/test_validator_parallel.py`

**Criterio de Éxito**:
- Speedup de 2-3x en datasets >10k registros
- Overhead mínimo en datasets pequeños
- Resultados consistentes con validación serial

---

### **Fase 4.6: Explicabilidad (SHAP)** 🔍
**Duración**: 2-3 horas  
**Prioridad**: MEDIA-ALTA

**Sub-tareas**:
1. Crear `src/validation/explainer.py`
   - SHAP values para registros inválidos
   - Feature importance
   - Formateo de explicaciones
2. Modificar `validator_agent.py` para integrar explicabilidad
3. Instalar dependencia `shap`
4. Test: validar explicaciones en `test_validator_explainer.py`

**Criterio de Éxito**:
- Explicaciones claras para registros inválidos
- SHAP values correctos
- Reportes más accionables

---

### **Fase 4.7: Detector de Anomalías (ML)** 🤖
**Duración**: 3-4 horas  
**Prioridad**: MEDIA

**Sub-tareas**:
1. Crear `src/validation/anomaly_detector.py`
   - Entrenamiento de Isolation Forest
   - Aplicación a datos sintéticos
   - Score de anomalía por registro
2. Modificar `validator_agent.py` para usar detector
3. Caché de modelos entrenados
4. Test: `tests/test_validator_anomaly_detector.py`

**Criterio de Éxito**:
- Modelo entrenado en datos originales
- Anomalías detectadas correctamente
- Precision/Recall aceptables (>0.70)

---

### **Fase 4.8: Validación Incremental** 🔄
**Duración**: 1-2 horas  
**Prioridad**: BAJA

**Sub-tareas**:
1. Crear `src/validation/incremental_validator.py`
   - Tracking de hashes de registros validados
   - Delta detection
   - Merge con validaciones anteriores
2. Modificar `validator_agent.py` para soportar modo incremental
3. Test: validar delta validation

**Criterio de Éxito**:
- Solo registros nuevos validados en llamadas sucesivas
- Speedup significativo para cambios pequeños

---

### **Fase 4.9: Testing Completo** ✅
**Duración**: 2-3 horas  
**Prioridad**: ALTA

**Sub-tareas**:
1. Ejecutar todos los tests creados
2. Test de integración completo (`test_validator_complete.py`)
3. Validar con datasets reales (COVID-19, diabetes)
4. Benchmarking de performance (serial vs paralelo, con/sin caché)
5. Validar que no hay regresiones

**Criterio de Éxito**:
- Todos los tests pasan
- No regresiones en funcionalidad
- Performance mejorado significativamente

---

### **Fase 4.10: Documentación** 📚
**Duración**: 1-2 horas  
**Prioridad**: ALTA

**Sub-tareas**:
1. Actualizar docstrings en todos los módulos
2. Crear `docs/phases/FASE4_VALIDADOR_COMPLETADA.md`
3. Documentar nuevas variables de entorno
4. Actualizar `README.md` principal
5. Actualizar `INDICE_DOCUMENTACION.md`
6. Crear guía de uso de reglas YAML

---

## ⚙️ Variables de Entorno (Nuevas)

```bash
# ========== CACHÉ DE VALIDACIONES ==========
VALIDATOR_CACHE_ENABLED=true
VALIDATOR_CACHE_DIR=temp_generations/validator_cache
VALIDATOR_CACHE_TTL_HOURS=24

# ========== PERFORMANCE ==========
VALIDATOR_PARALLEL_ENABLED=false  # Habilitar solo para datasets >10k
VALIDATOR_MAX_WORKERS=4
VALIDATOR_BATCH_SIZE=1000

# ========== CONFIGURACIÓN DE REGLAS ==========
VALIDATOR_RULES_FILE=src/config/clinical_rules.yaml
VALIDATOR_SCHEMAS_DIR=src/config/json_schemas
VALIDATOR_STRICT_MODE=false  # true = errores críticos paran validación
VALIDATOR_HOT_RELOAD=true    # Recargar reglas sin reiniciar

# ========== DETECCIÓN DE ANOMALÍAS ==========
VALIDATOR_ANOMALY_DETECTION_ENABLED=false  # ML para anomalías
VALIDATOR_ANOMALY_THRESHOLD=0.7            # Score > 0.7 = anomalía
VALIDATOR_ANOMALY_MODEL=isolation_forest   # isolation_forest | one_class_svm

# ========== EXPLICABILIDAD ==========
VALIDATOR_EXPLAIN_INVALID=true             # SHAP para registros inválidos
VALIDATOR_EXPLAIN_TOP_N=5                  # Top 5 features más importantes

# ========== VALIDACIÓN INCREMENTAL ==========
VALIDATOR_INCREMENTAL_MODE=false           # Solo validar registros nuevos
```

---

## 🎯 Criterios de Éxito Global

### 1. **Caché funcional** ✅
- Hit en 2da validación con mismos datos
- Reducción de tiempo ≥ 90%
- Invalidación correcta al cambiar reglas

### 2. **Reglas configurables** ✅
- Reglas cargadas desde YAML sin errores
- Hot-reload funcional
- Fácil añadir nuevas reglas sin código

### 3. **Métricas completas** ✅
- Clinical Coherence y Data Quality calculados
- Tiempo, reglas aplicadas, errores/warnings en logs
- Throughput medido (records/sec, rules/sec)

### 4. **Validación paralela** ✅
- Speedup de 2-3x en datasets >10k registros
- Overhead mínimo en datasets pequeños

### 5. **Explicabilidad** ✅
- SHAP values para registros inválidos
- Reportes claros y accionables

### 6. **Detector de anomalías** ✅
- Precision/Recall ≥ 0.70 en detección de anomalías
- Complemento efectivo a reglas estáticas

### 7. **No regresiones** ✅
- Todos los tests anteriores pasan
- Misma calidad de validación

### 8. **Documentación completa** ✅
- Plan y resultados documentados
- Guía de uso para nuevas variables
- README actualizado

---

## 📊 Métricas Esperadas

### Performance
- **Sin caché (1ra vez)**: 5-30s para validar 1000 registros
- **Con caché (2da vez)**: <1s para re-validación
- **Speedup con caché**: 10-30x en llamadas repetidas
- **Validación paralela**: 2-3x más rápido en datasets >10k

### Calidad de Validación
- Mantener 100% de detección de errores actuales
- Mejorar categorización (crítico vs warning)
- Añadir sugerencias útiles para corrección
- Precision/Recall de anomalías ≥ 0.70

### Usabilidad
- Tiempo para añadir nueva regla: <5 min (editar YAML)
- Hot-reload sin reiniciar servidor
- Reportes más claros y accionables

---

## 📝 Notas de Implementación

- ✅ Mantener compatibilidad con código existente
- ✅ Caché y paralelización opcionales (pueden deshabilitarse)
- ✅ Tests exhaustivos antes de integrar
- ✅ Validar con datasets reales médicos (COVID-19, diabetes)
- ✅ Documentar cada mejora tras implementarla
- ✅ No sobreoptimizar (KISS - Keep It Simple)

---

## 🚀 Próximos Pasos Inmediatos

1. **Leer y analizar** `validator_agent.py`, `clinical_rules.py`, `json_schema.py`
2. **Iniciar Fase 4.1**: Backup y análisis detallado
3. **Iniciar Fase 4.2**: Implementar reglas configurables (YAML)
4. **Continuar secuencialmente** según plan de fases

---

## 🔗 Referencias

- **Arquitectura**: `ARQUITECTURA_AGENTES_DETALLADA.md` (Sección 4: Agente Validador)
- **Fase 3 (Generador)**: `docs/phases/FASE3_GENERADOR_ESTADO_ACTUAL.md`
- **Tests existentes**: `tests/test_analyzer_improvements.py` (patrón a seguir)

---

**Fecha de inicio**: 2025-01-16  
**Responsable**: Copilot + Usuario  
**Estado**: 📋 En planificación (Plan Completo v2)  
**Dependencias**: Fase 3 al 70% (puede continuar en paralelo)  
**Duración estimada**: 15-20 horas de desarrollo

---

## ✅ Checklist de Progreso

### Fase 4.1: Backup y Análisis
- [x] Backup de `validator_agent.py`
- [ ] Analizar estructura actual del validador
- [ ] Identificar reglas de validación existentes
- [ ] Mapear dependencias entre reglas

### Fase 4.2: Reglas Configurables
- [ ] Crear `clinical_rules.yaml`
- [ ] Crear `rules_parser.py`
- [ ] Crear esquemas JSON
- [ ] Crear `schema_loader.py`
- [ ] Modificar `validator_agent.py`
- [ ] Test de rules parser

### Fase 4.3: Métricas y Logging
- [ ] Crear `validation_metrics.py`
- [ ] Integrar métricas en validator
- [ ] Logging estructurado

### Fase 4.4: Sistema de Caché
- [ ] Crear `validation_cache.py`
- [ ] Integrar caché en validator
- [ ] Test de caché

### Fase 4.5: Validación Paralela
- [ ] Crear `parallel_validator.py`
- [ ] Integrar en validator
- [ ] Test de paralelización

### Fase 4.6: Explicabilidad
- [ ] Crear `explainer.py`
- [ ] Integrar SHAP
- [ ] Test de explicabilidad

### Fase 4.7: Detector de Anomalías
- [ ] Crear `anomaly_detector.py`
- [ ] Entrenar modelo
- [ ] Test de detección

### Fase 4.8: Validación Incremental
- [ ] Crear `incremental_validator.py`
- [ ] Integrar en validator

### Fase 4.9: Testing Completo
- [ ] Ejecutar todos los tests
- [ ] Test de integración
- [ ] Validar con datasets reales
- [ ] Benchmarking

### Fase 4.10: Documentación
- [ ] Actualizar docstrings
- [ ] Crear `FASE4_VALIDADOR_COMPLETADA.md`
- [ ] Documentar variables de entorno
- [ ] Actualizar README
- [ ] Guía de uso de reglas YAML

---

**Total de Progreso**: 1/42 tareas completadas (2.4%)

