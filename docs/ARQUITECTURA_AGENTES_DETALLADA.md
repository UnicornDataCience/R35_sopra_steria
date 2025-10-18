# 🏥 Arquitectura Detallada del Sistema Multi-Agente Patient-IA

> **Documento Técnico para Investigación**  
> Última actualización: 15 de Octubre, 2025  
> Versión: 1.0

---

## 📋 Índice

1. [Introducción](#introducción)
2. [Arquitectura Global](#arquitectura-global)
3. [Agente Coordinador](#1-agente-coordinador)
4. [Agente Analizador](#2-agente-analizador)
5. [Agente Generador](#3-agente-generador)
6. [Agente Validador](#4-agente-validador)
7. [Agente Evaluador](#5-agente-evaluador)
8. [Agente Simulador](#6-agente-simulador)
9. [Flujo de Procesamiento](#flujo-de-procesamiento)
10. [Módulos de Soporte](#módulos-de-soporte)
11. [Configuración y Parámetros](#configuración-y-parámetros)
12. [Oportunidades de Optimización](#oportunidades-de-optimización)
13. [Consideraciones para Investigación](#consideraciones-para-investigación)

---

## 🎯 Introducción

Patient-IA es un sistema multi-agente especializado en el procesamiento, análisis, generación, validación y evaluación de datos médicos sintéticos. El sistema utiliza:

- **LangChain** para la orquestación de agentes
- **LangGraph** para la gestión de flujos
- **OpenAI GPT-4** como LLM principal
- **SDV (Synthetic Data Vault)** con CTGAN, TVAE para generación de datos
- **Scikit-learn** para evaluación de utilidad ML
- **Pandas/NumPy** para manipulación de datos

---

## 🏗️ Arquitectura Global

```
┌─────────────────────────────────────────────────────────────┐
│                     USUARIO / API REST                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              ORQUESTADOR (LangGraph)                        │
│  • Gestión de estado                                        │
│  • Enrutamiento de mensajes                                 │
│  • Memoria de contexto                                      │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────────────────────────────────────┐
│            COORDINADOR (Decision Agent)                    │
│  • Detección de intenciones                               │
│  • Clasificación: conversación vs comando                 │
│  • Extracción de parámetros                               │
│  • Enrutamiento a agentes especializados                  │
└─────┬──────────────────────────────────────────────────────┘
      │
      ├─────────────┬──────────────┬──────────────┬──────────┐
      ▼             ▼              ▼              ▼          ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ANALIZADOR│  │GENERADOR │  │VALIDADOR │  │EVALUADOR │  │SIMULADOR │
│  AGENT   │  │  AGENT   │  │  AGENT   │  │  AGENT   │  │  AGENT   │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │              │              │              │
     ▼             ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ ANÁLISIS │  │GENERACIÓN│  │VALIDACIÓN│  │EVALUACIÓN│  │SIMULACIÓN│
│  MODULE  │  │  MODULE  │  │  MODULE  │  │  MODULE  │  │  MODULE  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

---

## 1. 🎯 Agente Coordinador

### 📝 Descripción
El **Coordinador** es el punto de entrada principal del sistema. Actúa como un router inteligente que:
1. Clasifica la intención del usuario (conversación vs comando)
2. Detecta si es una consulta médica
3. Extrae parámetros de la solicitud
4. Enruta al agente especializado apropiado

### 🔧 Configuración

```python
Nombre: "Coordinador"
LLM: GPT-4
Temperatura: 0.0  # Determinista para clasificación precisa
Max Tokens: 1500  # Default
Herramientas: Ninguna (solo clasificación)
```

### 📊 Flujo de Procesamiento

```
INPUT: Mensaje del usuario + Contexto
   │
   ▼
┌─────────────────────────────────────────┐
│ 1. PREPROCESAMIENTO                     │
│    • Extraer contexto del dataset       │
│    • Agregar resultados de operaciones  │
│    • Construir prompt enriquecido       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 2. LLAMADA AL LLM (GPT-4)               │
│    • System Prompt: Instrucciones       │
│    • User Input: Mensaje + Contexto     │
│    • Formato: JSON estructurado         │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 3. PARSEO DE RESPUESTA                  │
│    • Extraer JSON de bloques de código  │
│    • Validar con Pydantic Schema        │
│    • Fallback heurístico si falla       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│ 4. DECISIÓN DE ENRUTAMIENTO             │
│    • intention: conversacion/comando    │
│    • agent: target agent name           │
│    • is_medical_query: bool             │
│    • parameters: dict                   │
│    • message: respuesta del coordinador │
└────────────┬────────────────────────────┘
             │
             ▼
OUTPUT: CoordinatorDecision (JSON)
```

### 🎯 Esquema de Salida (Pydantic)

```python
class CoordinatorDecision(BaseModel):
    intention: str  # "conversacion" | "comando"
    agent: str      # "coordinator" | "analyzer" | "generator" | ...
    is_medical_query: bool
    parameters: Dict[str, Any]  # Parámetros extraídos
    message: str    # Respuesta al usuario
```

### 🔍 Detección de Intenciones

**Conversación:**
- Keywords: hola, gracias, adiós, síntomas, tratamiento, diagnóstico
- Resultado: `intention="conversacion"`, `agent="coordinator"`
- El coordinador responde directamente usando contexto del dataset

**Comando:**
- Keywords: analizar, generar, validar, simular, evaluar
- Resultado: `intention="comando"`, `agent="<agente_especializado>"`
- Se enruta al agente correspondiente con parámetros extraídos

### ⚠️ Manejo de Errores

1. **JSON Inválido**: Fallback heurístico por keywords
2. **Intención Ambigua**: Default a conversación
3. **Agente Desconocido**: Redirige a coordinador

### 🚀 Optimizaciones Recomendadas

1. **Caché de Respuestas Comunes**: Implementar caché para saludos y preguntas frecuentes
2. **Fine-tuning del Prompt**: Ajustar ejemplos para mejorar precisión de clasificación
3. **Validación de Parámetros**: Agregar validación más estricta de parámetros extraídos
4. **Métricas de Performance**: Logging de tiempo de respuesta y acierto en clasificación

---

## 2. 📊 Agente Analizador

### 📝 Descripción
El **Analizador** es responsable de realizar análisis exploratorio de datos (EDA) sobre datasets médicos. Genera informes en Markdown con insights estadísticos y médicos.

### 🔧 Configuración

```python
Nombre: "Analizador Clínico"
LLM: GPT-4
Temperatura: 0.2
Max Tokens: 2500  # Incrementado para informes largos
Herramientas: Ninguna (recibe análisis preprocesado)
```

### 📊 Flujo de Procesamiento

```
INPUT: DataFrame + Contexto
   │
   ▼
┌──────────────────────────────────────────────┐
│ 1. ANÁLISIS UNIVERSAL (Preprocesamiento)    │
│    Archivo: src/adapters/universal_dataset_  │
│            detector.py                        │
│    • Detectar tipo de dataset (COVID-19,     │
│      Diabetes, Cardiología, General)         │
│    • Identificar columnas médicas clave      │
│    • Mapeo de columnas (age, gender, etc.)   │
│    • Calcular estadísticas básicas           │
│    • Detectar valores faltantes             │
│    • Análisis de correlaciones              │
│    • Detectar patrones médicos              │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 2. RESUMEN PARA LLM                          │
│    Método: _summarize_analysis_for_llm()     │
│    • Reducir tamaño de análisis             │
│    • Extraer solo métricas clave            │
│    • Limitar a primeras 10 columnas sample  │
│    • Top 5 correlaciones                    │
│    • Truncar si > 8000 caracteres           │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 3. GENERACIÓN DE INFORME (LLM)              │
│    • System Prompt: Instrucciones EDA        │
│    • Input: JSON con análisis resumido       │
│    • Output: Markdown con 5 secciones:      │
│      1. Resumen Ejecutivo                   │
│      2. Análisis Descriptivo                │
│      3. Calidad de Datos                    │
│      4. Análisis de Variables Clave         │
│      5. Conclusiones y Recomendaciones      │
└──────────────┬───────────────────────────────┘
               │
               ▼
OUTPUT: Informe Markdown (2500 tokens aprox)
```

### 🔬 Módulo de Análisis Universal

**Archivo**: `src/adapters/universal_dataset_detector.py`

**Funciones Principales**:
1. `detect_dataset_type(df)`: Detecta tipo de dataset por keywords en columnas
2. `map_columns(df, dataset_type)`: Mapea columnas a roles estándar (age, gender, diagnosis)
3. `calculate_statistics(df)`: Estadísticas básicas (media, mediana, std, quartiles)
4. `detect_missing_values(df)`: Conteo y porcentaje de nulos por columna
5. `analyze_correlations(df)`: Matriz de correlación de Pearson para numéricas
6. `detect_medical_patterns(df)`: Detecta columnas ID, fechas, diagnósticos

### 📈 Métricas Calculadas

**Estadísticas Básicas**:
- Filas, columnas, tamaño en memoria
- Tipos de datos por columna
- Valores únicos por columna categórica

**Valores Faltantes**:
- Total de nulos
- Porcentaje de nulos
- Columnas más afectadas

**Correlaciones**:
- Matriz de correlación Pearson
- Top correlaciones (|r| > 0.5)
- Pares altamente correlacionados

**Patrones Médicos**:
- Identificación de Patient ID
- Columnas de edad, género, diagnóstico
- Detección de fechas/timestamps
- Clasificación de severidad/prioridad

### ⚙️ Optimización: Muestreo Inteligente

Si el dataset tiene **> 5000 filas**, se aplica muestreo:
- Muestra aleatoria de **5000 filas**
- Mantiene distribución estratificada si hay target
- Reduce tiempo de análisis de O(n²) a O(1)

**Código**:
```python
if len(df) > 5000:
    sample_df = df.sample(n=5000, random_state=42)
    sampling_info = {
        "sampled": True,
        "original_rows": len(df),
        "sample_rows": len(sample_df)
    }
```

### 🚀 Optimizaciones Recomendadas

1. **Caché de Análisis**: Guardar análisis por hash del dataset
2. **Análisis Incremental**: Solo recalcular si cambian datos
3. **Paralelización**: Usar multiprocessing para correlaciones grandes
4. **Reducción de Tokens**: Comprimir más el JSON (actualmente 8000 chars max)
5. **Fine-tuning**: Entrenar modelo específico para generar EDAs médicos

---

## 3. 🧬 Agente Generador

### 📝 Descripción
El **Generador** es responsable de crear datos médicos sintéticos utilizando técnicas de ML avanzadas (CTGAN, TVAE, SDV).

### 🔧 Configuración

```python
Nombre: "Generador Sintético"
LLM: GPT-4
Temperatura: 0.2
Max Tokens: 1500
Herramientas: Ninguna (usa módulos de generación)
Seed: 42 (reproducibilidad)
```

### 📊 Flujo de Procesamiento

```
INPUT: DataFrame Original + Parámetros
   │
   ▼
┌──────────────────────────────────────────────┐
│ 1. SELECCIÓN DE COLUMNAS                     │
│    Archivo: src/adapters/medical_column_     │
│            selector.py                        │
│    • Excluir columnas ID, timestamps         │
│    • Priorizar columnas médicas relevantes   │
│    • Limitar a máx. 30 columnas             │
│    • Balancear numéricas y categóricas      │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 2. SELECCIÓN DE MODELO                       │
│    Método: _choose_model_auto()              │
│    Heurística:                               │
│    • Si rows < 300: SDV (modelos simples)   │
│    • Si cat_ratio >= 60%: CTGAN             │
│    • Caso contrario: TVAE                   │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 3. CONFIGURACIÓN DE SEMILLAS                 │
│    Método: _set_global_seeds()               │
│    • Python random: seed=42                  │
│    • NumPy: seed=42                         │
│    • PyTorch: seed=42 (si disponible)       │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 4. GENERACIÓN DE DATOS SINTÉTICOS           │
│    Módulos:                                  │
│    • CTGANGenerator (GAN condicional)       │
│    • TVAEGenerator (Variational Autoencoder)│
│    • SDVGenerator (múltiples modelos)       │
│                                              │
│    Proceso:                                  │
│    • Preprocesamiento de datos              │
│    • Entrenamiento del modelo               │
│    • Sampling de nuevos registros           │
│    • Post-procesamiento                     │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 5. VALIDACIÓN BÁSICA                         │
│    • Verificar estructura (columnas)         │
│    • Verificar tipos de datos               │
│    • Verificar ausencia de NaN masivos      │
│    • Log de warnings                        │
└──────────────┬───────────────────────────────┘
               │
               ▼
OUTPUT: DataFrame Sintético + Metadata
```

### 🔬 Modelos de Generación

#### **CTGAN (Conditional GAN)**

**Archivo**: `src/generation/ctgan_generator.py`

**Algoritmo**:
1. **Discriminador**: Clasifica registros como reales o sintéticos
2. **Generador**: Aprende a crear registros que engañen al discriminador
3. **Condicional**: Usa conditional vectors para controlar generación

**Parámetros Importantes**:
```python
epochs: 300  # Número de iteraciones de entrenamiento
batch_size: 500  # Tamaño de lote
generator_dim: (256, 256)  # Capas del generador
discriminator_dim: (256, 256)  # Capas del discriminador
```

**Ventajas**:
- Excelente para datos categóricos
- Preserva bien distribuciones multimodales
- Buena para correlaciones complejas

**Desventajas**:
- Lento para entrenar
- Requiere muchos datos (> 500 filas recomendado)
- Puede generar outliers irreales

#### **TVAE (Variational Autoencoder)**

**Archivo**: `src/generation/tvae_generator.py`

**Algoritmo**:
1. **Encoder**: Comprime datos a espacio latente
2. **Latent Space**: Representación probabilística
3. **Decoder**: Reconstruye datos desde espacio latente

**Parámetros Importantes**:
```python
epochs: 300
batch_size: 500
compress_dims: (128, 128)  # Capas del encoder
decompress_dims: (128, 128)  # Capas del decoder
```

**Ventajas**:
- Más rápido que CTGAN
- Mejor para datos continuos
- Más estable en entrenamiento

**Desventajas**:
- Puede "suavizar" distribuciones discretas
- Menos efectivo con muchas categorías

#### **SDV (Synthetic Data Vault)**

**Archivo**: `src/generation/sdv_generator.py`

**Modelos Incluidos**:
- GaussianCopula: Más rápido, asume distribuciones normales
- CTGAN: Implementación de SDV
- CopulaGAN: Híbrido

**Ventajas**:
- Muy rápido
- Bueno para datasets pequeños (< 300 filas)
- Fácil de configurar

**Desventajas**:
- Menos flexible que CTGAN/TVAE puros
- Puede no capturar relaciones complejas

### 🎯 Selector de Columnas Médicas

**Archivo**: `src/adapters/medical_column_selector.py`

**Proceso**:
1. **Excluir automáticamente**:
   - Columnas ID (patient_id, id, identifier)
   - Timestamps puros (fechas sin relevancia clínica)
   - Columnas con 100% valores únicos (índices)

2. **Priorizar**:
   - Columnas médicas clave (edad, género, diagnóstico, signos vitales)
   - Columnas con keywords médicos (disease, symptom, treatment)
   - Columnas numéricas con varianza significativa

3. **Limitar**:
   - Máximo 30 columnas (evitar overfitting)
   - Balance 60% numéricas / 40% categóricas (aprox)

### 🔒 Reproducibilidad

**Semillas Globales**:
```python
seed = 42  # Configurable vía env GENERATOR_SEED
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)  # Si hay GPU
```

**Resultado**: Mismos datos sintéticos en ejecuciones sucesivas con mismo dataset e input.

### 🚀 Optimizaciones Recomendadas

1. **Caché de Modelos Entrenados**: Guardar modelo entrenado por hash de dataset
2. **Transfer Learning**: Pre-entrenar en datasets médicos grandes
3. **Hyperparameter Tuning**: Grid search para epochs, batch_size, learning_rate
4. **Validación Durante Entrenamiento**: Early stopping basado en métricas
5. **Generación Incremental**: Generar en lotes pequeños para datasets muy grandes
6. **Privacy Budget**: Implementar differential privacy (DP-CTGAN, DP-TVAE)

---

## 4. ✅ Agente Validador

### 📝 Descripción
El **Validador** verifica la coherencia médica y clínica de los datos sintéticos (o originales si no hay sintéticos).

### 🔧 Configuración

```python
Nombre: "Validador Médico"
LLM: GPT-4
Temperatura: 0.2
Max Tokens: 1500
Herramientas: Ninguna (usa módulos de validación)
```

### 📊 Flujo de Procesamiento

```
INPUT: DataFrame Sintético/Original + Contexto
   │
   ▼
┌──────────────────────────────────────────────┐
│ 1. DETERMINACIÓN DE MODO                     │
│    • Si hay synthetic_data: validar sintético│
│    • Si solo hay dataframe: validar original │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 2. VALIDACIÓN DE ESQUEMA JSON                │
│    Archivo: src/validation/json_schema.py    │
│    • Validar estructura de cada registro     │
│    • Verificar tipos de datos               │
│    • Verificar rangos permitidos            │
│    Schema: pacient_schema (JSONSchema)      │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 3. VALIDACIÓN DE REGLAS CLÍNICAS            │
│    Archivo: src/validation/clinical_rules.py │
│                                              │
│    Reglas COVID-19:                          │
│    • PCR Result: 'Positive' | 'Negative'    │
│    • Severity: Low/Medium/High/Critical     │
│    • Temperature: 35-42°C                   │
│    • SpO2: 70-100%                          │
│    • Days in Hospital: 0-60                 │
│    • Age: 0-120 años                        │
│                                              │
│    Reglas Generales:                         │
│    • Age >= 0                               │
│    • Gender: M/F/Other                      │
│    • BMI: 10-60                             │
│    • Blood Pressure: Sistólica > Diastólica │
│    • Heart Rate: 30-200 bpm                 │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 4. CÁLCULO DE COHERENCIA                     │
│    Método: _calculate_coherence_scores()     │
│                                              │
│    • Clinical Coherence:                     │
│      - Signos vitales coherentes (40%)      │
│      - Correlaciones demográficas (30%)     │
│      - Validez médica (30%)                 │
│                                              │
│    • Data Quality:                           │
│      - Errores de esquema (50%)             │
│      - Valores fuera de rango (30%)         │
│      - Datos faltantes (20%)                │
│                                              │
│    • Overall Score: Promedio ponderado      │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 5. GENERACIÓN DE INFORME (LLM)              │
│    • Resumen de validación                   │
│    • Lista de issues encontrados            │
│    • Recomendaciones de uso                 │
│    Output: Markdown                         │
└──────────────┬───────────────────────────────┘
               │
               ▼
OUTPUT: Informe Markdown + Métricas
```

### 🔬 Reglas Clínicas

**Archivo**: `src/validation/clinical_rules.py`

**Función Principal**: `validate_patient_case(patient_dict, is_covid)`

**Validaciones COVID-19**:
1. **PCR Result**: Debe ser 'Positive' o 'Negative'
2. **Severity**: Debe ser Low, Medium, High o Critical
3. **Temperature**: 35°C ≤ T ≤ 42°C
4. **SpO2**: 70% ≤ SpO2 ≤ 100%
5. **Days in Hospital**: 0 ≤ days ≤ 60
6. **Age**: 0 ≤ age ≤ 120
7. **Correlación Severity-SpO2**: Si Severity=Critical, SpO2 < 90%

**Validaciones Generales**:
1. **Age**: ≥ 0 años
2. **Gender**: M, F, Other, Male, Female
3. **BMI**: 10 ≤ BMI ≤ 60
4. **Blood Pressure**: systolic > diastolic
5. **Heart Rate**: 30 ≤ HR ≤ 200 bpm
6. **Temperature**: 35°C ≤ T ≤ 42°C

### 📊 Esquema JSON

**Archivo**: `src/validation/json_schema.py`

```python
pacient_schema = {
    "type": "object",
    "properties": {
        "Age": {"type": "integer", "minimum": 0, "maximum": 120},
        "Gender": {"type": "string", "enum": ["M", "F", "Male", "Female", "Other"]},
        "Temperature": {"type": "number", "minimum": 35, "maximum": 42},
        "SpO2": {"type": "number", "minimum": 70, "maximum": 100},
        # ... más propiedades
    },
    "required": ["Age"]  # Campos obligatorios
}
```

### 📈 Métricas de Validación

**Clinical Coherence** (0-1):
- **Signos Vitales Coherentes** (40%): % de registros con vitales en rango
- **Correlaciones Demográficas** (30%): Correlación edad-género, edad-diagnóstico
- **Validez Médica** (30%): % de registros que pasan reglas clínicas

**Data Quality** (0-1):
- **Errores de Esquema** (50%): % de registros que pasan validación JSON
- **Valores Fuera de Rango** (30%): % de valores numéricos en rangos válidos
- **Datos Faltantes** (20%): 1 - (% de valores nulos)

**Overall Score** (0-1):
```
overall_score = (clinical_coherence * 0.6) + (data_quality * 0.4)
```

### 🚀 Optimizaciones Recomendadas

1. **Reglas Configurables**: Cargar reglas desde archivo YAML/JSON
2. **Validación Paralela**: Usar multiprocessing para datasets grandes
3. **Reglas ML**: Entrenar modelo para detectar registros "sospechosos"
4. **Explicabilidad**: SHAP values para explicar por qué un registro es inválido
5. **Validación Incremental**: Solo validar registros nuevos/modificados
6. **Benchmarking**: Comparar con validaciones de expertos humanos

---

## 5. 📊 Agente Evaluador

### 📝 Descripción
El **Evaluador** mide la calidad, fidelidad y utilidad de los datos sintéticos comparándolos con los originales.

### 🔧 Configuración

```python
Nombre: "Evaluador de Utilidad"
LLM: GPT-4
Temperatura: 0.2
Max Tokens: 4000  # Incrementado para informes completos
Herramientas: Ninguna (usa módulos de evaluación)
```

### 📊 Flujo de Procesamiento

```
INPUT: DataFrame Original + DataFrame Sintético
   │
   ▼
┌──────────────────────────────────────────────┐
│ 1. FIDELIDAD ESTADÍSTICA                    │
│    Archivo: src/evaluation/evaluator.py      │
│                                              │
│    • Preservación de Correlaciones:          │
│      - Calcular matriz de correlación orig  │
│      - Calcular matriz de correlación synt  │
│      - Métrica: 1 - mean_absolute_error     │
│                                              │
│    • Similaridad de Distribuciones:          │
│      - KS Test para numéricas               │
│      - Chi-squared para categóricas         │
│      - Métrica: % de columnas similares     │
│                                              │
│    • Cobertura de Valores Únicos:            │
│      - Ratio valores únicos synt/orig       │
│      - Por columna categórica               │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 2. UTILIDAD PARA MACHINE LEARNING           │
│    Función: evaluate_ml_performance()        │
│                                              │
│    Proceso:                                  │
│    • Detectar columna target (diagnóstico)  │
│    • Entrenar Random Forest en ORIGINAL     │
│    • Evaluar en test set ORIGINAL           │
│    • Entrenar Random Forest en SINTÉTICO    │
│    • Evaluar en test set ORIGINAL           │
│                                              │
│    Métricas:                                 │
│    • F1 Score (Original)                    │
│    • F1 Score (Sintético)                   │
│    • F1 Preservation: F1_synt / F1_orig     │
│    • Accuracy Preservation                  │
│    • Precision/Recall Preservation          │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 3. EXTRACCIÓN DE ENTIDADES MÉDICAS         │
│    Función: evaluate_medical_entities()      │
│                                              │
│    • Detectar entidades médicas con NER     │
│      (diagnósticos, síntomas, tratamientos) │
│    • Comparar frecuencia y distribución     │
│    • Calcular F1 de preservación            │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 4. SCORE DE PRIVACIDAD                      │
│    • Distance to Closest Record (DCR)       │
│    • Nearest Neighbor Distance Ratio (NNDR) │
│    • Métrica: % de registros "alejados"     │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 5. CÁLCULO DE SCORE FINAL                   │
│    final_score = (                           │
│      fidelity * 0.40 +                       │
│      ml_utility * 0.35 +                     │
│      privacy * 0.25                          │
│    )                                         │
│                                              │
│    Tiers:                                    │
│    • Excellent: ≥ 0.85                      │
│    • Good: 0.70 - 0.85                      │
│    • Acceptable: 0.50 - 0.70                │
│    • Poor: < 0.50                           │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 6. GENERACIÓN DE INFORME (LLM)              │
│    • Certificación de calidad                │
│    • Tabla de métricas detalladas           │
│    • Análisis de fidelidad estadística      │
│    • Análisis de utilidad ML                │
│    • Recomendaciones de uso                 │
│    Output: Markdown (hasta 4000 tokens)     │
└──────────────┬───────────────────────────────┘
               │
               ▼
OUTPUT: Informe Markdown + Métricas Completas
```

### 🔬 Métricas Detalladas

#### **Fidelidad Estadística**

**1. Preservación de Correlaciones**:
```python
corr_orig = df_original.corr()
corr_synt = df_synthetic.corr()
mae = np.mean(np.abs(corr_orig - corr_synt))
correlation_preservation = 1 - mae
```

**2. Similaridad de Distribuciones**:
- **Numéricas**: Kolmogorov-Smirnov Test (p-value > 0.05 = similar)
- **Categóricas**: Chi-squared Test (p-value > 0.05 = similar)
```python
similar_columns = []
for col in numeric_cols:
    ks_stat, p_value = ks_2samp(orig[col], synt[col])
    if p_value > 0.05:
        similar_columns.append(col)
distribution_similarity = len(similar_columns) / len(numeric_cols)
```

**3. Cobertura de Valores Únicos**:
```python
for col in categorical_cols:
    unique_orig = set(orig[col].unique())
    unique_synt = set(synt[col].unique())
    coverage = len(unique_synt & unique_orig) / len(unique_orig)
```

#### **Utilidad para Machine Learning**

**Modelo**: Random Forest Classifier (n_estimators=100)

**Proceso**:
1. Split original: 80% train, 20% test
2. Train RF en **original train**
3. Evaluate en **original test** → F1_baseline
4. Train RF en **synthetic data**
5. Evaluate en **original test** → F1_synthetic
6. Calcular preservation: F1_synthetic / F1_baseline

**Interpretación**:
- **≥ 0.90**: Excelente utilidad, modelo sintético casi igual
- **0.70-0.90**: Buena utilidad, modelo sintético útil
- **0.50-0.70**: Utilidad aceptable, modelo sintético degradado
- **< 0.50**: Utilidad pobre, modelo sintético no es útil

#### **Score de Privacidad**

**Distance to Closest Record (DCR)**:
- Para cada registro sintético, encontrar el registro original más cercano
- Métrica: Distancia euclidiana normalizada
- Score alto = Mayor privacidad

**Umbral de Privacidad**:
- Si DCR < threshold → Registro "muy similar" a original (riesgo de re-identificación)
- Privacy Score = % de registros con DCR > threshold

### 🚀 Optimizaciones Recomendadas

1. **Métricas Adicionales**:
   - Propensity Score (PATE)
   - Membership Inference Attack resistance
   - Discriminator Score (AUC de GAN)
   
2. **Evaluación Paralela**: Calcular métricas en paralelo (multiprocessing)

3. **Caché de Modelos ML**: Guardar modelo entrenado en original para comparaciones

4. **Visualizaciones**: Generar plots (PCA, t-SNE) para comparar distribuciones

5. **Cross-Validation**: Usar k-fold CV para ML utility (más robusto)

6. **Domain-Specific Metrics**: Métricas médicas específicas (survival curves, ROC para diagnósticos)

---

## 6. 🔬 Agente Simulador

### 📝 Descripción
El **Simulador** genera evoluciones temporales realistas de pacientes basándose en datos médicos.

### 🔧 Configuración

```python
Nombre: "Simulador de Pacientes"
LLM: GPT-4
Temperatura: 0.2
Max Tokens: 3000  # Incrementado para informes completos
Herramientas: Ninguna (usa módulo de simulación)
```

### 📊 Flujo de Procesamiento

```
INPUT: DataFrame (Sintético o Original) + Contexto
   │
   ▼
┌──────────────────────────────────────────────┐
│ 1. DETECCIÓN DE TIPO DE ENFERMEDAD          │
│    • Extraer de universal_analysis            │
│    • Tipos: COVID-19, General                │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 2. INICIALIZACIÓN DEL SIMULADOR             │
│    Archivo: src/simulation/progress_         │
│            simulator.py                       │
│    Clase: ProgressSimulator                  │
│                                              │
│    Parámetros:                               │
│    • data: DataFrame base                    │
│    • disease_type: "covid19" | "general"    │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 3. SIMULACIÓN BATCH                          │
│    Método: simulate_batch_evolution()        │
│                                              │
│    Para cada paciente:                       │
│    • Extraer estado inicial                  │
│    • Determinar número de visitas (2-6)     │
│    • Para cada visita:                       │
│      - Simular evolución de signos vitales  │
│      - Actualizar severidad                 │
│      - Agregar ruido realista               │
│      - Aplicar transiciones de estado       │
│    • Retornar registro de visitas           │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 4. CÁLCULO DE ESTADÍSTICAS                  │
│    • Total de visitas generadas              │
│    • Promedio visitas por paciente          │
│    • Pacientes con mejoría                  │
│    • Pacientes con deterioro                │
│    • Pacientes estables                     │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 5. GENERACIÓN DE INFORME (LLM)              │
│    • Resumen de simulación                   │
│    • Estadísticas de evolución              │
│    • Análisis de realismo                   │
│    Output: Markdown                         │
└──────────────┬───────────────────────────────┘
               │
               ▼
OUTPUT: DataFrame Evolucionado + Estadísticas
```

### 🔬 Motor de Simulación

**Archivo**: `src/simulation/progress_simulator.py`

**Clase**: `ProgressSimulator`

**Método Principal**: `simulate_batch_evolution(data)`

### 🧬 Modelo de Evolución COVID-19

**Parámetros de Simulación**:
```python
num_visits: random(2, 6)  # Número de visitas por paciente
time_delta: random(1, 5) days  # Tiempo entre visitas
```

**Variables Simuladas**:
1. **Temperature** (Temperatura):
   - Inicio: valor del dataset
   - Evolución: ±0.5°C por visita (tendencia a normalizar)
   - Rango: 35.5°C - 41.5°C

2. **SpO2** (Saturación de Oxígeno):
   - Inicio: valor del dataset
   - Mejoría: +2 a +5 puntos por visita
   - Deterioro: -1 a -3 puntos por visita
   - Rango: 75% - 100%

3. **Severity** (Severidad):
   - Estados: Low → Medium → High → Critical
   - Transiciones basadas en SpO2:
     - SpO2 > 95%: mejora hacia Low
     - SpO2 < 85%: empeora hacia Critical
   - Probabilidad de transición: 30% por visita

4. **Days in Hospital**:
   - Incremento acumulativo: +1 a +5 días por visita
   - Máximo: 60 días

### 🧬 Modelo de Evolución General

Para datasets no-COVID:
- Evolución basada en correlaciones detectadas
- Ruido gaussiano con std = 0.1 * valor_inicial
- Tendencia hacia valores normales (media del dataset)

### 📊 Estadísticas de Simulación

```python
stats = {
    "total_visits": int,  # Total de registros generados
    "avg_visits_per_patient": float,  # Promedio de visitas
    "patients_with_improvement": int,  # Pacientes que mejoraron
    "patients_with_deterioration": int,  # Pacientes que empeoraron
    "patients_stable": int  # Pacientes sin cambio significativo
}
```

**Criterios**:
- **Mejoría**: SpO2_final > SpO2_inicial + 5 O Severity disminuyó
- **Deterioro**: SpO2_final < SpO2_inicial - 5 O Severity aumentó
- **Estable**: Sin cambios significativos

### 🚀 Optimizaciones Recomendadas

1. **Modelos de Transición Aprendidos**:
   - Usar Markov Chains aprendidas de datos reales
   - Hidden Markov Models para estados latentes
   - RNNs/LSTMs para secuencias temporales

2. **Simulación Basada en Agentes (ABM)**:
   - Modelar interacciones entre pacientes (contagios)
   - Simular recursos hospitalarios (camas, ventiladores)
   - Eventos estocásticos (complicaciones)

3. **Validación con Datos Reales**:
   - Comparar distribuciones de evoluciones simuladas vs reales
   - Métricas de realismo (KS test, Chi-squared)
   - Expert evaluation (médicos revisan evoluciones)

4. **Generación de Counterfactuals**:
   - "¿Qué pasaría si el paciente hubiera recibido tratamiento X?"
   - Útil para investigación de efectividad de tratamientos

5. **Simulación Multiescala**:
   - Nivel celular (carga viral, respuesta inmune)
   - Nivel organismo (signos vitales)
   - Nivel población (epidemiología)

---

## 📊 Flujo de Procesamiento Completo

### Escenario: Análisis → Generación → Validación → Evaluación → Simulación

```
1. USUARIO: "Analizar dataset de COVID"
   │
   ▼
2. COORDINADOR: Detecta intención "comando", agent="analyzer"
   │
   ▼
3. ANÁLISIS UNIVERSAL: Detecta tipo COVID-19, mapea columnas
   │
   ▼
4. ANALIZADOR: Genera informe EDA en Markdown
   │
   └─→ RESULTADO: Informe con estadísticas, calidad, recomendaciones
   
5. USUARIO: "Generar 500 datos sintéticos con CTGAN"
   │
   ▼
6. COORDINADOR: Detecta "comando", agent="generator", params={model: CTGAN, n: 500}
   │
   ▼
7. SELECTOR DE COLUMNAS: Excluye IDs, selecciona 25 columnas médicas relevantes
   │
   ▼
8. GENERADOR CTGAN: Entrena GAN, genera 500 registros sintéticos
   │
   └─→ RESULTADO: DataFrame sintético + metadata (modelo, seed, timestamp)
   
9. USUARIO: "Validar datos sintéticos"
   │
   ▼
10. COORDINADOR: Detecta "comando", agent="validator"
    │
    ▼
11. VALIDADOR: Aplica reglas clínicas COVID-19, valida esquema JSON
    │
    └─→ RESULTADO: Informe de coherencia (85%), lista de issues
    
12. USUARIO: "Evaluar calidad de datos sintéticos"
    │
    ▼
13. COORDINADOR: Detecta "comando", agent="evaluator"
    │
    ▼
14. EVALUADOR: Calcula fidelidad, utilidad ML, privacidad
    │
    └─→ RESULTADO: Informe completo (Score: 82%, Tier: Good)
    
15. USUARIO: "Simular evolución de pacientes"
    │
    ▼
16. COORDINADOR: Detecta "comando", agent="simulator"
    │
    ▼
17. SIMULADOR: Genera 2-6 visitas por paciente, evoluciona signos vitales
    │
    └─→ RESULTADO: DataFrame evolucionado (2,341 visitas), stats de mejoría/deterioro
```

---

## 🛠️ Módulos de Soporte

### 1. **Universal Dataset Detector**
**Archivo**: `src/adapters/universal_dataset_detector.py`

**Función**: Detectar automáticamente tipo y estructura de datasets médicos

**Capabilities**:
- Detección de tipo: COVID-19, Diabetes, Cardiología, General
- Mapeo inteligente de columnas (age, gender, diagnosis, etc.)
- Análisis estadístico completo
- Detección de patrones médicos

### 2. **Medical Column Selector**
**Archivo**: `src/adapters/medical_column_selector.py`

**Función**: Seleccionar columnas relevantes para generación sintética

**Criteria**:
- Excluir IDs y timestamps
- Priorizar columnas médicas clave
- Balance numérico/categórico
- Límite de 30 columnas

### 3. **Clinical Rules Validator**
**Archivo**: `src/validation/clinical_rules.py`

**Función**: Validar coherencia médica de registros

**Rules**:
- Rangos de signos vitales
- Correlaciones esperadas (edad-género, severity-SpO2)
- Validez de diagnósticos y tratamientos

### 4. **ML Utility Evaluator**
**Archivo**: `src/evaluation/evaluator.py`

**Función**: Evaluar utilidad de datos sintéticos para ML

**Methods**:
- Random Forest training/testing
- F1, Accuracy, Precision, Recall preservation
- Cross-validation

### 5. **Progress Simulator**
**Archivo**: `src/simulation/progress_simulator.py`

**Función**: Simular evoluciones temporales de pacientes

**Models**:
- COVID-19 progression model
- General disease progression
- Stochastic transitions

---

## ⚙️ Configuración y Parámetros

### Variables de Entorno

```bash
# LLM Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4  # o gpt-3.5-turbo
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2500

# Agent Configuration
MEMORY_K=8  # Memoria de chat (últimos K mensajes)
GENERATOR_SEED=42  # Seed para reproducibilidad

# Data Processing
MAX_SAMPLE_SIZE=5000  # Muestreo para datasets grandes
MAX_COLUMNS_FOR_GENERATION=30  # Límite de columnas

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

### Parámetros de Generación

```python
# CTGAN
epochs: 300
batch_size: 500
generator_dim: (256, 256)
discriminator_dim: (256, 256)
generator_lr: 2e-4
discriminator_lr: 2e-4

# TVAE
epochs: 300
batch_size: 500
compress_dims: (128, 128)
decompress_dims: (128, 128)
l2scale: 1e-5

# SDV
model: 'GaussianCopula'  # o 'CTGAN', 'CopulaGAN'
```

### Parámetros de Validación

```python
# Umbrales
min_age: 0
max_age: 120
min_temperature: 35.0
max_temperature: 42.0
min_spo2: 70
max_spo2: 100

# Pesos de coherencia
vital_signs_weight: 0.40
demographic_correlation_weight: 0.30
medical_validity_weight: 0.30
```

### Parámetros de Evaluación

```python
# ML Utility
test_size: 0.2  # 20% para test
random_state: 42
n_estimators: 100  # Random Forest

# Privacy
dcr_threshold: 0.1  # Distance to Closest Record

# Weights para score final
fidelity_weight: 0.40
ml_utility_weight: 0.35
privacy_weight: 0.25
```

---

## 🚀 Oportunidades de Optimización

### 1. **Performance**

#### A. Caché y Memoización
- **Análisis**: Cachear análisis por hash de dataset
- **Modelos ML**: Guardar modelos entrenados (pickle)
- **Generación**: Cachear modelos CTGAN/TVAE entrenados

#### B. Paralelización
- **Validación**: Validar registros en paralelo (multiprocessing)
- **Evaluación**: Calcular métricas en paralelo
- **Simulación**: Simular pacientes en paralelo

#### C. Muestreo Adaptativo
- Datasets > 10,000 filas: muestrear para análisis
- Mantener estratificación por target
- Reducir tiempo de O(n²) a O(1)

### 2. **Calidad de Generación**

#### A. Hyperparameter Tuning
- Grid search para epochs, batch_size, learning_rate
- Early stopping basado en validation loss
- Bayesian Optimization para búsqueda eficiente

#### B. Transfer Learning
- Pre-entrenar en datasets médicos grandes (MIMIC-III)
- Fine-tuning en dataset específico
- Reducir tiempo de entrenamiento 50%

#### C. Differential Privacy
- Implementar DP-CTGAN, DP-TVAE
- Balance privacidad-utilidad configurable
- Privacy budget (epsilon) ajustable

### 3. **Validación y Evaluación**

#### A. Métricas Avanzadas
- **Propensity Score**: PATE score para privacidad
- **Discriminator Score**: AUC del discriminador GAN
- **Membership Inference**: Resistencia a ataques

#### B. Validación con Expertos
- Interface para que médicos revisen registros
- Recolectar feedback (realista / no realista)
- Fine-tuning basado en feedback

#### C. Domain-Specific Metrics
- Survival curves (Kaplan-Meier)
- ROC/AUC para diagnósticos específicos
- Time-to-event analysis

### 4. **Simulación**

#### A. Modelos Aprendidos
- Entrenar HMM en evoluciones reales
- RNNs/LSTMs para secuencias temporales
- Attention mechanisms para long-term dependencies

#### B. Simulación Basada en Agentes
- Modelar interacciones (contagios)
- Recursos hospitalarios (camas, personal)
- Eventos estocásticos (complicaciones)

#### C. Counterfactuals
- "¿Qué pasaría si el paciente recibiera tratamiento X?"
- Útil para investigación de efectividad
- Causal inference methods

### 5. **Arquitectura**

#### A. Microservicios
- Separar agentes en servicios independientes
- Escalabilidad horizontal
- Tolerancia a fallos

#### B. Queue System
- RabbitMQ / Celery para tareas asíncronas
- Procesamiento en background
- Mejor experiencia de usuario

#### C. Caching Layer
- Redis para caché distribuido
- Compartir entre instancias
- TTL configurable

---

## 📚 Consideraciones para Investigación

### 1. **Reproducibilidad**

✅ **Implementado**:
- Seeds globales (random, numpy, torch)
- Logging exhaustivo de parámetros
- Metadata de generación (modelo, seed, timestamp)

🔄 **Por Implementar**:
- Versionado de datasets (DVC)
- Versionado de modelos (MLflow)
- Experimentación sistemática (Weights & Biases)

### 2. **Trazabilidad**

✅ **Implementado**:
- Logging centralizado por agente
- Contexto pasado entre agentes
- Historial de operaciones

🔄 **Por Implementar**:
- Database de experimentos (PostgreSQL)
- Visualización de flujos (DAG)
- Auditoría completa de decisiones

### 3. **Validación Científica**

✅ **Implementado**:
- Métricas estándar (F1, Accuracy)
- Comparación con baselines
- Coherencia médica

🔄 **Por Implementar**:
- Statistical significance tests (t-test, Mann-Whitney)
- Cross-validation robusta (k-fold, stratified)
- Confidence intervals para métricas
- Comparación con sota methods (SDV benchmarks)

### 4. **Documentación**

✅ **Implementado**:
- Docstrings en funciones
- Logging de parámetros
- Informes en Markdown

🔄 **Por Mejorar**:
- Auto-documentación (Sphinx)
- Diagramas de flujo (Mermaid)
- Paper-ready tables/figures
- Jupyter notebooks de ejemplos

### 5. **Ética y Privacidad**

✅ **Implementado**:
- Exclusión de IDs
- Métricas de privacidad (DCR)
- Validación de coherencia médica

🔄 **Por Implementar**:
- Differential Privacy formal
- K-anonymity checks
- Informed consent tracking
- IRB compliance checks

### 6. **Benchmarking**

🔄 **Por Implementar**:
- Comparación con métodos baseline:
  - SMOTE, ADASYN (oversampling)
  - Gaussian Copula (SDV)
  - Otros GANs (WGAN, StyleGAN)
- Datasets estándar:
  - UCI ML Repository (Diabetes, Heart Disease)
  - Kaggle medical datasets
  - MIMIC-III (si hay acceso)
- Métricas estándar:
  - SDV metrics suite
  - TSTR (Train on Synthetic, Test on Real)
  - Privacy metrics (k-anonymity, l-diversity)

---

## 📊 Resumen de Flujo por Agente

| Agente | Input | Procesamiento | Output | Tiempo Aprox |
|--------|-------|---------------|--------|--------------|
| **Coordinador** | Mensaje + Contexto | LLM classification | JSON decision | < 2s |
| **Analizador** | DataFrame | Universal Analysis + LLM | Informe EDA (MD) | 5-15s |
| **Generador** | DataFrame + Params | Column Selection + CTGAN/TVAE | DataFrame sintético | 30-300s |
| **Validador** | DataFrame sintético | Rules + Schema validation | Informe validación | 3-10s |
| **Evaluador** | DF orig + DF synt | Fidelity + ML + Privacy | Informe evaluación | 10-60s |
| **Simulador** | DataFrame | Evolution model | DF evolucionado | 5-20s |

---

## 🎓 Conclusión

El sistema Patient-IA implementa una arquitectura multi-agente robusta para el procesamiento de datos médicos sintéticos. Cada agente tiene responsabilidades claras y está optimizado para su tarea específica.

**Puntos Fuertes**:
✅ Arquitectura modular y escalable  
✅ Reproducibilidad garantizada (seeds)  
✅ Métricas completas (fidelidad, utilidad, privacidad)  
✅ Validación médica rigurosa  
✅ Simulaciones realistas  

**Áreas de Mejora**:
🔄 Optimización de performance (caché, paralelización)  
🔄 Métricas avanzadas (differential privacy)  
🔄 Modelos aprendidos (HMM, RNN)  
🔄 Validación con expertos humanos  
🔄 Benchmarking contra SOTA  

Este documento debe servir como referencia técnica para la investigación y guía para futuras optimizaciones del sistema.

---

**Autor**: Sistema Patient-IA  
**Versión**: 1.0  
**Fecha**: 15 de Octubre, 2025  
**Licencia**: MIT
