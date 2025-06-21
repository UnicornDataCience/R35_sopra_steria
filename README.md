
Workspace: Collecting workspace information# Patientia - Generador de Historias Clínicas Sintéticas

## Descripción

**Patientia** es una herramienta avanzada para la generación de datos sintéticos en el ámbito de la salud, especialmente diseñada para crear historias clínicas artificiales que mantienen las características estadísticas y clínicas de los datos reales sin comprometer la privacidad de los pacientes.

Este proyecto utiliza técnicas de inteligencia artificial, incluyendo modelos generativos (SDV, CTGAN) y LLMs, para producir datos clínicos sintéticos de alta calidad que pueden utilizarse para investigación, formación médica, y desarrollo de soluciones tecnológicas en salud.

## Características Principales

- 🧬 **Generación Sintética Tabular**: Creación de datasets estructurados que preservan correlaciones y distribuciones estadísticas
- 🔍 **Análisis Exploratorio**: Herramientas para comprender y visualizar patrones en datos clínicos
- 🤖 **Agentes de IA Especializados**: Sistema multi-agente para análisis, generación, validación y evaluación
- 📊 **Interfaz Interactiva**: Dashboard web para interactuar con los datos y generar nuevos registros
- ✅ **Validación Clínica**: Verificación automática de la coherencia médica de los datos generados
- 📈 **Evaluación de Calidad**: Métricas para valorar la utilidad y el realismo de los datos sintéticos

## Requisitos

- **Python**: 3.12.7 o superior
- **Sistema Operativo**: Compatible con Windows, macOS y Linux
- **Memoria**: Mínimo 4GB RAM (8GB+ recomendado)
- **Espacio en disco**: 2GB para instalación completa
- **API Key**: Cuenta en Azure OpenAI (opcional para funcionalidades LLM)

## Instalación

### Opción 1: Usando Pipenv (recomendado)

```bash
# Clonar el repositorio
git clone https://github.com/username/R35_sopra_steria.git
cd R35_sopra_steria

# Instalar dependencias con Pipenv
pipenv install

# Activar el entorno virtual
pipenv shell
```

### Opción 2: Usando pip

```bash
# Clonar el repositorio
git clone https://github.com/username/R35_sopra_steria.git
cd R35_sopra_steria

# Crear un entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración de Variables de Entorno

Para funcionalidades avanzadas que utilizan LLMs, crea un archivo .env en la raíz del proyecto:

```
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-01
```

## Uso

### 1. Interfaz de Chat para Datos Sintéticos

```bash
# Iniciar la interfaz de chat
streamlit run interfaces/chat_llm.py
```

Este comando abrirá una aplicación web donde podrás:
- Cargar archivos de datos reales
- Analizar las características clínicas
- Generar datos sintéticos
- Validar calidad médica
- Descargar resultados

### 2. Generación por Línea de Comandos

```bash
# Generar datos sintéticos desde línea de comandos
python src/generation/sdv_generator.py --input data/real/df_final.csv --output data/synthetic/ --rows 1000
```

### 3. Análisis de Datos (Jupyter Notebooks)

```bash
# Iniciar Jupyter para explorar los notebooks
jupyter notebook notebooks/
```

Explora los notebooks para:
- EDA.ipynb - Análisis exploratorio de datos
- umap_hdbscan_faiss.ipynb - Reducción de dimensionalidad y clustering
- FAISS.ipynb - Búsqueda de similitud en datos clínicos

## Estructura del Proyecto

```
r35_historia_clinica_sintetica/
├── configs/                    # Parámetros YAML, configuración de agentes y generación
├── data/
│   ├── real/                   # Dataset original procesado
│   └── synthetic/              # Cohortes sintéticas generadas
├── docs/                       # Documentación del proyecto
├── models/                     # Modelos entrenados o checkpoints LLM
├── notebooks/                  # Análisis y validaciones exploratorias
├── outputs/                    # Historias clínicas generadas, informes de validación
├── src/
│   ├── agents/                 # Sistema multi-agente para procesamiento de datos
│   ├── extraction/             # Extracción y vectorización de patrones
│   ├── generation/             # Módulos SDV, CTGAN, creación de pacientes
│   ├── validation/             # Verificación clínica automatizada
│   ├── simulation/             # Evolución temporal sintética
│   ├── narration/              # Prompts y generación de texto con LLM
│   ├── evaluation/             # Métricas de utilidad, realismo, plausibilidad
│   └── orchestration/          # Grafo LangGraph y control de agentes
├── interfaces/                 # Interfaces de usuario (Streamlit, CLI)
├── tests/                      # Tests unitarios y de integración
└── utils/                      # Utilidades y herramientas auxiliares
```

## Componentes Principales

### Módulo de Generación

El núcleo del sistema utiliza modelos generativos como SDV (Synthetic Data Vault), CTGAN (GAN condicional tabular), y técnicas de impresión basadas en LLM para producir datos sintéticos de alta fidelidad estadística y clínica.

### Sistema Multi-Agente

La arquitectura se basa en agentes especializados con responsabilidades específicas:

- **Coordinador**: Orquesta el flujo de trabajo global
- **Analizador**: Extrae patrones de los datos originales
- **Generador**: Crea los datos sintéticos
- **Validador**: Verifica la coherencia clínica
- **Simulador**: Modela la evolución temporal de pacientes
- **Evaluador**: Mide la calidad de los datos generados

### Interfaz de Usuario

Utilizamos Streamlit para crear una interfaz intuitiva que permite a usuarios sin conocimientos técnicos interactuar con el sistema a través de comandos en lenguaje natural y visualizaciones interactivas.

## Ejemplos de Uso

### Ejemplo 1: Generar cohorte de pacientes COVID-19

```python
from src.generation.sdv_generator import SDVGenerator
import pandas as pd

# Cargar datos reales
real_data = pd.read_csv("data/real/df_final.csv")

# Configurar generador
generator = SDVGenerator()

# Generar datos sintéticos
synthetic_data = generator.generate(real_data, num_samples=500)

# Guardar resultados
synthetic_data.to_csv("data/synthetic/pacientes_covid_sinteticos.csv", index=False)

# Obtener score de calidad
quality_score = generator.get_quality_score()
print(f"Calidad de los datos generados: {quality_score:.2%}")
```

### Ejemplo 2: Validar coherencia médica

```python
from src.validation.medical_validator import MedicalValidator
import pandas as pd

# Cargar datos sintéticos
synthetic_data = pd.read_csv("data/synthetic/datos_sinteticos.csv")

# Inicializar validador
validator = MedicalValidator()

# Validar coherencia médica
validation_results = validator.validate(synthetic_data)

# Ver inconsistencias detectadas
validation_results.summary()
```

## Soporte y Mantenimiento

Para obtener ayuda con Patientia:

- **Problemas y sugerencias**: Abrir un issue en el repositorio de GitHub
- **Contacto**: [correo_de_contacto@ejemplo.com]
- **Documentación**: Ver carpeta docs para guías detalladas

## Licencia

Este proyecto está licenciado bajo los términos especificados en el archivo LICENSE.

## Agradecimientos

Este proyecto fue desarrollado como parte de un Trabajo Fin de Máster (TFM) en colaboración con Sopra Steria. Agradecemos a todos los colaboradores y mentores que han hecho posible este desarrollo.

---

&copy; 2024 Patientia - Generador de Historias Clínicas Sintéticas

