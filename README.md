## Descripción

**Patientia** es una herramienta avanzada para la generación de datos sintéticos en el ámbito de la salud, especialmente diseñada para crear historias clínicas artificiales que mantienen las características estadísticas y clínicas de los datos reales sin comprometer la privacidad de los pacientes.

Este proyecto utiliza técnicas de inteligencia artificial, incluyendo modelos generativos (SDV, CTGAN) y LLMs, para producir datos clínicos sintéticos de alta calidad que pueden utilizarse para investigación, formación médica, y desarrollo de soluciones tecnológicas en salud.

## Características Principales

- 🧬 **Generación Sintética Tabular**: Creación de datasets estructurados que preservan correlaciones y distribuciones estadísticas.
- 🔍 **Análisis Exploratorio**: Herramientas para comprender y visualizar patrones en datos clínicos.
- 🤖 **Agentes de IA Especializados**: Sistema multi-agente para análisis, generación, validación y evaluación.
- 📊 **Interfaz Interactiva**: Dashboard web para interactuar con los datos y generar nuevos registros.
- ✅ **Validación Clínica**: Verificación automática de la coherencia médica de los datos generados.
- 📈 **Evaluación de Calidad**: Métricas para valorar la utilidad y el realismo de los datos sintéticos.

## Requisitos

- **Python**: 3.11 o superior
- **Sistema Operativo**: Compatible con Windows, macOS y Linux
- **Memoria**: Mínimo 8GB RAM (16GB+ recomendado)
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
# En Windows
venv\Scripts\activate
# En macOS/Linux
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Configuración de Variables de Entorno

Para funcionalidades avanzadas que utilizan LLMs, crea un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```
# Azure OpenAI
AZURE_OPENAI_API_KEY="your_api_key_here"
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_DEPLOYMENT="your_deployment_name"
AZURE_OPENAI_API_VERSION="2024-02-01"
```

## Uso

### 1. Interfaz de Chat para Datos Sintéticos

Para iniciar la aplicación, ejecuta la interfaz de Streamlit:

```bash
streamlit run interfaces/chat_llm.py
```

Este comando abrirá una aplicación web donde podrás:
- Cargar archivos de datos reales.
- Analizar las características clínicas.
- Generar datos sintéticos.
- Validar la calidad médica de los datos.
- Descargar los resultados.

### 2. Análisis de Datos (Jupyter Notebooks)

Para explorar los notebooks de análisis y modelado:

```bash
# Iniciar Jupyter Lab o Jupyter Notebook
jupyter lab
# o
jupyter notebook
```

Una vez iniciado, navega a la carpeta `notebooks/` para explorar los análisis existentes, como:
- `EDA.ipynb`: Análisis exploratorio de datos.
- `umap_hdbscan_faiss.ipynb`: Reducción de dimensionalidad y clustering.
- `FAISS.ipynb`: Búsqueda de similitud en datos clínicos.

## Estructura del Proyecto

```
R35_sopra_steria/
├── data/
│   ├── real/                   # Datasets originales procesados
│   └── synthetic/              # Cohortes sintéticas generadas
├── docs/                       # Documentación del proyecto
├── interfaces/                 # Interfaces de usuario (Streamlit, CLI)
├── models/                     # Modelos generativos entrenados (checkpoints)
├── notebooks/                  # Notebooks para análisis y experimentación
├── outputs/                    # Salidas generadas (historias clínicas, informes)
├── src/
│   ├── agents/                 # Sistema multi-agente (analizador, generador, etc.)
│   ├── config/                 # Configuración de Azure y otros servicios
│   ├── evaluation/             # Módulos para evaluar la calidad de los datos
│   ├── extraction/             # Extracción de patrones y características
│   ├── generation/             # Módulos de generación (SDV, CTGAN, TVAE)
│   ├── narration/              # Generación de texto narrativo con LLMs
│   ├── orchestration/          # Orquestación del flujo de trabajo con LangGraph
│   ├── simulation/             # Simulación de la progresión de pacientes

│   ├── utils/                  # Funciones y herramientas auxiliares
│   └── validation/             # Reglas de validación clínica y de esquema
├── tests/                      # Pruebas unitarias y de integración
├── .gitignore                  # Archivos y carpetas ignorados por Git
├── Pipfile                     # Dependencias del proyecto para Pipenv
├── requirements.txt            # Lista de dependencias para pip
└── README.md                   # Este archivo
```

## Componentes Principales

### Módulo de Generación (`src/generation`)
El núcleo del sistema utiliza modelos generativos como **SDV (Synthetic Data Vault)**, **CTGAN (Conditional Tabular GAN)** y **TVAE (Tabular VAE)** para producir datos sintéticos de alta fidelidad.

### Sistema Multi-Agente (`src/agents`)
La arquitectura se basa en agentes especializados con responsabilidades claras:
- **Coordinador**: Orquesta el flujo de trabajo global.
- **Analizador**: Extrae patrones de los datos originales.
- **Generador**: Crea los datos sintéticos.
- **Validador**: Verifica la coherencia clínica y estructural.
- **Simulador**: Modela la evolución temporal de los pacientes.
- **Evaluador**: Mide la calidad de los datos generados.

### Interfaz de Usuario (`interfaces/`)
Utilizamos **Streamlit** para crear una interfaz web intuitiva que permite a los usuarios interactuar con el sistema, generar datos y visualizar resultados sin necesidad de conocimientos técnicos avanzados.

## Soporte y Mantenimiento

- **Problemas y sugerencias**: Abrir un *issue* en el repositorio de GitHub.
- **Documentación**: Consultar la carpeta `docs/` para guías detalladas.

## Licencia

Este proyecto está licenciado bajo los términos especificados en el archivo `LICENSE`.

## Agradecimientos

Este proyecto fue desarrollado como parte de un Trabajo Fin de Máster (TFM) en colaboración con Sopra Steria. Agradecemos a todos los colaboradores y mentores que han hecho posible este desarrollo.

---

&copy; 2024 Patientia - Generador de Historias Clínicas Sintéticas