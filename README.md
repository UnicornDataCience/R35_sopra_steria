# R35_sopra_steria
TFM

Python 3.12.7

Librerías: 
scikit-learn
pandas
sdv
transformers
langgraph
numpy

Directorio de trabajo:
r35_historia_clinica_sintetica/
├── configs/                    # Parámetros YAML, configuración de agentes y generación
├── data/
│   ├── real/                   # Dataset original procesado
│   └── synthetic/              # Cohortes sintéticas generadas
├── models/                     # Modelos entrenados o checkpoints LLM
├── notebooks/                  # Análisis y validaciones exploratorias
├── outputs/                    # Historias clínicas generadas, informes de validación
├── src/
│   ├── extraction/             # Extracción y vectorización de patrones
│   ├── generation/             # Módulos SDV, CTGAN, creación de pacientes
│   ├── validation/             # Verificación clínica automatizada
│   ├── simulation/             # Evolución temporal sintética
│   ├── narration/              # Prompts y generación de texto con LLM
│   ├── evaluation/             # Métricas de utilidad, realismo, plausibilidad
│   └── orchestration/          # Grafo LangGraph y control de agentes
