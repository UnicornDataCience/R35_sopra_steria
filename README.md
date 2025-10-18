# PatientIA: Sistema de Generación de Datos Médicos Sintéticos con Agentes IA

PatientIA es un sistema avanzado que utiliza un flujo de trabajo basado en agentes de IA para analizar, generar, validar y simular datos de pacientes, con el objetivo de aumentar datasets médicos para investigación y entrenamiento de modelos.

## 📋 Arquitectura y Flujo de Trabajo

El núcleo del sistema es un grafo de agentes orquestado por LangGraph. Cada agente es un especialista en una tarea concreta, permitiendo un flujo de trabajo modular y robusto.

### Diagrama del Flujo de Agentes

```
                    ┌─────────────┐
                    │ Coordinator │
                    │  (Central)  │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
    ┌─────────┐      ┌─────────────┐    ┌─────────────┐
    │Analyzer │◄────►│  Generator  │◄──►│  Validator  │
    │(Análisis│      │(Generación) │    │(Validación) │
    │   EDA)  │      └─────────────┘    └─────────────┘
    └─────────┘             │                   │
         │                  ▼                   ▼
         │            ┌─────────────┐    ┌─────────────┐
         └───────────►│  Evaluator  │◄───│  Simulator  │
                      │(Evaluación) │    │(Simulación) │
                      └─────────────┘    └─────────────┘

    Flujo Principal: Analyzer → Generator → Validator → Evaluator
    
    Características:
    • Coordinator: Punto de entrada y coordinación central
    • Analyzer: Análisis exploratorio de datos (EDA)
    • Generator: Generación de datos sintéticos (CTGAN/TVAE/SDV)
    • Validator: Validación de reglas clínicas
    • Evaluator: Evaluación de calidad
    • Simulator: Simulación temporal de condiciones
```

### Descripción de los Agentes
- **Coordinator:** Punto de entrada. Dirige la tarea inicial al agente apropiado.
- **Analyzer:** Realiza un análisis exploratorio de los datos médicos.
- **Generator:** Genera datos sintéticos utilizando modelos como CTGAN, TVAE o SDV.
- **Validator:** Aplica reglas clínicas y esquemas para asegurar la coherencia de los datos.
- **Evaluator:** Mide la calidad y realismo de los datos generados.
- **Simulator:** Simula la evolución temporal de las condiciones del paciente.

---

## 🚀 Guía de Uso y Comandos

Este proyecto utiliza `uv` para la gestión de entornos virtuales y dependencias.

### Comandos de `uv`

- **Crear el entorno virtual (si no existe):**
  ```shell
  uv venv
  ```

- **Activar el entorno:**
  - En Windows (CMD): `\.venv\Scripts\activate`
  - En Windows (PowerShell): `\.venv\Scripts\Activate.ps1`
  - En Linux/macOS: `source .venv/bin/activate`

- **Instalar dependencias:**
  ```shell
  uv pip install -r requirements-api.txt
  uv pip install -r requirements-client.txt
  ```

- **Sincronizar dependencias (instala/desinstala para que coincida con el `requirements.txt`):**
  ```shell
  uv pip sync requirements-api.txt
  ```

- **Añadir una nueva dependencia:**
  ```shell
  uv pip install <nombre_paquete>
  ```

- **Generar `requirements.txt`:**
  ```shell
  uv pip freeze > requirements.txt
  ```

### Lanzar la Aplicación

La aplicación consta de dos componentes principales: la API (backend) y el cliente de Streamlit (frontend).

- **Paso 1: Ejecutar la API (backend)**
  En una terminal, ejecuta el siguiente comando para iniciar el servidor de la API:
  ```shell
  python run_api.py
  ```
  La API estará disponible en `http://127.0.0.1:8000`. Puedes explorar la documentación interactiva de la API en `http://127.0.0.1:8000/docs`.

- **Paso 2: Ejecutar el Cliente de Streamlit (frontend)**
  En otra terminal, ejecuta el siguiente comando para iniciar la interfaz de usuario:
  ```shell
  python launch_client.py
  ```
  Esto abrirá una nueva pestaña en tu navegador con la aplicación de Streamlit.

---

## 📚 API Endpoints

La API de PatientIA proporciona los siguientes endpoints para interactuar con el sistema:

### Health

- **`GET /api/v1/health`**: Verifica el estado de la API.
- **`GET /api/v1/health/llm`**: Verifica el estado del LLM.

### Chat

- **`POST /api/v1/chat`**: Maneja las interacciones de chat con el sistema.

### Datasets

- **`POST /api/v1/datasets/upload`**: Sube un nuevo dataset.
- **`GET /api/v1/datasets`**: Lista los datasets disponibles.
- **`GET /api/v1/datasets/{dataset_id}`**: Obtiene un dataset específico.
- **`DELETE /api/v1/datasets/{dataset_id}`**: Elimina un dataset específico.

### Analysis

- **`POST /api/v1/analysis`**: Realiza un análisis sobre un dataset.

### Generation

- **`POST /api/v1/generation`**: Genera datos sintéticos.

### Validation

- **`POST /api/v1/validation`**: Valida un dataset.

### Evaluation

- **`POST /api/v1/evaluation`**: Evalúa la calidad de un dataset.

### Simulation

- **`POST /api/v1/simulation`**: Simula datos.

---

## ️ Estado del Sistema y Bitácora

### Estado Actual del Sistema

- **API:** Operativa. Expone los endpoints para interactuar con el flujo de agentes.
- **Flujo de Agentes:** Implementado con LangGraph. El grafo principal es funcional.
- **Modelos de Generación:** Integrados (CTGAN, SDV, TVAE).
- **Validación:** Módulos de validación por reglas clínicas y esquema JSON implementados.
- **Cliente:** Cliente de prueba basado en Streamlit disponible para demostraciones.

### Bitácora de Desarrollo

- **Fase 1: Diseño y prototipado.**
  - Se diseñó la arquitectura de agentes.
  - Se crearon los agentes base y se definió el `WorkflowState`.
  - Se implementó un orquestador simple.
- **Fase 2: Implementación del grafo.**
  - Migración a LangGraph para una orquestación más robusta.
  - Se implementaron las transiciones condicionales entre agentes.
  - Se crearon los routers de la API para cada funcionalidad principal.
- **Fase 3: Pruebas y Refinamiento.**
  - Se realizaron pruebas de integración entre los agentes.
  - Se depuraron los modelos de generación de datos.
  - Se creó el cliente de Streamlit para facilitar la visualización y prueba del flujo completo.
- **Fase 4: Limpieza y Consolidación (Actual).**
  - Unificación de la documentación.
  - Planificación de la limpieza de código obsoleto.
  - Definición de la arquitectura final y preparación para la entrega al equipo de frontend.


