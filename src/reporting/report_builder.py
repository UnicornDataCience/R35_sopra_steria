"""
Ensamblador del informe consolidado de cohorte (HTML).

Toma la salida del pipeline determinista
(`MedicalAgentsOrchestrator.process_clinical_history`) y produce un único
documento HTML con: EDA, generación, validación, métricas de calidad,
evolución temporal (con gráficos embebidos) y tratamiento (reglas
deterministas). Reutiliza `src/utils/image_embedding.py` para embeber imágenes.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.logging_config import get_logger
from src.utils.image_embedding import create_markdown_image
from src.orchestration import treatment as treatment_mod

logger = get_logger(__name__)

# Nombres de agente tal y como aparecen en el estado del pipeline.
AGENT_ANALYZER = "Analizador Clínico"
AGENT_GENERATOR = "Generador Sintético"
AGENT_VALIDATOR = "Validador Médico"
AGENT_EVALUATOR = "Evaluador de Utilidad"
AGENT_SIMULATOR = "Simulador de Pacientes"

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Historial clínico sintético de cohorte - Patient-IA</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          line-height: 1.6; max-width: 1100px; margin: 0 auto; padding: 20px; background: #f5f6fa; color: #2c3e50; }}
  .container {{ background: #fff; padding: 40px; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
  h1 {{ border-bottom: 3px solid #6c5ce7; padding-bottom: 10px; }}
  h2 {{ margin-top: 34px; border-left: 4px solid #6c5ce7; padding-left: 12px; }}
  h3 {{ color: #34495e; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9em; }}
  th, td {{ border: 1px solid #e1e4e8; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f0f6; }}
  img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.12); margin: 16px 0; }}
  code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 3px; }}
  blockquote {{ border-left: 4px solid #f39c12; background: #fff9e6; margin: 12px 0; padding: 8px 14px; }}
  .timestamp {{ text-align: right; color: #7f8c8d; font-size: 0.85em; margin-top: 40px;
                border-top: 1px solid #ecf0f1; padding-top: 12px; }}
</style>
</head>
<body>
  <div class="container">
    {content}
    <div class="timestamp">Generado el {ts} por Patient-IA</div>
  </div>
</body>
</html>"""


def _step_message(steps: Dict[str, Any], agent_name: str) -> str:
    step = steps.get(agent_name) or {}
    return (step.get("message") or "").strip()


def _synthetic_table_html(synthetic_data, max_rows: int = 15) -> str:
    if synthetic_data is None:
        return "_No se generaron datos sintéticos._"
    try:
        preview = synthetic_data.head(max_rows)
        return preview.to_html(index=False, border=0)
    except Exception as e:  # pragma: no cover - defensivo
        logger.warning("No se pudo renderizar la tabla sintética: %s", e)
        return "_No se pudo renderizar la vista previa de la cohorte sintética._"


def build_cohort_report(pipeline_result: Dict[str, Any], dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """Construye el informe HTML consolidado a partir del resultado del pipeline.

    Devuelve {"html": str, "markdown": str, "path": str|None, "error": str|None}.
    """
    if pipeline_result.get("error"):
        return {"html": None, "markdown": None, "path": None, "error": pipeline_result["error"]}

    steps = pipeline_result.get("steps", {}) or {}
    synthetic_data = pipeline_result.get("synthetic_data")

    n_rows = None
    try:
        n_rows = int(len(synthetic_data)) if synthetic_data is not None else None
    except Exception:
        n_rows = None

    # Evaluación de tratamiento sobre la cohorte sintética.
    treatment_md = ""
    try:
        assessment = treatment_mod.assess_cohort_treatment(synthetic_data)
        treatment_md = treatment_mod.build_treatment_markdown(assessment)
    except Exception as e:
        logger.warning("Fallo evaluando tratamiento: %s", e)
        treatment_md = "## 💊 Tratamiento\n\n> No se pudo evaluar el tratamiento de la cohorte."

    # Gráficos de evolución temporal (del simulador).
    sim_step = steps.get(AGENT_SIMULATOR) or {}
    timeline_path = sim_step.get("timeline_path")
    heatmap_path = sim_step.get("heatmap_path")
    viz_md_parts = []
    if timeline_path:
        viz_md_parts.append(create_markdown_image(timeline_path, "Evolución temporal", width=820))
    if heatmap_path:
        viz_md_parts.append(create_markdown_image(heatmap_path, "Mapa de calor de evolución", width=820))
    viz_md = "\n\n".join(viz_md_parts) if viz_md_parts else "_No hay visualizaciones de evolución disponibles._"

    header = f"# 🩺 Historial clínico sintético de cohorte\n\n"
    if dataset_name:
        header += f"**Dataset base:** {dataset_name}  \n"
    if n_rows is not None:
        header += f"**Pacientes sintéticos generados:** {n_rows}  \n"
    header += f"**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"

    md_parts = [
        header,
        "## 📊 Análisis exploratorio (EDA)",
        _step_message(steps, AGENT_ANALYZER) or "_Sin informe de análisis._",
        "## 🎲 Generación de datos sintéticos",
        _step_message(steps, AGENT_GENERATOR) or "_Sin informe de generación._",
        "### Vista previa de la cohorte sintética",
        _synthetic_table_html(synthetic_data),
        "## ✅ Validación médica",
        _step_message(steps, AGENT_VALIDATOR) or "_Sin informe de validación._",
        "## 📈 Evaluación de calidad y utilidad",
        _step_message(steps, AGENT_EVALUATOR) or "_Sin informe de evaluación._",
        "## 🧬 Evolución temporal de la cohorte",
        _step_message(steps, AGENT_SIMULATOR) or "_Sin informe de simulación._",
        viz_md,
        treatment_md,
    ]
    markdown_doc = "\n\n".join(part for part in md_parts if part)

    # Convertir a HTML.
    try:
        import markdown as md_lib
        content_html = md_lib.markdown(markdown_doc, extensions=["extra", "tables", "sane_lists"])
    except Exception as e:
        logger.warning("markdown no disponible o falló (%s); usando <pre>", e)
        content_html = f"<pre>{markdown_doc}</pre>"

    html = _HTML_TEMPLATE.format(content=content_html, ts=datetime.now().strftime("%d/%m/%Y a las %H:%M:%S"))

    # Guardar a disco (best-effort).
    path = None
    try:
        out_dir = Path("temp_generations/reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"cohort_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("📄 Informe de cohorte guardado en %s", path)
    except Exception as e:
        logger.warning("No se pudo guardar el informe en disco: %s", e)
        path = None

    return {"html": html, "markdown": markdown_doc, "path": path, "error": None}
