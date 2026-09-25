"""
Evaluación determinista de tratamiento a nivel de cohorte.

Reutiliza el motor de reglas clínicas existente
(`src/validation/clinical_rules.py`), que evalúa la idoneidad/seguridad
farmacológica de pacientes COVID-19 positivos a partir del fármaco prescrito y
de PCR/SatO2/Temperatura. No inventa recomendaciones: si la cohorte no es
COVID-19 o carece de columna de fármaco, lo indica explícitamente.

Opción A del plan (decisión de producto): solo se cubre la lógica COVID
existente. Una tabla determinista enfermedad->tratamiento para otras patologías
(opción B) se puede añadir después en `src/config/treatment_rules.yaml` y
engancharse aquí sin cambiar el resto del pipeline.
"""
from typing import Any, Dict, List, Optional

from src.utils.logging_config import get_logger
import src.validation.clinical_rules as clinical_rules

logger = get_logger(__name__)

DIAG_COL = "DIAG ING/INPAT"
DRUG_COL = "FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME"
COVID_POSITIVE = "COVID19 - POSITIVO"

# Etiquetas legibles por grupo farmacológico.
GROUP_LABELS: Dict[str, str] = {
    "analgesicos_antiinflamatorios": "Analgésicos / Antiinflamatorios",
    "opioides_potentes": "Opioides potentes",
    "psicofarmacos": "Psicofármacos",
    "farmacos_cardiovasculares": "Fármacos cardiovasculares",
    "farmacos_respiratorios": "Fármacos respiratorios",
    "antivirales": "Antivirales",
    "antifungicos": "Antifúngicos",
    "antibioticos": "Antibióticos",
    "corticosteroides": "Corticosteroides",
    "farmacos_digestivos": "Fármacos digestivos",
    "anestesicos_locales": "Anestésicos locales",
    "agentes_vasoactivos": "Agentes vasoactivos",
    "inmunomoduladores": "Inmunomoduladores",
    "hormonas_y_metabolismo": "Hormonas y metabolismo",
    "antihistaminicos": "Antihistamínicos",
    "antisepticos_desinfectantes": "Antisépticos / Desinfectantes",
    "soluciones_suplementos_y_productos_sanitarios": "Soluciones / Suplementos",
    "otros_tratamientos": "Otros tratamientos",
}


def _classify_drug_group(drug_name: str) -> Optional[str]:
    """Clasifica un nombre comercial de fármaco en su grupo terapéutico."""
    if not isinstance(drug_name, str) or not drug_name.strip():
        return None
    drug_up = drug_name.upper()
    for group in clinical_rules.EVALUADORES_MEDICAMENTOS:
        med_list = getattr(clinical_rules, group, [])
        if any(med in drug_up for med in med_list):
            return group
    return None


def assess_cohort_treatment(df, max_rows: int = 1000) -> Dict[str, Any]:
    """Evalúa el tratamiento de una cohorte usando las reglas clínicas COVID.

    Devuelve un dict con:
      - applicable: bool
      - reason: str (por qué no aplica, si procede)
      - total_evaluated, covid_positive
      - alerts_by_group: {grupo_legible: {"count": int, "example": str}}
    """
    result: Dict[str, Any] = {
        "applicable": False,
        "reason": "",
        "total_evaluated": 0,
        "covid_positive": 0,
        "alerts_by_group": {},
    }

    if df is None or getattr(df, "empty", True):
        result["reason"] = "No hay datos de cohorte para evaluar el tratamiento."
        return result

    if DIAG_COL not in df.columns or DRUG_COL not in df.columns:
        result["reason"] = (
            "La cohorte no contiene columnas de diagnóstico y fármaco de tipo COVID-19 "
            f"('{DIAG_COL}', '{DRUG_COL}'). No hay recomendación de tratamiento determinista "
            "disponible para este tipo de dataset (opción A del plan)."
        )
        return result

    sample = df.head(max_rows)
    alerts_by_group: Dict[str, Dict[str, Any]] = {}
    covid_positive = 0

    for _, row in sample.iterrows():
        patient = row.to_dict()
        diagnosis = str(patient.get(DIAG_COL, ""))
        is_covid = diagnosis == COVID_POSITIVE
        if is_covid:
            covid_positive += 1

        warnings = clinical_rules.validate_patient_case(patient)
        if not warnings:
            continue
        for warning in warnings:
            if warning.startswith("No se han encontrado"):
                continue
            group = _classify_drug_group(patient.get(DRUG_COL, "")) or "otros_tratamientos"
            label = GROUP_LABELS.get(group, group)
            entry = alerts_by_group.setdefault(label, {"count": 0, "example": warning})
            entry["count"] += 1

    result["total_evaluated"] = int(len(sample))
    result["covid_positive"] = int(covid_positive)
    result["alerts_by_group"] = alerts_by_group

    if covid_positive == 0:
        result["reason"] = (
            "La cohorte no contiene pacientes 'COVID19 - POSITIVO'; el motor de reglas "
            "deterministas (opción A) no genera recomendaciones de tratamiento."
        )
        result["applicable"] = False
    else:
        result["applicable"] = True

    return result


def build_treatment_markdown(assessment: Dict[str, Any]) -> str:
    """Construye la sección Markdown de tratamiento para el informe."""
    lines: List[str] = ["## 💊 Idoneidad y seguridad del tratamiento (reglas deterministas)"]

    if not assessment.get("applicable"):
        lines.append("")
        lines.append(f"> {assessment.get('reason', 'No aplicable.')}")
        return "\n".join(lines)

    total = assessment.get("total_evaluated", 0)
    covid = assessment.get("covid_positive", 0)
    lines.append("")
    lines.append(
        f"Se evaluaron **{total}** pacientes de la cohorte; **{covid}** son COVID-19 positivos. "
        "Las alertas siguientes provienen de reglas clínicas deterministas sobre la compatibilidad "
        "diagnóstico–tratamiento (fuente: `src/validation/clinical_rules.py`)."
    )
    lines.append("")
    alerts = assessment.get("alerts_by_group", {})
    if not alerts:
        lines.append("- Sin alertas de idoneidad terapéutica para la cohorte evaluada.")
    else:
        lines.append("| Grupo terapéutico | Nº alertas |")
        lines.append("| --- | --- |")
        for label, data in sorted(alerts.items(), key=lambda x: -x[1]["count"]):
            lines.append(f"| {label} | {data['count']} |")
        lines.append("")
        lines.append("### Ejemplos de alerta por grupo")
        for label, data in sorted(alerts.items(), key=lambda x: -x[1]["count"]):
            example = (data.get("example") or "").strip()
            lines.append(f"- **{label}:** {example}")
    return "\n".join(lines)
