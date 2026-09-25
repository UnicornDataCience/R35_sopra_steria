"""
Contrato único de transiciones entre agentes.

Fuente de verdad para "qué agente puede invocar a cuál". Se usa tanto para:
  1. Validar en tiempo de ejecución la decisión de enrutamiento del coordinador
     (modo chat, 1 salto), evitando destinos no permitidos.
  2. Definir la secuencia lineal determinista del pipeline de "historial de
     cohorte" (analyzer -> generator -> validator -> evaluator -> simulator).

El grafo declarado aquí es acíclico por construcción; `assert_acyclic()` lo
verifica en tiempo de importación para que cualquier cambio que introduzca un
ciclo falle de forma temprana y explícita.
"""
from typing import Dict, List, Set

# Nodo terminal del grafo (equivalente a langgraph.END en el enrutamiento).
END = "__end__"

# Agentes válidos del sistema.
AGENTS: Set[str] = {
    "coordinator",
    "analyzer",
    "generator",
    "validator",
    "evaluator",
    "simulator",
}

# --- Modo chat (1 salto): el coordinador delega en un único subagente ---------
# El coordinador puede delegar en cualquier subagente o terminar. Los subagentes
# terminan siempre (no invocan a otros agentes en modo conversacional).
CHAT_TRANSITIONS: Dict[str, Set[str]] = {
    "coordinator": {"analyzer", "generator", "validator", "evaluator", "simulator", END},
    "analyzer": {END},
    "generator": {END},
    "validator": {END},
    "evaluator": {END},
    "simulator": {END},
}

# --- Pipeline determinista de "historial de cohorte" --------------------------
# Secuencia lineal fija. Es intrínsecamente acíclica y no admite saltos hacia
# atrás, por lo que no puede generar bucles ni dejar el flujo en latencia.
PIPELINE_SEQUENCE: List[str] = [
    "analyzer",
    "generator",
    "validator",
    "evaluator",
    "simulator",
]

# Presupuesto de pasos del pipeline (guarda defensiva adicional al orden fijo).
PIPELINE_STEP_BUDGET: int = len(PIPELINE_SEQUENCE) + 2

# Límite de recursión que se pasa a LangGraph al invocar cualquier workflow.
RECURSION_LIMIT: int = 12


def is_allowed(source: str, target: str) -> bool:
    """Devuelve True si `source` puede transicionar a `target` en modo chat."""
    return target in CHAT_TRANSITIONS.get(source, set())


def allowed_targets(source: str) -> Set[str]:
    """Destinos permitidos para `source` en modo chat (incluye END)."""
    return set(CHAT_TRANSITIONS.get(source, set()))


def _pipeline_transitions() -> Dict[str, Set[str]]:
    """Adyacencia derivada de la secuencia lineal del pipeline."""
    transitions: Dict[str, Set[str]] = {}
    for i, node in enumerate(PIPELINE_SEQUENCE):
        nxt = PIPELINE_SEQUENCE[i + 1] if i + 1 < len(PIPELINE_SEQUENCE) else END
        transitions.setdefault(node, set()).add(nxt)
    return transitions


def assert_acyclic() -> None:
    """Verifica que el grafo combinado (chat + pipeline) sea acíclico.

    Lanza ValueError si detecta un ciclo. END no tiene salidas.
    """
    graph: Dict[str, Set[str]] = {}
    for src, dsts in CHAT_TRANSITIONS.items():
        graph.setdefault(src, set()).update(d for d in dsts if d != END)
    for src, dsts in _pipeline_transitions().items():
        graph.setdefault(src, set()).update(d for d in dsts if d != END)

    WHITE, GREY, BLACK = 0, 1, 2
    color: Dict[str, int] = {node: WHITE for node in graph}

    def visit(node: str, stack: List[str]) -> None:
        color[node] = GREY
        for nxt in graph.get(node, set()):
            if nxt not in color:
                color[nxt] = WHITE
            if color[nxt] == GREY:
                cycle = " -> ".join(stack + [node, nxt])
                raise ValueError(f"Ciclo detectado en el contrato de agentes: {cycle}")
            if color[nxt] == WHITE:
                visit(nxt, stack + [node])
        color[node] = BLACK

    for node in list(graph.keys()):
        if color.get(node, WHITE) == WHITE:
            visit(node, [])


# Verificación temprana: si alguien introduce un ciclo, el import falla.
assert_acyclic()
