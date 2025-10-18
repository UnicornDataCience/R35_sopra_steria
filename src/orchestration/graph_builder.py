from langgraph.graph import StateGraph
from typing import Dict, Any

def build_medical_workflow_graph():
    """
    Construye un grafo flexible para el flujo de trabajo médico
    Permite transiciones inteligentes entre agentes según el contexto
    Compatible con LangGraph Studio
    """
    return create_langgraph_studio_graph()

def build_graph():
    """
    Función legacy - redirige al nuevo grafo médico
    """
    return build_medical_workflow_graph()

def get_agent_transitions() -> Dict[str, Dict[str, str]]:
    """
    Define las transiciones posibles y sus condiciones
    """
    return {
        "coordinator": {
            "analyzer": "Análisis inicial de datos",
            "generator": "Generación directa de datos sintéticos",
            "validator": "Validación directa de datos existentes",
            "evaluator": "Evaluación directa de calidad",
            "simulator": "Simulación directa de evolución"
        },
        "analyzer": {
            "generator": "Proceder a generar datos sintéticos",
            "simulator": "Analizar evolución temporal sin generar datos",
            "coordinator": "Volver al coordinador"
        },
        "generator": {
            "validator": "Validar datos generados (recomendado)",
            "evaluator": "Evaluar calidad directamente",
            "analyzer": "Re-analizar si hay problemas",
            "coordinator": "Volver al coordinador"
        },
        "validator": {
            "evaluator": "Evaluar calidad después de validar",
            "simulator": "Simular evolución con datos validados",
            "generator": "Regenerar si la validación falla",
            "coordinator": "Volver al coordinador"
        },
        "evaluator": {
            "simulator": "Simular evolución con datos evaluados",
            "generator": "Regenerar si la calidad es baja",
            "analyzer": "Re-analizar si se detectan problemas",
            "coordinator": "Volver al coordinador"
        },
        "simulator": {
            "coordinator": "Volver al coordinador",
            "evaluator": "Evaluar realismo de simulaciones"
        }
    }

def visualize_medical_workflow_graph(save_path: str = None, show_plot: bool = True):
    """
    Visualiza el grafo de flujo de trabajo médico usando networkx y matplotlib
    
    Args:
        save_path: Ruta donde guardar la imagen (opcional)
        show_plot: Si mostrar el gráfico en pantalla
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("⚠️  Para visualizar el grafo, instala las dependencias:")
        print("   uv add networkx matplotlib")
        return
    
    # Crear el grafo dirigido
    G = nx.DiGraph()
    
    # Obtener las transiciones
    transitions = get_agent_transitions()
    
    # Agregar nodos con sus descripciones
    node_descriptions = {
        "coordinator": "Coordinador\n(Punto de entrada)",
        "analyzer": "Analizador\n(Análisis médico)",
        "generator": "Generador\n(Datos sintéticos)",
        "validator": "Validador\n(Validación médica)",
        "evaluator": "Evaluador\n(Calidad de datos)",
        "simulator": "Simulador\n(Evolución temporal)"
    }
    
    # Agregar nodos
    for node in node_descriptions.keys():
        G.add_node(node)
    
    # Agregar aristas basadas en las transiciones
    for from_node, destinations in transitions.items():
        for to_node, description in destinations.items():
            G.add_edge(from_node, to_node, label=description)
    
    # Configurar el layout del gráfico
    plt.figure(figsize=(16, 12))
    
    # Usar un layout jerárquico personalizado
    pos = {
        "coordinator": (0, 3),
        "analyzer": (-2, 1),
        "generator": (0, 1),
        "validator": (2, 1),
        "evaluator": (-1, -1),
        "simulator": (1, -1)
    }
    
    # Colores para cada tipo de agente
    node_colors = {
        "coordinator": "#FF6B6B",  # Rojo suave
        "analyzer": "#4ECDC4",     # Turquesa
        "generator": "#45B7D1",    # Azul
        "validator": "#96CEB4",    # Verde suave
        "evaluator": "#FFEAA7",    # Amarillo suave
        "simulator": "#DDA0DD"     # Púrpura suave
    }
    
    # Dibujar nodos
    for node, (x, y) in pos.items():
        # Crear un rectángulo redondeado para cada nodo
        bbox = FancyBboxPatch(
            (x-0.6, y-0.3), 1.2, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=node_colors[node],
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        plt.gca().add_patch(bbox)
        
        # Agregar texto del nodo
        plt.text(x, y, node_descriptions[node], 
                ha='center', va='center', 
                fontsize=10, fontweight='bold',
                wrap=True)
    
    # Dibujar aristas
    for edge in G.edges():
        start_pos = pos[edge[0]]
        end_pos = pos[edge[1]]
        
        # Calcular offset para evitar solapamiento
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Ajustar puntos de inicio y fin para que no se solapen con los nodos
        if dx != 0 or dy != 0:
            length = (dx**2 + dy**2)**0.5
            unit_dx = dx / length
            unit_dy = dy / length
            
            # Offset desde el borde del nodo
            offset = 0.4
            start_x = start_pos[0] + unit_dx * offset
            start_y = start_pos[1] + unit_dy * offset
            end_x = end_pos[0] - unit_dx * offset
            end_y = end_pos[1] - unit_dy * offset
            
            # Dibujar flecha
            plt.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle='->', 
                                      connectionstyle="arc3,rad=0.1",
                                      color='gray', 
                                      alpha=0.7,
                                      linewidth=1.5))
    
    # Configurar el gráfico
    plt.title('🩺 Patientia - Flujo de Trabajo de Agentes Médicos', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Agregar leyenda
    legend_text = """
    Flujo Principal:
    Coordinator → Analyzer → Generator → Validator → Evaluator
    
    Flujos Opcionales:
    • Validator ⟷ Simulator (Simulación con datos validados)
    • Evaluator ⟷ Simulator (Análisis temporal)
    • Loops de refinamiento (regeneración, re-análisis)
    
    Flujos Directos:
    • Coordinator puede ir directamente a cualquier agente
    • Generator puede ir directo a Evaluator
    """
    
    plt.figtext(0.02, 0.02, legend_text, fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Ajustar límites y aspecto
    plt.xlim(-3.5, 3.5)
    plt.ylim(-2.5, 4)
    plt.axis('off')
    plt.tight_layout()
    
    # Guardar si se especifica ruta
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Grafo guardado en: {save_path}")
    
    # Mostrar si se requiere
    if show_plot:
        plt.show()
    
    return G

def print_workflow_summary():
    """
    Imprime un resumen textual del flujo de trabajo
    """
    print("🩺" + "="*60 + "🩺")
    print("           PATIENTIA - FLUJO DE AGENTES MÉDICOS")
    print("🩺" + "="*60 + "🩺")
    
    transitions = get_agent_transitions()
    
    for agent, destinations in transitions.items():
        print(f"\n🤖 {agent.upper()}:")
        for dest, description in destinations.items():
            print(f"   └─ {dest}: {description}")
    
    print("\n" + "="*65)
    print("📊 ESTADÍSTICAS:")
    print(f"   • Total de agentes: {len(transitions)}")
    print(f"   • Total de transiciones: {sum(len(dest) for dest in transitions.values())}")
    print(f"   • Punto de entrada: coordinator")
    print("="*65)

def create_langgraph_studio_graph():
    """
    Crea el grafo optimizado para LangGraph Studio
    """
    from langgraph.graph import StateGraph
    from typing import TypedDict
    
    # Definir el estado del workflow
    class WorkflowState(TypedDict):
        messages: list
        current_data: dict
        analysis_results: dict
        generated_data: dict
        validation_status: str
        quality_score: float
        simulation_results: dict
        next_action: str
    
    # Crear el grafo
    workflow = StateGraph(WorkflowState)
    
    # Funciones de los agentes (simplificadas para el estudio)
    def coordinator_agent(state: WorkflowState):
        """Agente coordinador - punto de entrada"""
        return {
            **state,
            "messages": state.get("messages", []) + ["Coordinador activado"],
            "next_action": "analyze"
        }
    
    def analyzer_agent(state: WorkflowState):
        """Agente analizador - análisis de datos médicos"""
        return {
            **state,
            "messages": state.get("messages", []) + ["Análisis médico completado"],
            "analysis_results": {"status": "analyzed", "patterns": "detected"},
            "next_action": "generate"
        }
    
    def generator_agent(state: WorkflowState):
        """Agente generador - creación de datos sintéticos"""
        return {
            **state,
            "messages": state.get("messages", []) + ["Datos sintéticos generados"],
            "generated_data": {"synthetic_records": 1000, "method": "CTGAN"},
            "next_action": "validate"
        }
    
    def validator_agent(state: WorkflowState):
        """Agente validador - validación médica"""
        return {
            **state,
            "messages": state.get("messages", []) + ["Validación médica completada"],
            "validation_status": "passed",
            "next_action": "evaluate"
        }
    
    def evaluator_agent(state: WorkflowState):
        """Agente evaluador - evaluación de calidad"""
        return {
            **state,
            "messages": state.get("messages", []) + ["Evaluación de calidad completada"],
            "quality_score": 0.95,
            "next_action": "simulate"
        }
    
    def simulator_agent(state: WorkflowState):
        """Agente simulador - evolución temporal"""
        return {
            **state,
            "messages": state.get("messages", []) + ["Simulación temporal completada"],
            "simulation_results": {"temporal_patterns": "stable", "evolution": "realistic"},
            "next_action": "end"
        }
    
    # === NODOS DEL GRAFO ===
    workflow.add_node("coordinator", coordinator_agent)
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("generator", generator_agent)
    workflow.add_node("validator", validator_agent)
    workflow.add_node("evaluator", evaluator_agent)
    workflow.add_node("simulator", simulator_agent)
    
    # === FLUJO PRINCIPAL SECUENCIAL ===
    workflow.add_edge("coordinator", "analyzer")
    workflow.add_edge("analyzer", "generator")
    workflow.add_edge("generator", "validator")
    workflow.add_edge("validator", "evaluator")
    workflow.add_edge("evaluator", "simulator")
    
    # === FLUJOS CONDICIONALES Y LOOPS ===
    # Desde validator, puede ir a simulator o regenerar
    workflow.add_conditional_edges(
        "validator",
        lambda state: "regenerate" if state.get("validation_status") == "failed" else "evaluate",
        {
            "evaluate": "evaluator",
            "regenerate": "generator",
            "simulate": "simulator"
        }
    )
    
    # Desde evaluator, puede continuar o mejorar
    workflow.add_conditional_edges(
        "evaluator",
        lambda state: "improve" if state.get("quality_score", 0) < 0.8 else "simulate",
        {
            "simulate": "simulator",
            "improve": "generator",
            "reanalyze": "analyzer"
        }
    )
    
    # === FLUJOS DIRECTOS DESDE COORDINATOR ===
    workflow.add_conditional_edges(
        "coordinator",
        lambda state: state.get("next_action", "analyze"),
        {
            "analyze": "analyzer",
            "generate": "generator",
            "validate": "validator",
            "evaluate": "evaluator",
            "simulate": "simulator"
        }
    )
    
    # Punto de entrada
    workflow.set_entry_point("coordinator")
    
    return workflow.compile()

def setup_langgraph_studio():
    """
    Configuración para LangGraph Studio
    Imprime las instrucciones para usar con LangGraph Studio
    """
    print("🎯" + "="*60 + "🎯")
    print("           CONFIGURACIÓN PARA LANGGRAPH STUDIO")
    print("🎯" + "="*60 + "🎯")
    
    print("\n� PASOS PARA USAR CON LANGGRAPH STUDIO:")
    print("1. Instalar LangGraph Studio:")
    print("   pip install langgraph-studio")
    print("\n2. Crear archivo de configuración (langgraph.json):")
    print("   {")
    print('     "dependencies": ["./src"],')
    print('     "graphs": {')
    print('       "medical_workflow": "./src/orchestration/graph_builder.py:create_langgraph_studio_graph"')
    print('     }')
    print("   }")
    print("\n3. Ejecutar LangGraph Studio:")
    print("   langgraph up")
    print("\n4. Abrir en el navegador:")
    print("   http://localhost:8123")
    
    print("\n🔧 CONFIGURACIÓN ALTERNATIVA - Ejecutar directamente:")
    print("   from src.orchestration.graph_builder import create_langgraph_studio_graph")
    print("   graph = create_langgraph_studio_graph()")
    print("   # Usar graph.get_graph().draw_mermaid() para ver el diagrama")
    
    print("\n" + "="*65)

if __name__ == "__main__":
    # Configuración para LangGraph Studio
    print("🎯 Configurando para LangGraph Studio...")
    setup_langgraph_studio()
    
    # Crear el grafo para LangGraph Studio
    print("\n🏗️ Creando grafo para LangGraph Studio...")
    try:
        graph = create_langgraph_studio_graph()
        print("✅ Grafo creado exitosamente!")
        
        # Intentar mostrar el diagrama Mermaid si está disponible
        try:
            mermaid_diagram = graph.get_graph().draw_mermaid()
            print("\n📊 Diagrama Mermaid generado:")
            print(mermaid_diagram)
        except Exception as e:
            print(f"ℹ️  Para ver el diagrama completo, usar LangGraph Studio: {e}")
            
    except Exception as e:
        print(f"❌ Error creando el grafo: {e}")
        print("💡 Asegúrate de tener langgraph instalado: pip install langgraph")
    
    # También imprimir el resumen textual
    print("\n📋 Resumen del flujo de trabajo:")
    print_workflow_summary()

