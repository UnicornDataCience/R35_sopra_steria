from langgraph.graph import StateGraph

def build_graph():
    '''
    La función build_graph construye un grafo de ejecución con
    nodos y aristas enlazadaos secuencialmente, que contienen
    una serie de estados que representa el flujo de trabajo
    '''
    graph = StateGraph()
    graph.add_node("extract", lambda state: state)
    graph.add_node("generate", lambda state: state)
    graph.add_node("validate", lambda state: state)
    graph.add_node("simulate", lambda state: state)
    graph.add_node("narrate", lambda state: state)
    graph.add_node("extract", lambda state: state)
    graph.add_edge("extract", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "simulate")
    graph.add_edge("simulate", "narrate")
    graph.set_entry_point("extract")
    return graph.compile()

        