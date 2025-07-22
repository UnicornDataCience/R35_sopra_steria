"""
Test para identificar el problema específico de StateGraph
"""
try:
    from langgraph.graph import StateGraph
    print("✅ StateGraph importado correctamente")
    print(f"StateGraph MRO: {StateGraph.__mro__}")
    
    # Crear un dict simple como estado
    simple_state = dict
    workflow = StateGraph(simple_state)
    print("✅ StateGraph con dict simple funciona")
    
except Exception as e:
    print(f"❌ Error con StateGraph: {e}")
    import traceback
    traceback.print_exc()

try:
    from typing import TypedDict
    
    class TestState(TypedDict):
        test: str
    
    from langgraph.graph import StateGraph
    workflow = StateGraph(TestState)
    print("✅ StateGraph con TypedDict funciona")
    
except Exception as e:
    print(f"❌ Error con StateGraph + TypedDict: {e}")
    import traceback
    traceback.print_exc()
