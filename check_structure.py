import os

def check_project_structure():
    """Verifica que la estructura del proyecto sea correcta"""
    
    required_files = [
        "src/__init__.py",
        "src/agents/__init__.py",
        "src/agents/base_agent.py",
        "src/agents/coordinator_agent.py",
        "src/agents/analyzer_agent.py",
        "src/agents/generator_agent.py",
        "src/agents/validator_agent.py",
        "src/agents/simulator_agent.py",
        "src/agents/evaluator_agent.py",
        "src/orchestration/__init__.py",
        "src/orchestration/langgraph_orchestrator.py",
        "interfaces/chat_llm.py",
        ".env"
    ]
    
    print("🔍 Verificando estructura del proyecto...\n")
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - FALTA")
            all_good = False
    
    print(f"\n{'✅ Estructura correcta!' if all_good else '❌ Faltan archivos importantes'}")
    return all_good

if __name__ == "__main__":
    check_project_structure()
