import os
import re

def fix_tool_classes():
    """Corrige las clases Tool añadiendo anotaciones de tipo"""
    
    tool_files = [
        "src/agents/analyzer_agent.py",
        "src/agents/generator_agent.py", 
        "src/agents/validator_agent.py",
        "src/agents/simulator_agent.py",
        "src/agents/evaluator_agent.py"
    ]
    
    print("🔧 Corrigiendo clases Tool...")
    
    for file_path in tool_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Reemplazar definiciones de name y description sin tipo
                content = re.sub(
                    r'(\s+)name = "([^"]+)"',
                    r'\1name: str = "\2"',
                    content
                )
                
                content = re.sub(
                    r'(\s+)description = "([^"]+)"',
                    r'\1description: str = "\2"',
                    content
                )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ {file_path}")
                
            except Exception as e:
                print(f"❌ Error en {file_path}: {e}")
        else:
            print(f"⚠️ Archivo no encontrado: {file_path}")
    
    print("\n🎉 Corrección de Tools completada!")

if __name__ == "__main__":
    fix_tool_classes()