import os

def fix_init_files():
    """Corrige todos los archivos __init__.py del proyecto"""
    
    # Contenido correcto para cada __init__.py
    init_contents = {
        "src/__init__.py": '''# Inicialización del paquete src
__version__ = "1.0.0"
''',
        
        "src/agents/__init__.py": '''from .base_agent import BaseLLMAgent, BaseAgentConfig
from .coordinator_agent import CoordinatorAgent
from .analyzer_agent import ClinicalAnalyzerAgent
from .generator_agent import SyntheticGeneratorAgent
from .validator_agent import MedicalValidatorAgent
from .simulator_agent import PatientSimulatorAgent
from .evaluator_agent import UtilityEvaluatorAgent

__all__ = [
    'BaseLLMAgent',
    'BaseAgentConfig',
    'CoordinatorAgent',
    'ClinicalAnalyzerAgent', 
    'SyntheticGeneratorAgent',
    'MedicalValidatorAgent',
    'PatientSimulatorAgent',
    'UtilityEvaluatorAgent'
]
''',
        
        "src/config/__init__.py": '''from .azure_config import AzureOpenAIConfig, azure_config

__all__ = ['AzureOpenAIConfig', 'azure_config']
''',
        
        "src/extraction/__init__.py": '''from .data_extractor import DataExtractor

__all__ = ['DataExtractor']
''',
        
        "src/generation/__init__.py": '''from .sdv_generator import SDVGenerator

__all__ = ['SDVGenerator']
''',
        
        "src/validation/__init__.py": '''# Validation modules placeholder
__all__ = []
''',
        
        "src/simulation/__init__.py": '''# Simulation modules placeholder  
__all__ = []
''',
        
        "src/evaluation/__init__.py": '''# Evaluation modules placeholder
__all__ = []
''',
        
        "src/orchestration/__init__.py": '''# Orchestration modules placeholder
__all__ = []
''',
        
        "src/narration/__init__.py": '''# Narration modules placeholder
__all__ = []
'''
    }
    
    print("🔧 Corrigiendo archivos __init__.py...")
    
    for file_path, content in init_contents.items():
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Escribir contenido correcto
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ {file_path}")
            
        except Exception as e:
            print(f"❌ Error en {file_path}: {e}")
    
    print("\n🎉 Corrección completada!")

if __name__ == "__main__":
    fix_init_files()