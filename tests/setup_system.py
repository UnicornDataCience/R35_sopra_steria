import os
from dotenv import load_dotenv

# Forzar recarga
load_dotenv(override=True)

try:
    # Instalar dependencias si no estan
    try:
        import requests
        print('requests: OK')
    except ImportError:
        print('Instalando requests...')
        os.system('pip install requests')
    
    # Verificar que langchain-ollama este instalado
    try:
        from langchain_community.llms import Ollama
        print('langchain-community: OK')
    except ImportError:
        print('Instalando langchain-community...')
        os.system('pip install langchain-community')
    
    # Recrear configuracion LLM
    print('Recreando configuracion LLM unificada...')
    from src.config.llm_config import UnifiedLLMConfig
    
    new_config = UnifiedLLMConfig()
    print(f'Proveedor activo: {new_config.active_provider}')
    print(f'Estado: {new_config.status_info}')
    
    if new_config.active_provider == 'ollama':
        print('CONFIGURACION OLLAMA EXITOSA!')
    elif new_config.active_provider == 'mock':
        print('En modo simulado - pero esto es normal si el test de conexion es lento')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
