#!/usr/bin/env python3
"""
Script para inicializar y ejecutar la API de Patient-IA
"""
import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv 

load_dotenv()

def main():
    """Función principal"""
    print("Iniciando Patient-IA API Server...")
    
    # Verificar que estamos en el directorio correcto
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Verificar variables de entorno
    check_environment()
    
    # Crear directorios necesarios
    create_directories()
    
    # Ejecutar la API
    run_api()

def check_environment():
    """Verificar variables de entorno necesarias"""
    print("Verificando configuración...")
    
    required_vars = ["GROQ_API_KEY", "LLM_PROVIDER"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Variables de entorno faltantes: {missing_vars}")
        print("Asegúrate de configurar tu archivo .env")
        sys.exit(1)
    
    print("Configuración verificada")

def create_directories():
    """Crear directorios necesarios"""
    directories = [
        "temp_uploads",
        "temp_generations",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("Directorios creados")

def run_api():
    """Ejecutar la API con uvicorn"""
    print("Iniciando servidor API...")
    
    # Configuración del servidor
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    try:
        import uvicorn
    except ImportError:
        print("uvicorn no está instalado")
        print("Instalar con: pip install uvicorn[standard]")
        sys.exit(1)
        
    try:
        # Importa la app desde src/api/main.py
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServidor detenido por usuario")
    except Exception as e:
        print(f"Error ejecutando servidor: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
