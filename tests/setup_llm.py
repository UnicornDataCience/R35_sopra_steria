#!/usr/bin/env python3
"""
Script de configuración rápida para proveedores LLM
Permite cambiar fácilmente entre Azure OpenAI, Ollama y Grok
"""

import os
import sys
from pathlib import Path

# Añadir la ruta del proyecto
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def update_env_provider(provider: str):
    """Actualiza la variable LLM_PROVIDER en .env"""
    env_path = project_root / ".env"
    
    if not env_path.exists():
        print("❌ Archivo .env no encontrado")
        return False
    
    # Leer archivo actual
    lines = []
    with open(env_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Actualizar línea de proveedor
    updated = False
    for i, line in enumerate(lines):
        if line.startswith('LLM_PROVIDER='):
            lines[i] = f'LLM_PROVIDER={provider}\n'
            updated = True
            break
    
    if not updated:
        lines.append(f'LLM_PROVIDER={provider}\n')
    
    # Escribir archivo actualizado
    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✅ Proveedor LLM actualizado a: {provider}")
    return True

def test_providers():
    """Prueba todos los proveedores disponibles"""
    try:
        from src.config.llm_config import unified_llm_config
        
        print("🔍 Probando proveedores LLM disponibles...\n")
        
        for name, provider in unified_llm_config.providers.items():
            print(f"📡 {name.upper()}:")
            print(f"   Disponible: {'✅' if provider.available else '❌'}")
            
            if provider.available:
                test_result = provider.test_connection()
                print(f"   Conexión: {'✅' if test_result else '❌'}")
                if hasattr(provider, 'model'):
                    print(f"   Modelo: {provider.model}")
            print()
        
        print(f"🎯 Proveedor activo: {unified_llm_config.active_provider}")
        return True
        
    except Exception as e:
        print(f"❌ Error probando proveedores: {e}")
        return False

def setup_ollama():
    """Guía de configuración para Ollama"""
    print("🦙 CONFIGURACIÓN DE OLLAMA")
    print("=" * 40)
    print("1. Instalar Ollama:")
    print("   - Descarga desde: https://ollama.ai/")
    print("   - O usa: curl -fsSL https://ollama.ai/install.sh | sh")
    print()
    print("2. Instalar modelo:")
    print("   ollama pull llama3.1")
    print("   # o")
    print("   ollama pull llama2")
    print()
    print("3. Verificar que esté ejecutándose:")
    print("   ollama list")
    print()
    print("4. Si necesitas cambiar el puerto:")
    print("   Edita OLLAMA_BASE_URL en .env")
    
def setup_grok():
    """Guía de configuración para Grok"""
    print("🚀 CONFIGURACIÓN DE GROK (X.AI)")
    print("=" * 40)
    print("1. Obtener API Key:")
    print("   - Ve a: https://console.x.ai/")
    print("   - Crea una cuenta y genera una API key")
    print()
    print("2. Configurar en .env:")
    print("   GROK_API_KEY=tu_api_key_aqui")
    print()
    print("3. Verificar configuración:")
    print("   python setup_llm.py test")

def main():
    if len(sys.argv) < 2:
        print("🔧 CONFIGURADOR DE PROVEEDORES LLM")
        print("=" * 40)
        print("Uso:")
        print("  python setup_llm.py <comando>")
        print()
        print("Comandos:")
        print("  azure    - Cambiar a Azure OpenAI")
        print("  ollama   - Cambiar a Ollama local")
        print("  grok     - Cambiar a Grok (X.AI)")
        print("  test     - Probar todos los proveedores")
        print("  setup-ollama - Guía de configuración Ollama")
        print("  setup-grok   - Guía de configuración Grok")
        return
    
    command = sys.argv[1].lower()
    
    if command in ["azure", "ollama", "grok"]:
        if update_env_provider(command):
            print("🔄 Reinicia la aplicación para aplicar los cambios")
            
    elif command == "test":
        test_providers()
        
    elif command == "setup-ollama":
        setup_ollama()
        
    elif command == "setup-grok":
        setup_grok()
        
    else:
        print(f"❌ Comando desconocido: {command}")

if __name__ == "__main__":
    main()
