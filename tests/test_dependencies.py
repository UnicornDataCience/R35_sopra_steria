#!/usr/bin/env python3
"""
Script para verificar que todas las dependencias estén instaladas correctamente
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Prueba importar un módulo"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        package = package_name or module_name
        print(f"❌ {module_name} - Instalar: pip install {package}")
        return False

def main():
    print("🔍 Verificando dependencias para Patientia AI...")
    print("=" * 50)
    
    # Lista de dependencias críticas
    dependencies = [
        ("langchain", "langchain"),
        ("langchain_openai", "langchain-openai"),
        ("langchain.agents", "langchain"),
        ("langchain.tools", "langchain"),
        ("langchain.prompts", "langchain"),
        ("langchain.memory", "langchain"),
        ("langchain.schema", "langchain"),
        ("openai", "openai[azure]"),
        ("pydantic", "pydantic"),
        ("tiktoken", "tiktoken"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("streamlit", "streamlit"),
        ("dotenv", "python-dotenv"),
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for module, package in dependencies:
        if test_import(module, package):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Resultado: {success_count}/{total_count} dependencias disponibles")
    
    if success_count == total_count:
        print("🎉 ¡Todas las dependencias están instaladas!")
        return True
    else:
        print("⚠️ Faltan dependencias. Ejecuta:")
        print("pip install langchain langchain-openai openai[azure] pydantic")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)