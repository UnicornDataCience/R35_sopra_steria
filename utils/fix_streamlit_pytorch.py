# fix_streamlit_pytorch.py

import sys
import streamlit as st
import importlib.util

def patch_streamlit():
    """Aplica parche para evitar problemas entre PyTorch y Streamlit"""
    try:
        # Verificar si torch está instalado
        if importlib.util.find_spec("torch") is not None:
            # Modificar el comportamiento de local_sources_watcher.py
            from streamlit.watcher import local_sources_watcher
            
            # Guardar la función original
            original_extract_paths = local_sources_watcher.extract_paths
            
            # Crear una versión segura de extract_paths
            def safe_extract_paths(module):
                try:
                    return original_extract_paths(module)
                except RuntimeError as e:
                    if "torch.classes" in str(e) or "no running event loop" in str(e):
                        # Ignorar errores específicos de torch.classes
                        return []
                    raise e
                
            # Reemplazar con la versión segura
            local_sources_watcher.extract_paths = safe_extract_paths
            print("✅ Parche PyTorch-Streamlit aplicado correctamente")
    except Exception as e:
        print(f"ℹ️ No se pudo aplicar el parche: {e}")

# Aplicar parche al importar este módulo
patch_streamlit()