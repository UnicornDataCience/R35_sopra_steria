"""
Configuración de optimización de rendimiento para Ollama
"""

# Configuración optimizada para MedLlama2
OLLAMA_PERFORMANCE_CONFIG = {
    # Parámetros de generación optimizados para velocidad
    "temperature": 0.3,      # Menos aleatorio = más rápido
    "top_p": 0.8,           # Limitar vocabulario = más rápido  
    "top_k": 40,            # Limitar opciones = más rápido
    "num_predict": 200,     # Máximo tokens de respuesta
    "num_ctx": 2048,        # Contexto moderado para balance velocidad/calidad
    "repeat_penalty": 1.1,   # Evitar repeticiones
    "stream": False,         # Sin streaming para Streamlit
    
    # Configuración de timeout
    "timeout": 30,           # 30 segundos timeout
    
    # Configuración de conexión
    "keep_alive": "5m",      # Mantener modelo en memoria 5 minutos
    "num_thread": 4,         # Usar 4 threads
}

# Configuración específica para diferentes tipos de consultas
QUERY_CONFIGS = {
    "conversacion": {
        "temperature": 0.4,
        "num_predict": 150,
        "num_ctx": 1024,
    },
    "analisis": {
        "temperature": 0.2,
        "num_predict": 300,
        "num_ctx": 2048,
    },
    "generacion": {
        "temperature": 0.6,
        "num_predict": 500,
        "num_ctx": 3072,
    }
}

def get_optimized_config(query_type: str = "conversacion") -> dict:
    """Obtiene configuración optimizada según el tipo de consulta"""
    base_config = OLLAMA_PERFORMANCE_CONFIG.copy()
    if query_type in QUERY_CONFIGS:
        base_config.update(QUERY_CONFIGS[query_type])
    return base_config
