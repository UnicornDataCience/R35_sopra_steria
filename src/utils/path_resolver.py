import os

class PathResolver:
    """Resuelve rutas del proyecto automáticamente"""
    
    @staticmethod
    def get_project_root():
        """Encuentra la raíz del proyecto automáticamente"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Buscar hacia arriba hasta encontrar el directorio que contiene 'data'
        while current_dir != os.path.dirname(current_dir):
            if os.path.exists(os.path.join(current_dir, 'data')):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        
        # Fallback: usar directorio actual
        return os.getcwd()
    
    @staticmethod
    def get_synthetic_dir():
        """Obtiene el directorio de datos sintéticos"""
        return os.path.join(PathResolver.get_project_root(), 'data', 'synthetic')

def get_synthetic_files():
    """Obtiene las rutas de todos los archivos sintéticos"""
    synthetic_dir = PathResolver.get_synthetic_dir()
    
    return {
        'sdv_json': os.path.join(synthetic_dir, 'datos_sinteticos_sdv.json'),
        'tvae_json': os.path.join(synthetic_dir, 'datos_sinteticos_tvae.json'),
        'ctgan_json': os.path.join(synthetic_dir, 'datos_sinteticos_ctgan.json'),
    }