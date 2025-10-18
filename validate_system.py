"""
Script de Validación del Sistema Patient-IA

Verifica que el sistema está funcionando correctamente antes de aplicar mejoras.
Ejecutar este script después de cada fase de optimización.

Uso:
    python validate_system.py
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.optimization_utils import validate_system_state, log_dataframe_info
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

def main():
    """Validación principal del sistema"""
    
    print("=" * 80)
    print("🔍 VALIDACIÓN DEL SISTEMA PATIENT-IA")
    print("=" * 80)
    print()
    
    # 1. Validar estado del sistema
    print("1️⃣ Validando estructura del sistema...")
    state = validate_system_state()
    
    passed = sum(state.values())
    total = len(state)
    
    if passed == total:
        print(f"   ✅ Todas las validaciones pasaron ({passed}/{total})")
    else:
        print(f"   ⚠️ Algunas validaciones fallaron ({passed}/{total})")
        print("\n   Detalles de fallos:")
        for key, value in state.items():
            if not value:
                print(f"     ❌ {key}")
        print()
    
    # 2. Verificar que los módulos se pueden importar
    print("\n2️⃣ Verificando imports de módulos...")
    modules_to_test = [
        'src.agents.base_agent',
        'src.agents.coordinator_agent',
        'src.agents.analyzer_agent',
        'src.agents.generator_agent',
        'src.agents.validator_agent',
        'src.agents.evaluator_agent',
        'src.agents.simulator_agent',
        'src.utils.optimization_utils'
    ]
    
    import_errors = []
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"   ✅ {module_name}")
        except Exception as e:
            print(f"   ❌ {module_name}: {e}")
            import_errors.append((module_name, str(e)))
    
    # 3. Verificar API server (opcional - solo si está corriendo)
    print("\n3️⃣ Verificando API server...")
    try:
        import requests
        response = requests.get('http://localhost:8000/health', timeout=2)
        if response.status_code == 200:
            print("   ✅ API server está corriendo")
        else:
            print(f"   ⚠️ API server respondió con código: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ℹ️ API server no está corriendo (esto es normal si no lo has iniciado)")
    except Exception as e:
        print(f"   ⚠️ Error verificando API: {e}")
    
    # 4. Probar utilidades de optimización
    print("\n4️⃣ Probando utilidades de optimización...")
    try:
        # Probar CacheManager
        from src.utils.optimization_utils import CacheManager
        cache = CacheManager(cache_dir='cache_test')
        cache.save('test_key', {'data': 'test'}, 'test_category')
        loaded = cache.load('test_key', 'test_category')
        
        if loaded and loaded.get('data') == 'test':
            print("   ✅ CacheManager funcionando")
            cache.clear('test_category')
        else:
            print("   ❌ CacheManager falló")
        
        # Probar hashing
        import pandas as pd
        from src.utils.optimization_utils import get_dataframe_hash
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        hash_val = get_dataframe_hash(test_df)
        
        if hash_val and len(hash_val) == 16:
            print("   ✅ DataFrame hashing funcionando")
        else:
            print("   ❌ DataFrame hashing falló")
        
        # Probar PerformanceTracker
        from src.utils.optimization_utils import PerformanceTracker
        tracker = PerformanceTracker('test_operation')
        tracker.start()
        import time
        time.sleep(0.1)
        tracker.stop()
        metrics = tracker.get_metrics()
        
        if 'duration_seconds' in metrics and metrics['duration_seconds'] > 0:
            print("   ✅ PerformanceTracker funcionando")
        else:
            print("   ❌ PerformanceTracker falló")
            
    except Exception as e:
        print(f"   ❌ Error probando utilidades: {e}")
        import traceback
        traceback.print_exc()
    
    # Resumen final
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE VALIDACIÓN")
    print("=" * 80)
    
    all_passed = (passed == total and len(import_errors) == 0)
    
    if all_passed:
        print("✅ SISTEMA VALIDADO - Listo para optimizaciones")
        return 0
    else:
        print("⚠️ SISTEMA CON WARNINGS - Revisar detalles arriba")
        if import_errors:
            print("\nErrores de import:")
            for module, error in import_errors:
                print(f"  - {module}: {error}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
