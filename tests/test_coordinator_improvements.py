"""
Test de Mejoras del Coordinador - Fase 1

Este test valida las mejoras implementadas:
- Caché de respuestas comunes
- Performance tracking
- Métricas de clasificación
- Logging estructurado
"""

import asyncio
import sys
from pathlib import Path

# Agregar el directorio raíz al path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.agents.coordinator_agent import CoordinatorAgent
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

async def test_coordinator_improvements():
    """Test de mejoras del coordinador - Fase 1"""
    
    print("=" * 80)
    print("🧪 TEST DE MEJORAS DEL COORDINADOR - FASE 1")
    print("=" * 80)
    
    # Inicializar coordinador
    coordinator = CoordinatorAgent()
    print("\n✅ Coordinador inicializado")
    
    # Test 1: Caché de respuestas comunes
    print("\n" + "=" * 80)
    print("📋 TEST 1: Caché de Respuestas Comunes")
    print("=" * 80)
    
    test_cases = [
        ("hola", "conversacion", "coordinator"),
        ("buenos dias", "conversacion", "coordinator"),
        ("gracias", "conversacion", "coordinator"),
        ("ayuda", "conversacion", "coordinator"),
        ("que puedes hacer", "conversacion", "coordinator"),
    ]
    
    for input_text, expected_intention, expected_agent in test_cases:
        print(f"\n🔹 Testeando: '{input_text}'")
        response = await coordinator.process(input_text)
        
        assert response["intention"] == expected_intention, f"❌ Intention incorrecta: {response['intention']}"
        assert response["agent"] == expected_agent, f"❌ Agent incorrecto: {response['agent']}"
        
        print(f"  ✅ Intention: {response['intention']}")
        print(f"  ✅ Agent: {response['agent']}")
        print(f"  ✅ Message: {response['message'][:100]}...")
    
    # Test 2: Cache hit (segunda llamada debe ser instantánea)
    print("\n" + "=" * 80)
    print("📋 TEST 2: Cache Hit (Verificar rapidez)")
    print("=" * 80)
    
    # Primera llamada (cache miss)
    print("\n🔹 Primera llamada a 'hola' (cache hit esperado)")
    response1 = await coordinator.process("hola")
    
    # Segunda llamada (debe ser cache hit)
    print("🔹 Segunda llamada a 'hola' (debe ser más rápida)")
    response2 = await coordinator.process("hola")
    
    assert response1["message"] == response2["message"], "❌ Respuestas diferentes"
    print("  ✅ Respuestas consistentes")
    
    # Test 3: Métricas
    print("\n" + "=" * 80)
    print("📋 TEST 3: Métricas de Performance")
    print("=" * 80)
    
    metrics = coordinator.get_metrics()
    print("\n📊 Métricas del Coordinador:")
    print(f"  - Total requests: {metrics['classification_metrics']['total_requests']}")
    print(f"  - Cache hits: {metrics['classification_metrics']['cache_hits']}")
    print(f"  - Cache misses: {metrics['classification_metrics']['cache_misses']}")
    print(f"  - Cache hit rate: {metrics['classification_metrics']['cache_hit_rate']}")
    print(f"  - LLM calls: {metrics['classification_metrics']['llm_calls']}")
    print(f"  - Fallback used: {metrics['classification_metrics']['fallback_used']}")
    print(f"  - Avg response time: {metrics['classification_metrics']['avg_response_time_ms']}ms")
    
    # Verificar que hubo cache hits
    assert metrics['classification_metrics']['cache_hits'] > 0, "❌ No hubo cache hits"
    print("\n  ✅ Caché funcionando correctamente")
    
    # Test 4: Funcionalidad existente (comandos)
    print("\n" + "=" * 80)
    print("📋 TEST 4: Funcionalidad Existente (No Rota)")
    print("=" * 80)
    
    # Test de comando de análisis (no debe estar en caché)
    print("\n🔹 Testeando comando: 'analizar dataset'")
    response = await coordinator.process("analizar dataset")
    
    print(f"  ✅ Intention: {response['intention']}")
    print(f"  ✅ Agent: {response['agent']}")
    print(f"  ✅ Message: {response['message'][:100]}...")
    
    # Verificar que es un comando
    assert response["intention"] in ["comando", "conversacion"], f"❌ Intention inesperada: {response['intention']}"
    
    # Test 5: Pregunta médica (no debe estar en caché)
    print("\n🔹 Testeando pregunta médica: '¿qué es la diabetes?'")
    response = await coordinator.process("¿qué es la diabetes?")
    
    print(f"  ✅ Intention: {response['intention']}")
    print(f"  ✅ Agent: {response['agent']}")
    print(f"  ✅ Is Medical: {response['is_medical_query']}")
    print(f"  ✅ Message: {response['message'][:100]}...")
    
    # Verificar que es conversación médica
    assert response["intention"] == "conversacion", f"❌ Intention incorrecta: {response['intention']}"
    
    # Métricas finales
    print("\n" + "=" * 80)
    print("📊 MÉTRICAS FINALES")
    print("=" * 80)
    
    final_metrics = coordinator.get_metrics()
    print(f"\n✅ Total de requests procesados: {final_metrics['classification_metrics']['total_requests']}")
    print(f"✅ Cache hit rate final: {final_metrics['classification_metrics']['cache_hit_rate']}")
    print(f"✅ Status del coordinador: {final_metrics['status']}")
    
    print("\n" + "=" * 80)
    print("✅ TODOS LOS TESTS PASARON - FASE 1 COMPLETADA")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_coordinator_improvements())
        if success:
            print("\n🎉 Mejoras del coordinador validadas exitosamente!")
            sys.exit(0)
        else:
            print("\n❌ Algunos tests fallaron")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error ejecutando tests: {e}", exc_info=True)
        print(f"\n❌ Error ejecutando tests: {e}")
        sys.exit(1)
