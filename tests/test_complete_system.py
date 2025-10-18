"""
Test completo del sistema de análisis tras el fix
"""
import asyncio
import pandas as pd
from src.orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator

async def test_complete_analysis():
    print("=" * 80)
    print("🧪 TEST COMPLETO: Análisis del Sistema tras Fix EDA")
    print("=" * 80)
    
    # Cargar dataset
    print("\n1️⃣ Cargando dataset...")
    df = pd.read_csv('data/real/df_final_v2.csv')
    print(f"   ✅ Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    # Inicializar orchestrator
    print("\n2️⃣ Inicializando orchestrator...")
    orchestrator = MedicalAgentsOrchestrator()
    
    # Solicitar análisis
    print("\n3️⃣ Solicitando análisis integral del dataset...")
    request = {
        "action": "analizar",
        "df": df
    }
    
    result = await orchestrator.arun(request)
    
    # Mostrar resultado
    print("\n" + "=" * 80)
    print("📊 RESULTADO DEL ANÁLISIS")
    print("=" * 80)
    
    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
        return False
    
    message = result.get("message", "")
    print(f"\n{message}")
    
    # Verificar que el análisis sea completo
    print("\n" + "=" * 80)
    print("✅ VERIFICACIÓN DE COMPLETITUD")
    print("=" * 80)
    
    checks = {
        "Menciona estadísticas (mean, std, etc.)": any(word in message.lower() for word in ['media', 'mean', 'desviación', 'std', 'promedio']),
        "Menciona correlaciones": 'correlaci' in message.lower(),
        "Menciona valores nulos": any(word in message.lower() for word in ['nulo', 'missing', 'faltante']),
        "Menciona columnas específicas": any(word in message.lower() for word in ['edad', 'age', 'paciente', 'patient']),
        "Menciona distribución/rango": any(word in message.lower() for word in ['distribución', 'rango', 'mínimo', 'máximo', 'min', 'max']),
        "Es suficientemente largo (>500 chars)": len(message) > 500
    }
    
    passed = 0
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")
        if result:
            passed += 1
    
    print(f"\n📊 Checks pasados: {passed}/{len(checks)}")
    
    if passed >= 4:
        print("\n🎉 ¡El análisis es completo y detallado!")
        return True
    else:
        print("\n⚠️ El análisis aún parece limitado")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_analysis())
    exit(0 if success else 1)
