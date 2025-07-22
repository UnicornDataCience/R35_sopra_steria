#!/usr/bin/env python3
"""
Test integrado del agente simulador con el orquestador optimizado.
Verifica que la simulación de evolución temporal de pacientes funciona correctamente.
"""

import asyncio
import pandas as pd
import json
import os
import sys
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.orchestration.fast_orchestrator import FastMedicalOrchestrator
from src.agents.simulator_agent import PatientSimulatorAgent

def create_mock_covid_data():
    """Crea datos de prueba simulando pacientes COVID-19"""
    return pd.DataFrame({
        'patient_id': [f'COVID_{i:03d}' for i in range(1, 11)],
        'age': [45, 67, 32, 78, 54, 29, 61, 38, 72, 41],
        'gender': ['MALE', 'FEMALE', 'MALE', 'FEMALE', 'MALE', 'FEMALE', 'MALE', 'FEMALE', 'MALE', 'FEMALE'],
        'diagnosis': ['COVID-19 - POSITIVO'] * 10,
        'medication': ['PARACETAMOL', 'DEXAMETASONA', 'REMDESIVIR', 'PARACETAMOL', 
                      'REMDESIVIR', 'PARACETAMOL', 'DEXAMETASONA', 'REMDESIVIR', 
                      'DEXAMETASONA', 'PARACETAMOL'],
        'icu_days': [0, 5, 0, 12, 3, 0, 7, 1, 15, 0],
        'temperature': [37.2, 38.9, 36.8, 39.5, 38.1, 36.9, 38.7, 37.1, 39.2, 37.0],
        'oxygen_saturation': [96.5, 89.2, 98.1, 85.3, 92.7, 97.8, 88.9, 95.2, 84.1, 98.5],
        'pcr_result': [2.1, 18.7, 0.8, 25.4, 12.3, 1.2, 15.8, 3.4, 22.1, 0.9],
        'discharge_motive': ['Domicilio', 'UCI', 'Domicilio', 'Exitus', 'Domicilio', 
                           'Domicilio', 'UCI', 'Domicilio', 'UCI', 'Domicilio']
    })

def create_mock_general_data():
    """Crea datos de prueba para casos médicos generales"""
    return pd.DataFrame({
        'patient_id': [f'GEN_{i:03d}' for i in range(1, 6)],
        'age': [52, 34, 68, 41, 59],
        'gender': ['FEMALE', 'MALE', 'FEMALE', 'MALE', 'FEMALE'],
        'diagnosis': ['Diabetes Tipo 2', 'Hipertensión', 'Cardiopatía', 'Diabetes Tipo 1', 'Hipertensión'],
        'medication': ['METFORMINA', 'ENALAPRIL', 'ATORVASTATINA', 'INSULINA', 'LOSARTAN'],
        'icu_days': [0, 0, 2, 0, 1],
        'temperature': [36.8, 37.0, 37.5, 36.9, 37.2],
        'oxygen_saturation': [98.2, 97.8, 94.5, 98.0, 96.8],
        'pcr_result': [1.2, 0.8, 8.5, 1.0, 2.1],
        'discharge_motive': ['Domicilio', 'Domicilio', 'Domicilio', 'Domicilio', 'Domicilio']
    })

async def test_simulator_agent_standalone():
    """Prueba el agente simulador de forma independiente"""
    print("🧪 Probando agente simulador de forma independiente...")
    
    simulator = PatientSimulatorAgent()
    
    # Datos de prueba COVID-19
    covid_data = create_mock_covid_data()
    context_covid = {
        "synthetic_data": covid_data,
        "universal_analysis": {"dataset_type": "COVID-19"}
    }
    
    try:
        response = await simulator.process("Simula la evolución de estos pacientes COVID-19", context_covid)
        
        print("✅ Simulación COVID-19 completada")
        print(f"📊 Agente: {response.get('agent', 'N/A')}")
        print(f"📈 Estadísticas disponibles: {bool(response.get('simulation_stats'))}")
        
        # Manejar DataFrame de forma segura
        evolved_data = response.get('evolved_data')
        if evolved_data is not None:
            if hasattr(evolved_data, 'empty'):  # Es un DataFrame
                print(f"🏥 Datos evolucionados disponibles: {not evolved_data.empty} ({len(evolved_data)} filas)")
            else:
                print(f"🏥 Datos evolucionados disponibles: {bool(evolved_data)}")
        else:
            print("🏥 Datos evolucionados disponibles: False")
        
        if response.get('simulation_stats'):
            stats = response['simulation_stats']
            print(f"   - Total pacientes: {stats.get('total_patients', 0)}")
            print(f"   - Total visitas: {stats.get('total_visits', 0)}")
            print(f"   - Promedio visitas/paciente: {stats.get('avg_visits_per_patient', 0):.1f}")
            print(f"   - Pacientes con mejoría: {stats.get('patients_with_improvement', 0)}")
            print(f"   - Pacientes con deterioro: {stats.get('patients_with_deterioration', 0)}")
        
        if response.get('message'):
            print(f"📝 Informe LLM: {response['message'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en simulación independiente: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_simulator_via_orchestrator():
    """Prueba el simulador a través del orquestador"""
    print("\n🧪 Probando simulador vía FastOrchestrator...")
    
    # Crear orquestador con el agente simulador
    simulator = PatientSimulatorAgent()
    agents = {"simulator": simulator}
    orchestrator = FastMedicalOrchestrator(agents)
    
    # Datos de prueba generales
    general_data = create_mock_general_data()
    context = {
        "synthetic_data": general_data,
        "universal_analysis": {"dataset_type": "general"}
    }
    
    try:
        response = await orchestrator.process_user_input(
            "Simula la evolución de estos pacientes", 
            context
        )
        
        print("✅ Simulación vía orquestador completada")
        print(f"🎯 Ruta detectada: {response.get('route', 'N/A')}")
        print(f"📊 Agente ejecutado: {response.get('agent', 'N/A')}")
        print(f"⏱️ Tiempo total: {response.get('total_time', 'N/A')}")
        
        if response.get('simulation_stats'):
            stats = response['simulation_stats']
            print(f"📈 Estadísticas de simulación:")
            print(f"   - Total pacientes: {stats.get('total_patients', 0)}")
            print(f"   - Total visitas: {stats.get('total_visits', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en simulación vía orquestador: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_simulation_with_different_diseases():
    """Prueba la simulación con diferentes tipos de enfermedades"""
    print("\n🧪 Probando simulación con diferentes tipos de enfermedades...")
    
    simulator = PatientSimulatorAgent()
    
    # Test 1: COVID-19
    covid_data = create_mock_covid_data()
    context_covid = {
        "synthetic_data": covid_data,
        "universal_analysis": {"dataset_type": "COVID-19"}
    }
    
    # Test 2: Casos generales
    general_data = create_mock_general_data()
    context_general = {
        "synthetic_data": general_data,
        "universal_analysis": {"dataset_type": "general"}
    }
    
    tests = [
        ("COVID-19", context_covid),
        ("General", context_general)
    ]
    
    results = []
    
    for disease_type, context in tests:
        try:
            print(f"\n🦠 Simulando tipo: {disease_type}")
            response = await simulator.process(f"Simula evolución para {disease_type}", context)
            
            if response.get('error'):
                print(f"❌ Error en {disease_type}: {response.get('message', 'Error desconocido')}")
                results.append(False)
            else:
                print(f"✅ {disease_type} simulado correctamente")
                stats = response.get('simulation_stats', {})
                print(f"   - Visitas simuladas: {stats.get('total_visits', 0)}")
                results.append(True)
                
        except Exception as e:
            print(f"❌ Excepción en {disease_type}: {e}")
            results.append(False)
    
    return all(results)

async def main():
    """Función principal de testing"""
    print(f"🚀 Iniciando tests del simulador - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)
    
    tests = [
        ("Simulador independiente", test_simulator_agent_standalone),
        ("Simulador vía orquestador", test_simulator_via_orchestrator),
        ("Simulación por tipos de enfermedad", test_simulation_with_different_diseases)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 EJECUTANDO: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
            status = "✅ PASÓ" if result else "❌ FALLÓ"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR CRÍTICO en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print(f"\n{'='*70}")
    print("📋 RESUMEN DE TESTS DEL SIMULADOR")
    print(f"{'='*70}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 RESULTADO FINAL: {passed}/{total} tests pasaron")
    
    if passed == total:
        print("🎉 ¡Todos los tests del simulador pasaron!")
        return True
    else:
        print("⚠️ Algunos tests fallaron. Revisar implementación.")
        return False

if __name__ == "__main__":
    asyncio.run(main())
