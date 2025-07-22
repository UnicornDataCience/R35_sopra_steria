#!/usr/bin/env python3
"""
Test para verificar que el validador funciona con datos reales
"""

import sys
import os
import pandas as pd
import asyncio

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

# FORZAR GROQ - Sobrescribir variable del sistema si FORCE_GROQ=true
if os.getenv('FORCE_GROQ', 'false').lower() == 'true':
    os.environ['LLM_PROVIDER'] = 'groq'
    print("🚀 [DEBUG] Variable LLM_PROVIDER forzada a 'groq'")

async def test_validator_with_real_data():
    """Test del validador con datos reales únicamente"""
    
    print("🧪 Testing Validator with Real Data Only")
    print("=" * 50)
    
    try:
        from src.agents.validator_agent import MedicalValidatorAgent
        
        # Crear instancia del validador
        validator = MedicalValidatorAgent()
        print("✅ Validator agent creado correctamente")
        
        # Crear datos de prueba (simulando datos reales cargados)
        test_data = pd.DataFrame([
            {
                "PATIENT ID": "P001",
                "EDAD/AGE": 45,
                "SEXO/SEX": "MALE",
                "TEMP_ING/INPAT": 36.5,
                "SAT_02_ING/INPAT": 95,
                "DIAG ING/INPAT": "COVID19 - POSITIVO"
            },
            {
                "PATIENT ID": "P002", 
                "EDAD/AGE": 32,
                "SEXO/SEX": "FEMALE",
                "TEMP_ING/INPAT": 37.2,
                "SAT_02_ING/INPAT": 98,
                "DIAG ING/INPAT": "COVID19 - NEGATIVO"
            },
            {
                "PATIENT ID": "P003",
                "EDAD/AGE": 67,
                "SEXO/SEX": "MALE", 
                "TEMP_ING/INPAT": 38.1,
                "SAT_02_ING/INPAT": 92,
                "DIAG ING/INPAT": "COVID19 - POSITIVO"
            }
        ])
        
        # Contexto con solo datos reales (sin synthetic_data)
        context = {
            "dataframe": test_data,
            "universal_analysis": {
                "dataset_type": "COVID-19"
            }
        }
        
        print("📊 Datos de prueba creados:")
        print(f"   - Registros: {len(test_data)}")
        print(f"   - Columnas: {test_data.columns.tolist()}")
        
        # Ejecutar validación
        print("\n🔍 Ejecutando validación...")
        result = await validator.process("Valida estos datos médicos", context)
        
        if result.get('error'):
            print(f"❌ Error en validación: {result.get('message')}")
            return False
        else:
            print("✅ Validación completada exitosamente")
            print(f"   - Agente: {result.get('agent')}")
            print(f"   - Modo: {result.get('validation_mode', 'N/A')}")
            
            # Mostrar extracto del mensaje
            message = result.get('message', '')
            if len(message) > 200:
                print(f"   - Mensaje: {message[:200]}...")
            else:
                print(f"   - Mensaje: {message}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_validator_with_synthetic_data():
    """Test del validador con datos sintéticos (modo original)"""
    
    print("\n🧪 Testing Validator with Synthetic Data")
    print("=" * 50)
    
    try:
        from src.agents.validator_agent import MedicalValidatorAgent
        
        # Crear instancia del validador
        validator = MedicalValidatorAgent()
        
        # Crear datos originales y sintéticos
        original_data = pd.DataFrame([
            {"PATIENT ID": "P001", "EDAD/AGE": 45, "SEXO/SEX": "MALE"},
            {"PATIENT ID": "P002", "EDAD/AGE": 32, "SEXO/SEX": "FEMALE"}
        ])
        
        synthetic_data = pd.DataFrame([
            {"PATIENT ID": "S001", "EDAD/AGE": 47, "SEXO/SEX": "MALE"},
            {"PATIENT ID": "S002", "EDAD/AGE": 29, "SEXO/SEX": "FEMALE"}
        ])
        
        # Contexto con ambos tipos de datos
        context = {
            "dataframe": original_data,
            "synthetic_data": synthetic_data,
            "universal_analysis": {
                "dataset_type": "COVID-19"
            }
        }
        
        print("📊 Datos de prueba creados:")
        print(f"   - Originales: {len(original_data)} registros")
        print(f"   - Sintéticos: {len(synthetic_data)} registros")
        
        # Ejecutar validación
        result = await validator.process("Valida estos datos sintéticos", context)
        
        if result.get('error'):
            print(f"❌ Error en validación: {result.get('message')}")
            return False
        else:
            print("✅ Validación completada exitosamente")
            print(f"   - Modo: {result.get('validation_mode', 'N/A')}")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    """Función principal"""
    
    print("🔬 Testing Validator Agent Fixes")
    print("=" * 60)
    
    # Test 1: Solo datos reales
    success1 = await test_validator_with_real_data()
    
    # Test 2: Datos sintéticos (verificar que sigue funcionando)
    success2 = await test_validator_with_synthetic_data()
    
    print("\n📋 Resumen de Tests:")
    print(f"   - Datos reales: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   - Datos sintéticos: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 TODOS LOS TESTS PASARON - Validator funciona correctamente")
    else:
        print("\n❌ ALGUNOS TESTS FALLARON - Revisar configuración")

if __name__ == "__main__":
    asyncio.run(main())
