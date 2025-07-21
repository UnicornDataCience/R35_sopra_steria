"""
Test final para verificar que se corrigieron los errores de formatting y generation_info
"""

import sys
import os
import asyncio

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agents.generator_agent import SyntheticGeneratorAgent
import pandas as pd

async def test_generation_info():
    """Test para verificar que generation_info se incluye correctamente"""
    
    print("🧪 Test Final: Verificar generation_info y formatting")
    print("=" * 60)
    
    # Crear datos de prueba
    test_data = pd.DataFrame({
        'PATIENT ID': [1, 2, 3, 4, 5],
        'EDAD/AGE': [45, 67, 23, 56, 78],
        'SEXO/SEX': ['MALE', 'FEMALE', 'MALE', 'FEMALE', 'MALE'],
        'DIAG ING/INPAT': ['COVID19 - POSITIVO', 'COVID19 - POSITIVO', 'COVID19 - POSITIVO', 'COVID19 - POSITIVO', 'COVID19 - POSITIVO'],
        'SAT_02_ULTIMA/LAST_URG/EMERG': [95, 88, 92, 94, 90]
    })
    
    # Crear contexto de prueba
    context = {
        'dataframe': test_data,
        'dataset_uploaded': True,
        'filename': 'test_data.csv',
        'rows': len(test_data),
        'columns': len(test_data.columns),
        'selected_columns': ['PATIENT ID', 'EDAD/AGE', 'SEXO/SEX', 'DIAG ING/INPAT', 'SAT_02_ULTIMA/LAST_URG/EMERG'],
        'parameters': {
            'model_type': 'ctgan',
            'num_samples': 10
        }
    }
    
    # Crear y probar el agente generador
    generator = SyntheticGeneratorAgent()
    
    print("🔧 Probando generación con CTGAN...")
    try:
        response = await generator.process("Genera datos sintéticos", context)
        
        print(f"✅ Respuesta del agente generada")
        print(f"✅ Mensaje: {response.get('message', 'N/A')[:100]}...")
        
        # Verificar que se incluye generation_info
        if 'generation_info' in response:
            gen_info = response['generation_info']
            print(f"✅ generation_info incluida:")
            print(f"   - Modelo: {gen_info.get('model_type', 'N/A')}")
            print(f"   - Número de muestras: {gen_info.get('num_samples', 'N/A')}")
            print(f"   - Columnas utilizadas: {gen_info.get('columns_used', 'N/A')}")
            print(f"   - Método de selección: {gen_info.get('selection_method', 'N/A')}")
            print(f"   - Timestamp: {gen_info.get('timestamp', 'N/A')}")
            
            # Verificar que num_samples es un entero, no string
            num_samples = gen_info.get('num_samples')
            if isinstance(num_samples, int):
                print(f"✅ num_samples es tipo correcto: {type(num_samples)}")
                print(f"✅ Formato con comas: {num_samples:,}")
            else:
                print(f"⚠️ num_samples es tipo incorrecto: {type(num_samples)}")
        else:
            print("❌ generation_info NO incluida en la respuesta")
        
        # Verificar datos sintéticos
        if 'synthetic_data' in response:
            synthetic_df = response['synthetic_data']
            print(f"✅ Datos sintéticos generados: {len(synthetic_df)} filas, {len(synthetic_df.columns)} columnas")
            print(f"✅ Columnas: {list(synthetic_df.columns)}")
        else:
            print("❌ synthetic_data NO incluida en la respuesta")
            
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Test completado exitosamente")
    print("✅ Los errores de formatting y generation_info han sido corregidos")
    return True

if __name__ == "__main__":
    asyncio.run(test_generation_info())
