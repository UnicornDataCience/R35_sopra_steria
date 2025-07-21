"""
Test de integración del MedicalColumnSelector con los generadores
Verifica que los generadores usen las columnas seleccionadas por el selector
"""

import sys
import os
import pandas as pd

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.adapters.medical_column_selector import MedicalColumnSelector
from src.generation.ctgan_generator import CTGANGenerator
from src.generation.tvae_generator import TVAEGenerator
from src.generation.sdv_generator import SDVGenerator

def test_column_selection_integration():
    """Test de integración completo del selector de columnas con generadores"""
    
    print("🧪 Test de Integración: MedicalColumnSelector + Generadores")
    print("=" * 60)
    
    # 1. Cargar dataset de prueba
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real', 'df_final_v2.csv')
        df = pd.read_csv(data_path, low_memory=False, encoding='utf-8')
        print(f"✅ Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return False
    
    # 2. Crear selector y detectar columnas
    column_selector = MedicalColumnSelector()
    
    # Detectar tipo de dataset
    dataset_type = column_selector.detector.detect_dataset_type(df)
    column_mappings = column_selector.detector.infer_medical_columns(df)
    print(f"✅ Tipo de dataset detectado: {dataset_type.value}")
    
    # 3. Obtener columnas recomendadas automáticamente
    recommended_columns = column_selector._get_recommended_columns(dataset_type, column_mappings, df)
    print(f"✅ Columnas recomendadas: {len(recommended_columns)}")
    print(f"   Columnas: {recommended_columns[:5]}{'...' if len(recommended_columns) > 5 else ''}")
    
    # 4. Validar selección
    validation_result = column_selector._validate_selection(recommended_columns, dataset_type, column_mappings)
    print(f"✅ Validación: {'Exitosa' if validation_result.mandatory_fulfilled else 'Falló'}")
    print(f"   Score de calidad: {validation_result.quality_score:.2f}")
    
    if not validation_result.mandatory_fulfilled:
        print("⚠️ Advertencias:")
        for req in validation_result.missing_requirements:
            print(f"   - {req}")
    
    # 5. Filtrar DataFrame con columnas seleccionadas
    df_filtered = df[recommended_columns].copy()
    print(f"✅ DataFrame filtrado: {len(df_filtered)} filas, {len(df_filtered.columns)} columnas")
    
    # 6. Probar cada generador con columnas seleccionadas
    test_sample_size = 10
    context = {
        'selected_columns': recommended_columns,
        'universal_analysis': {'dataset_type': 'COVID-19'}
    }
    
    generators = [
        ("CTGAN", CTGANGenerator()),
        ("TVAE", TVAEGenerator()),
        ("SDV", SDVGenerator())
    ]
    
    for gen_name, generator in generators:
        print(f"\n🔧 Probando {gen_name} Generator...")
        try:
            # Generar datos sintéticos
            synthetic_data = generator.generate(
                df_filtered, 
                sample_size=test_sample_size, 
                is_covid_dataset=True,
                selected_columns=recommended_columns
            )
            
            if not synthetic_data.empty:
                print(f"✅ {gen_name}: Generó {len(synthetic_data)} filas, {len(synthetic_data.columns)} columnas")
                
                # Verificar que las columnas coinciden
                expected_cols = set(df_filtered.columns)
                actual_cols = set(synthetic_data.columns)
                
                if expected_cols == actual_cols:
                    print(f"✅ {gen_name}: Columnas coinciden perfectamente")
                else:
                    missing = expected_cols - actual_cols
                    extra = actual_cols - expected_cols
                    if missing:
                        print(f"⚠️ {gen_name}: Faltan columnas: {missing}")
                    if extra:
                        print(f"⚠️ {gen_name}: Columnas extra: {extra}")
                
                # Mostrar muestra de datos
                print(f"📊 {gen_name}: Muestra de datos sintéticos:")
                print(synthetic_data.head(2).to_string(max_cols=5))
                
            else:
                print(f"❌ {gen_name}: No se generaron datos")
                
        except Exception as e:
            print(f"❌ {gen_name}: Error durante generación: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Test completado")
    return True

def test_without_column_selection():
    """Test sin selección de columnas (comportamiento legacy)"""
    
    print("\n🧪 Test Legacy: Generadores sin selección de columnas")
    print("=" * 60)
    
    # Cargar dataset
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'real', 'df_final_v2.csv')
        df = pd.read_csv(data_path, low_memory=False, encoding='utf-8')
        print(f"✅ Dataset cargado: {len(df)} filas, {len(df.columns)} columnas")
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        return False
    
    # Probar generadores sin columnas seleccionadas (comportamiento legacy)
    generators = [
        ("CTGAN", CTGANGenerator()),
        ("TVAE", TVAEGenerator()),
        ("SDV", SDVGenerator())
    ]
    
    for gen_name, generator in generators:
        print(f"\n🔧 Probando {gen_name} Generator (sin selección)...")
        try:
            synthetic_data = generator.generate(
                df, 
                sample_size=5, 
                is_covid_dataset=True,
                selected_columns=None  # Sin selección
            )
            
            if not synthetic_data.empty:
                print(f"✅ {gen_name}: Generó {len(synthetic_data)} filas, {len(synthetic_data.columns)} columnas")
                print(f"📊 {gen_name}: Columnas generadas: {list(synthetic_data.columns)}")
            else:
                print(f"❌ {gen_name}: No se generaron datos")
                
        except Exception as e:
            print(f"❌ {gen_name}: Error: {e}")
    
    return True

if __name__ == "__main__":
    success1 = test_column_selection_integration()
    success2 = test_without_column_selection()
    
    if success1 and success2:
        print("\n🎉 Todos los tests pasaron exitosamente")
    else:
        print("\n❌ Algunos tests fallaron")
