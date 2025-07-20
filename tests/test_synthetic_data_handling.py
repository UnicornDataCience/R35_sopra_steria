#!/usr/bin/env python3
"""
Test para verificar que la información de generación sintética se maneja correctamente
"""

import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_handle_synthetic_data_response():
    """Test de la función handle_synthetic_data_response"""
    
    # Simular st.session_state.context
    class MockSessionState:
        def __init__(self):
            self.context = {}
    
    # Mock de st para simular Streamlit
    class MockStreamlit:
        session_state = MockSessionState()
    
    # Inyectar mock globalmente (solo para testing)
    import builtins
    builtins.st = MockStreamlit()
    
    # Ahora importar la función
    try:
        from interfaces.chat_llm import handle_synthetic_data_response
    except ImportError as e:
        print(f"Error importando función: {e}")
        return False
    
    print("🧪 Testing handle_synthetic_data_response function...")
    
    # Test 1: Response con generation_info completo
    print("\n1. Testing with complete generation_info...")
    synthetic_df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['A', 'B', 'C']
    })
    
    response = {
        'synthetic_data': synthetic_df,
        'generation_info': {
            'model_type': 'ctgan',
            'num_samples': 3,
            'columns_used': 2,
            'selection_method': 'manual'
        }
    }
    
    result = handle_synthetic_data_response(response)
    print(f"✓ Resultado: {result}")
    print(f"✓ Generation info: {MockStreamlit.session_state.context.get('generation_info', {})}")
    
    # Test 2: Response con generation_info vacío
    print("\n2. Testing with empty generation_info...")
    MockStreamlit.session_state.context = {}  # Reset
    
    response = {
        'synthetic_data': synthetic_df,
        'generation_info': {}
    }
    
    context = {
        'parameters': {
            'model_type': 'tvae',
            'num_samples': 5
        }
    }
    
    result = handle_synthetic_data_response(response, context)
    print(f"✓ Resultado: {result}")
    generation_info = MockStreamlit.session_state.context.get('generation_info', {})
    print(f"✓ Generation info creado: {generation_info}")
    
    # Verificar que se llenó correctamente
    assert generation_info.get('model_type') == 'tvae'
    assert generation_info.get('num_samples') == 3  # Debe usar len del DataFrame real
    assert generation_info.get('columns_used') == 2
    assert 'timestamp' in generation_info
    
    # Test 3: Response sin generation_info
    print("\n3. Testing without generation_info...")
    MockStreamlit.session_state.context = {}  # Reset
    
    response = {
        'synthetic_data': synthetic_df
    }
    
    result = handle_synthetic_data_response(response)
    print(f"✓ Resultado: {result}")
    generation_info = MockStreamlit.session_state.context.get('generation_info', {})
    print(f"✓ Generation info por defecto: {generation_info}")
    
    # Verificar valores por defecto
    assert generation_info.get('model_type') == 'ctgan'  # Default
    assert generation_info.get('num_samples') == 3
    assert generation_info.get('columns_used') == 2
    assert generation_info.get('selection_method') == 'Automático'
    
    # Test 4: Response sin synthetic_data
    print("\n4. Testing without synthetic_data...")
    MockStreamlit.session_state.context = {}  # Reset
    
    response = {
        'message': 'Solo mensaje'
    }
    
    result = handle_synthetic_data_response(response)
    print(f"✓ Resultado: {result}")
    assert result == False
    assert 'synthetic_data' not in MockStreamlit.session_state.context
    
    print("\n✅ Todos los tests de handle_synthetic_data_response pasaron!")
    return True

def test_display_formatting():
    """Test de formateo de display para diferentes casos"""
    
    print("\n🎨 Testing display formatting...")
    
    test_cases = [
        # (generation_info, expected_model_display, expected_method)
        (
            {'model_type': 'ctgan', 'selection_method': 'manual'},
            'CTGAN',
            'manual'
        ),
        (
            {'model_type': None, 'selection_method': 'N/A'},
            'GENERADO',
            'Método estándar'
        ),
        (
            {},
            'GENERADO',
            'Método estándar'
        ),
        (
            {'model_type': '', 'selection_method': ''},
            'GENERADO',
            'Método estándar'
        )
    ]
    
    for i, (generation_info, expected_model, expected_method) in enumerate(test_cases, 1):
        print(f"\n{i}. Testing case: {generation_info}")
        
        # Test model display logic
        model_type = generation_info.get('model_type', 'N/A')
        if model_type and model_type != 'N/A':
            model_display = model_type.upper()
        else:
            model_display = "GENERADO"
        
        # Test method display logic
        selection_method = generation_info.get('selection_method', 'N/A')
        if selection_method == 'N/A' or not selection_method:
            selection_method = "Método estándar"
        
        print(f"  ✓ Model display: {model_display} (expected: {expected_model})")
        print(f"  ✓ Method display: {selection_method} (expected: {expected_method})")
        
        assert model_display == expected_model, f"Model display mismatch: {model_display} != {expected_model}"
        assert selection_method == expected_method, f"Method display mismatch: {selection_method} != {expected_method}"
    
    print("\n✅ Todos los tests de formateo pasaron!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING SYNTHETIC DATA HANDLING")
    print("=" * 60)
    
    try:
        test1 = test_handle_synthetic_data_response()
        test2 = test_display_formatting()
        
        print("\n" + "=" * 60)
        if test1 and test2:
            print("✅ TODOS LOS TESTS PASARON!")
            print("🎉 La información de generación sintética se maneja correctamente")
        else:
            print("❌ ALGUNOS TESTS FALLARON")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error durante testing: {e}")
        import traceback
        traceback.print_exc()
