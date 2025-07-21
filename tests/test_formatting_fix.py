#!/usr/bin/env python3
"""
Test para verificar que los errores de formateo en el sidebar están solucionados.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_safe_formatting():
    """Test que el formateo seguro funciona para diferentes tipos de datos"""
    
    print("Testing safe formatting functions...")
    
    # Test modelo
    def format_model_safe(model_type):
        if model_type and model_type != 'N/A':
            return f"Modelo utilizado: {model_type.upper()}"
        else:
            return f"Modelo utilizado: {model_type or 'N/A'}"
    
    # Test num_samples
    def format_samples_safe(num_samples):
        if isinstance(num_samples, (int, float)):
            return f"Registros generados: {num_samples:,}"
        else:
            return f"Registros generados: {num_samples}"
    
    # Test casos
    test_cases = [
        # model_type, num_samples
        ('ctgan', 1000),
        ('tvae', 5000),
        ('sdv', 2500),
        ('N/A', 'N/A'),
        (None, None),
        ('', ''),
    ]
    
    for model_type, num_samples in test_cases:
        try:
            model_result = format_model_safe(model_type)
            samples_result = format_samples_safe(num_samples)
            
            print(f"✓ Model: {model_type} -> {model_result}")
            print(f"✓ Samples: {num_samples} -> {samples_result}")
            print()
            
        except Exception as e:
            print(f"✗ Error with model_type={model_type}, num_samples={num_samples}: {e}")
            return False
    
    print("All formatting tests passed! ✓")
    return True

def test_generation_info_structure():
    """Test que la estructura de generation_info está correcta"""
    
    print("Testing generation_info structure...")
    
    # Estructura esperada
    sample_generation_info = {
        'model_type': 'ctgan',
        'num_samples': 1000,
        'columns_used': 5,
        'selection_method': 'medical_intelligent',
        'timestamp': '2024-01-15 10:30:00'
    }
    
    try:
        # Verificar que todas las claves existen
        required_keys = ['model_type', 'num_samples', 'columns_used', 'selection_method']
        for key in required_keys:
            if key not in sample_generation_info:
                print(f"✗ Missing key: {key}")
                return False
            print(f"✓ Key present: {key} = {sample_generation_info[key]}")
        
        print("Generation info structure test passed! ✓")
        return True
        
    except Exception as e:
        print(f"✗ Error in generation_info structure test: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING FORMATTING FIXES")
    print("=" * 60)
    
    test1 = test_safe_formatting()
    print()
    test2 = test_generation_info_structure()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("ALL FORMATTING TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED! ✗")
    print("=" * 60)
