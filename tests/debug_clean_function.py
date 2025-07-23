#!/usr/bin/env python3
"""Test simplificado para debug de la función clean_response_message"""

import sys
import os

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_unicode_fix():
    """Test del método de corrección Unicode"""
    
    # Texto problemático
    problematic = "Â¡Hola! Soy tu asistente de IA mÃ©dica. Â¿En quÃ© puedo ayudarte?"
    expected = "¡Hola! Soy tu asistente de IA médica. ¿En qué puedo ayudarte?"
    
    print(f"🔍 Texto problemático: {repr(problematic)}")
    print(f"🎯 Texto esperado: {repr(expected)}")
    
    # Método principal: latin-1 -> utf-8
    try:
        corrected = problematic.encode('latin-1').decode('utf-8')
        print(f"✅ Corrección exitosa: {repr(corrected)}")
        print(f"✅ Coincide: {corrected == expected}")
        return corrected
    except Exception as e:
        print(f"❌ Corrección falló: {e}")
        return problematic

def clean_response_message_fixed(message: str) -> str:
    """Función de limpieza corregida"""
    import json
    import re
    
    # Si el mensaje es JSON crudo, extraer solo la parte del mensaje
    if message.strip().startswith('{"intention"'):
        try:
            parsed = json.loads(message)
            if 'message' in parsed:
                message = parsed['message']
        except:
            pass
    
    # CORREGIR ENCODING - Método principal
    try:
        # El problema es que el texto está en UTF-8 pero interpretado como latin-1
        # La solución es re-encodear como latin-1 y decodear como UTF-8
        corrected = message.encode('latin-1').decode('utf-8')
        message = corrected
        print("🔧 Corrección de encoding aplicada")
    except Exception as e:
        print(f"⚠️ Corrección automática falló: {e}")
        # Fallback: reemplazos manuales
        manual_fixes = [
            ('Â¡', '¡'),
            ('Â¿', '¿'),
            ('Ã¡', 'á'),
            ('Ã©', 'é'),
            ('Ã­', 'í'),
            ('Ã³', 'ó'),
            ('Ãº', 'ú'),
            ('Ã±', 'ñ'),
            ('mÃ©dica', 'médica'),
            ('prÃ¡ctica', 'práctica'),
            ('informaciÃ³n', 'información')
        ]
        
        for wrong, correct in manual_fixes:
            message = message.replace(wrong, correct)
    
    # Limpiar secuencias Unicode escapadas como \u00a1
    try:
        unicode_fixes = [
            (r'\\u00a1', '¡'),
            (r'\\u00bf', '¿'),
            (r'\\u00e1', 'á'),
            (r'\\u00e9', 'é'),
            (r'\\u00ed', 'í'),
            (r'\\u00f3', 'ó'),
            (r'\\u00fa', 'ú'),
            (r'\\u00f1', 'ñ')
        ]
        
        for pattern, replacement in unicode_fixes:
            message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
            
        # Intentar unicode_escape como último recurso
        message = message.encode().decode('unicode_escape')
        
    except Exception as e:
        print(f"⚠️ Error en unicode_escape: {e}")
        pass
    
    return message.strip()

def run_tests():
    """Ejecutar todos los tests"""
    
    test_cases = [
        {
            'name': 'JSON con Unicode mal codificado',
            'input': '{"intention": "conversacion", "agent": "coordinator", "is_medical_query": false, "parameters": {}, "message": "Â¡Hola! Soy tu asistente de IA mÃ©dica. Â¿En quÃ© puedo ayudarte?"}',
            'expected': '¡Hola! Soy tu asistente de IA médica. ¿En qué puedo ayudarte?'
        },
        {
            'name': 'String directo mal codificado',
            'input': 'Â¡Hola! Soy tu asistente de IA mÃ©dica. Â¿En quÃ© puedo ayudarte?',
            'expected': '¡Hola! Soy tu asistente de IA médica. ¿En qué puedo ayudarte?'
        },
        {
            'name': 'Texto médico con acentos',
            'input': 'Los sÃ­ntomas comunes incluyen informaciÃ³n mÃ©dica prÃ¡ctica.',
            'expected': 'Los síntomas comunes incluyen información médica práctica.'
        }
    ]
    
    print("🧪 EJECUTANDO TESTS CON FUNCIÓN CORREGIDA\n")
    
    all_passed = True
    for i, case in enumerate(test_cases, 1):
        result = clean_response_message_fixed(case['input'])
        passed = result == case['expected']
        all_passed = all_passed and passed
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} Test {i}: {case['name']}")
        print(f"   Input: {case['input'][:60]}...")
        print(f"   Result: {result}")
        print(f"   Expected: {case['expected']}")
        
        if not passed:
            print(f"   ❌ DIFERENCIAS:")
            print(f"      Resultado: {repr(result)}")
            print(f"      Esperado:  {repr(case['expected'])}")
        print()
    
    return all_passed

if __name__ == "__main__":
    print("🛠️ DEBUG: FUNCIÓN clean_response_message\n")
    
    # Test básico de corrección
    test_unicode_fix()
    print()
    
    # Test completo
    success = run_tests()
    
    if success:
        print("✅ ¡TODOS LOS TESTS PASARON! La función está lista.")
    else:
        print("❌ Algunos tests fallaron. Revisa la implementación.")
