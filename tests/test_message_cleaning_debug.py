#!/usr/bin/env python3
"""Test detallado para debug de la función clean_response_message"""

import sys
import os

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_unicode_issues():
    """Test específico para problemas de Unicode"""
    
    # Test 1: Detectar el problema
    problematic_text = "Â¡Hola! Soy tu asistente de IA mÃ©dica. Â¿En quÃ© puedo ayudarte?"
    expected_text = "¡Hola! Soy tu asistente de IA médica. ¿En qué puedo ayudarte?"
    
    print("🔍 DEBUGGING UNICODE ENCODING")
    print(f"Texto problemático: {repr(problematic_text)}")
    print(f"Texto esperado: {repr(expected_text)}")
    
    # Test diferentes enfoques de decodificación
    print("\n📝 Probando diferentes enfoques:")
    
    # Enfoque 1: decode('latin-1').encode('utf-8').decode('utf-8')
    try:
        result1 = problematic_text.encode('latin-1').decode('utf-8')
        print(f"Enfoque 1 (latin-1 -> utf-8): {repr(result1)}")
        print(f"Funciona: {result1 == expected_text}")
    except Exception as e:
        print(f"Enfoque 1 falló: {e}")
    
    # Enfoque 2: ftfy library (fix text for you)
    try:
        import ftfy
        result2 = ftfy.fix_text(problematic_text)
        print(f"Enfoque 2 (ftfy): {repr(result2)}")
        print(f"Funciona: {result2 == expected_text}")
    except ImportError:
        print("Enfoque 2: ftfy no disponible")
    except Exception as e:
        print(f"Enfoque 2 falló: {e}")
    
    # Enfoque 3: Replacement manual específico
    def manual_fix(text):
        replacements = {
            'Â¡': '¡',
            'Â¿': '¿',
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú',
            'Ã±': 'ñ',
            'Ã': 'Á',
            'Ã‰': 'É',
            'Ã': 'Í',
            'Ã"': 'Ó',
            'Ãš': 'Ú',
            'Ã'': 'Ñ',
            'mÃ©dica': 'médica',
            'prÃ¡ctica': 'práctica',
            'informaciÃ³n': 'información'
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        return text
    
    result3 = manual_fix(problematic_text)
    print(f"Enfoque 3 (manual): {repr(result3)}")
    print(f"Funciona: {result3 == expected_text}")
    
    return result3 == expected_text

def test_improved_clean_function():
    """Test con función mejorada"""
    
    def clean_response_message_improved(message: str) -> str:
        """
        Versión mejorada de la función de limpieza
        """
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
        
        # CORREGIR PROBLEMAS DE ENCODING ESPECÍFICOS
        try:
            # Método 1: Intentar latin-1 -> utf-8
            try:
                corrected = message.encode('latin-1').decode('utf-8')
                message = corrected
                print("✅ Corrección latin-1 -> utf-8 exitosa")
            except:
                print("⚠️ Corrección latin-1 -> utf-8 falló, usando manual")
                # Método 2: Reemplazos manuales específicos
                replacements = {
                    'Â¡': '¡',
                    'Â¿': '¿',
                    'Ã¡': 'á',
                    'Ã©': 'é',
                    'Ã­': 'í',
                    'Ã³': 'ó',
                    'Ãº': 'ú',
                    'Ã±': 'ñ',
                    'Ã': 'Á',
                    'Ã‰': 'É',
                    'Ã': 'Í',
                    'Ã"': 'Ó',
                    'Ãš': 'Ú',
                    'mÃ©dica': 'médica',
                    'prÃ¡ctica': 'práctica',
                    'informaciÃ³n': 'información'
                }
                
                for wrong, correct in replacements.items():
                    message = message.replace(wrong, correct)
            
            # Reemplazar secuencias Unicode escapadas como \u00a1
            unicode_replacements = {
                r'\\u00a1': '¡',
                r'\\u00bf': '¿', 
                r'\\u00e1': 'á',
                r'\\u00e9': 'é',
                r'\\u00ed': 'í',
                r'\\u00f3': 'ó',
                r'\\u00fa': 'ú',
                r'\\u00f1': 'ñ'
            }
            
            for pattern, replacement in unicode_replacements.items():
                message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
                
        except Exception as e:
            print(f"⚠️ Error en corrección de encoding: {e}")
            pass
        
        return message.strip()
    
    # Casos de test
    test_cases = [
        {
            'input': '{"intention": "conversacion", "agent": "coordinator", "is_medical_query": false, "parameters": {}, "message": "Â¡Hola! Soy tu asistente de IA mÃ©dica. Â¿En quÃ© puedo ayudarte?"}',
            'expected': '¡Hola! Soy tu asistente de IA médica. ¿En qué puedo ayudarte?'
        },
        {
            'input': 'Â¡Hola! Soy tu asistente de IA mÃ©dica. Â¿En quÃ© puedo ayudarte?',
            'expected': '¡Hola! Soy tu asistente de IA médica. ¿En qué puedo ayudarte?'
        },
        {
            'input': 'Los sÃ­ntomas comunes de la diabetes incluyen informaciÃ³n mÃ©dica prÃ¡ctica.',
            'expected': 'Los síntomas comunes de la diabetes incluyen información médica práctica.'
        }
    ]
    
    print("\n🧪 TESTING FUNCIÓN MEJORADA:")
    all_passed = True
    
    for i, case in enumerate(test_cases, 1):
        result = clean_response_message_improved(case['input'])
        passed = result == case['expected']
        all_passed = all_passed and passed
        
        status = "✅" if passed else "❌"
        print(f"{status} Test {i}: {passed}")
        print(f"   Input: {case['input'][:50]}...")
        print(f"   Result: {repr(result)}")
        print(f"   Expected: {repr(case['expected'])}")
        
        if not passed:
            print(f"   DIFERENCIA DETECTADA!")
        print()
    
    return all_passed

if __name__ == "__main__":
    print("🔧 DEBUGGING UNICODE CLEANING FUNCTION\n")
    
    # Test 1: Analizar el problema
    unicode_works = test_unicode_issues()
    print(f"\n📊 Unicode fix funciona: {unicode_works}")
    
    # Test 2: Probar función mejorada
    improved_works = test_improved_clean_function()
    print(f"\n📊 Función mejorada funciona: {improved_works}")
    
    if improved_works:
        print("\n✅ ¡FUNCIÓN CORREGIDA! Lista para implementar.")
    else:
        print("\n❌ Función aún necesita trabajo.")
