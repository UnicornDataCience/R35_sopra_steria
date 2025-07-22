#!/usr/bin/env python
"""
Test rápido de la función clean_response_message mejorada
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Simular los imports necesarios
import json
import re

def clean_response_message(message: str) -> str:
    """
    Limpia el mensaje de respuesta de caracteres Unicode escapados, JSON crudo y problemas de encoding.
    Diseñado específicamente para el problema de doble encoding UTF-8/Latin-1 del modelo LLM.
    """
    
    # MÉTODO 0: Detectar y extraer JSON crudo más robustamente
    message_stripped = message.strip()
    
    # Casos de JSON directo
    if message_stripped.startswith('{"intention"') or message_stripped.startswith('{"message"'):
        try:
            parsed = json.loads(message_stripped)
            if 'message' in parsed:
                message = parsed['message']
                print("✅ [CLEAN] JSON crudo extraído exitosamente")
            elif 'content' in parsed:
                message = parsed['content']
                print("✅ [CLEAN] Contenido JSON extraído")
        except json.JSONDecodeError as e:
            print(f"⚠️ [CLEAN] Error parseando JSON crudo: {e}")
            # Si no se puede parsear, mantener el mensaje original
            pass
    
    # Casos de JSON anidado o mal formateado
    elif '"message"' in message_stripped and '"intention"' in message_stripped:
        try:
            # Buscar el patrón JSON dentro del texto
            json_pattern = r'\{"intention".*?"message"[^}]*\}'
            match = re.search(json_pattern, message_stripped, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if 'message' in parsed:
                    message = parsed['message']
                    print("✅ [CLEAN] JSON anidado extraído")
        except:
            pass
    
    try:
        # MÉTODO 1: Detección y corrección más agresiva del doble encoding UTF-8/Latin-1
        # Este es el problema principal con el modelo llama-4-scout
        
        # Lista de indicadores de corrupción más completa
        corrupted_patterns = [
            'Â¡', 'Â¿', 'Ã¡', 'Ã©', 'Ã­', 'Ã³', 'Ãº', 'Ã±', 'Ã‰', 'Ã"', 'Ãš', 'ÃÑ',
            'mÃ©dica', 'mÃ©dico', 'anÃ¡lisis', 'sÃ­ntomas', 'diagnÃ³stico', 'informaciÃ³n',
            'especializaciÃ³n', 'atenciÃ³n', 'evaluaciÃ³n', 'investigaciÃ³n', 'aquÃ­',
            'tambiÃ©n', 'prÃ¡ctica', 'clÃ­nicos', 'pÃ¡gina', 'dÃ­a', 'fÃ¡cil', 'difÃ­cil',
            'quÃ©', 'cÃ³mo', 'dÃ³nde', 'cuÃ¡ndo'
        ]
        
        if any(pattern in message for pattern in corrupted_patterns):
            print("🔍 [CLEAN] Detectada corrupción de encoding, aplicando corrección")
            
            # ESTRATEGIA MÚLTIPLE DE CORRECCIÓN
            original_message = message
            attempts = []
            
            # Intento 1: Corrección automática estándar
            try:
                corrected1 = message.encode('latin-1').decode('utf-8')
                attempts.append(('auto_standard', corrected1))
            except:
                pass
            
            # Intento 2: Corrección con manejo de errores 'ignore'
            try:
                corrected2 = message.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')
                attempts.append(('auto_ignore', corrected2))
            except:
                pass
            
            # Intento 3: Corrección manual con diccionario más completo
            corrected3 = message
            corruption_map = {
                # Signos de puntuación
                'Â¡': '¡', 'Â¿': '¿',
                # Vocales con tilde minúsculas
                'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
                # Eñe
                'Ã±': 'ñ', 'ÃÑ': 'Ñ',
                # Palabras completas muy comunes (más directas)
                'mÃ©dica': 'médica', 'mÃ©dico': 'médico', 'mÃ©dicos': 'médicos',
                'anÃ¡lisis': 'análisis', 'sÃ­ntomas': 'síntomas', 'sÃ­ntoma': 'síntoma',
                'diagnÃ³stico': 'diagnóstico', 'diagnÃ³sticos': 'diagnósticos',
                'informaciÃ³n': 'información', 'atenciÃ³n': 'atención',
                'especializaciÃ³n': 'especialización', 'evaluaciÃ³n': 'evaluación',
                'investigaciÃ³n': 'investigación', 'prÃ¡ctica': 'práctica',
                'clÃ­nicos': 'clínicos', 'clÃ­nica': 'clínica', 'clÃ­nico': 'clínico',
                'pÃ¡gina': 'página', 'pÃ¡ginas': 'páginas',
                'dÃ­a': 'día', 'dÃ­as': 'días',
                'aquÃ­': 'aquí', 'allÃ­': 'allí', 'tambiÃ©n': 'también',
                'fÃ¡cil': 'fácil', 'difÃ­cil': 'difícil',
                # Interrogativos
                'quÃ©': 'qué', 'cÃ³mo': 'cómo', 'dÃ³nde': 'dónde', 'cuÃ¡ndo': 'cuándo',
                'quiÃ©n': 'quién', 'cuÃ¡l': 'cuál', 'cuÃ¡nto': 'cuánto'
            }
            
            for corrupt, correct in corruption_map.items():
                if corrupt in corrected3:
                    corrected3 = corrected3.replace(corrupt, correct)
            
            attempts.append(('manual_dict', corrected3))
            
            # Evaluar cuál corrección es mejor
            def count_spanish_chars(text):
                spanish_chars = ['á', 'é', 'í', 'ó', 'ú', 'ñ', '¡', '¿', 'Á', 'É', 'Í', 'Ó', 'Ú', 'Ñ']
                return sum(1 for char in text if char in spanish_chars)
            
            def count_corrupted_chars(text):
                corrupted = ['Ã', 'Â']
                return sum(1 for char in text if char in corrupted)
            
            # Seleccionar la mejor corrección
            best_attempt = None
            best_score = -1
            
            for method, corrected in attempts:
                spanish_count = count_spanish_chars(corrected)
                corrupted_count = count_corrupted_chars(corrected)
                score = spanish_count - (corrupted_count * 2)  # Penalizar caracteres corruptos
                
                if score > best_score:
                    best_score = score
                    best_attempt = (method, corrected)
            
            if best_attempt:
                message = best_attempt[1]
                print(f"✅ [CLEAN] Mejor corrección: {best_attempt[0]} (score: {best_score})")
        
        # MÉTODO 2: Limpiar secuencias Unicode escapadas (ya funciona bien)
        unicode_patterns = [
            (r'\\u00a1', '¡'), (r'\\u00bf', '¿'),
            (r'\\u00e1', 'á'), (r'\\u00e9', 'é'), (r'\\u00ed', 'í'), 
            (r'\\u00f3', 'ó'), (r'\\u00fa', 'ú'), (r'\\u00f1', 'ñ'),
            (r'\\u00c1', 'Á'), (r'\\u00c9', 'É'), (r'\\u00cd', 'Í'), 
            (r'\\u00d3', 'Ó'), (r'\\u00da', 'Ú'), (r'\\u00d1', 'Ñ')
        ]
        
        unicode_replaced = False
        for pattern, replacement in unicode_patterns:
            if re.search(pattern, message, flags=re.IGNORECASE):
                message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
                unicode_replaced = True
        
        if unicode_replaced:
            print("✅ [CLEAN] Secuencias Unicode escapadas convertidas")
        
        # MÉTODO 3: Limpieza final conservadora
        try:
            decoded = message.encode().decode('unicode_escape')
            if decoded != message and not any(char in decoded for char in ['Ã', 'Â']):
                message = decoded
                print("✅ [CLEAN] Unicode escape final aplicado")
        except:
            pass
            
    except Exception as e:
        print(f"⚠️ [CLEAN] Error en limpieza de mensaje: {e}")
    
    return message.strip()

def test_json_cleaning():
    """Test específico del JSON que apareció en tu chat"""
    
    print("🧪 TESTING JSON CRUDO DEL CHAT")
    print("=" * 60)
    
    # El JSON exacto que apareció
    json_raw = '{"intention": "conversacion", "agent": "coordinator", "is_medical_query": false, "parameters": {}, "message": "¡Hola! Soy tu asistente de IA médica. ¿En qué puedo ayudarte?"}'
    
    print(f"📝 Input: {json_raw}")
    
    result = clean_response_message(json_raw)
    
    print(f"📤 Output: {result}")
    
    # Verificar resultado
    expected = "¡Hola! Soy tu asistente de IA médica. ¿En qué puedo ayudarte?"
    
    if result == expected:
        print("✅ SUCCESS: JSON correctamente extraído y limpiado")
    else:
        print("❌ FAIL: El resultado no coincide con lo esperado")
        print(f"   Esperado: {expected}")
        print(f"   Obtenido: {result}")

if __name__ == "__main__":
    test_json_cleaning()
