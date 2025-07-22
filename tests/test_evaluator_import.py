#!/usr/bin/env python3
"""
Test para verificar la importación del agente evaluador
"""
import sys
sys.path.append('.')

try:
    from src.agents.evaluator_agent import UtilityEvaluatorAgent
    agent = UtilityEvaluatorAgent()
    print('✅ Evaluator agent creado correctamente:', type(agent).__name__)
    print('📋 Métodos disponibles:', [method for method in dir(agent) if not method.startswith('_')])
except Exception as e:
    print('❌ Error:', str(e))
    import traceback
    traceback.print_exc()
