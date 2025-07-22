"""
Fast Medical Orchestrator - Optimizado para respuestas rápidas sin timeout
Separa las respuestas conversacionales directas de los workflows de agentes complejos.
"""

import time
import datetime
import os
import re
from typing import Dict, Any, Optional
import asyncio
import logging

# Importar el LLM configurado
try:
    from ..config.llm_config import unified_llm_config
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)

class FastMedicalOrchestrator:
    """
    Orquestador optimizado que evita timeouts separando:
    - Respuestas directas del LLM (rápidas)
    - Workflows de agentes (para tareas específicas)
    """
    
    def __init__(self, agents: Dict[str, Any] = None):
        self.agents = agents or {}
        print(f"🚀 [{datetime.datetime.now().strftime('%H:%M:%S')}] FastMedicalOrchestrator inicializado")
        
        # Debug de agentes disponibles
        if self.agents:
            agent_names = list(self.agents.keys())
            print(f"🔧 [DEBUG] Agentes disponibles en FastOrchestrator: {agent_names}")
        else:
            print(f"⚠️ [DEBUG] No hay agentes disponibles en FastOrchestrator")
        
    def _detect_intention_fast(self, user_input: str, context: dict) -> str:
        """
        Detector rápido de intención SIN llamar a ningún LLM.
        Clasifica el input en rutas específicas.
        """
        user_lower = user_input.lower()
        
        # 🎯 COMANDOS ESPECÍFICOS que requieren agentes
        analyze_keywords = ["analizar", "análisis", "analiza", "explora", "examina", "estudia"]
        if any(word in user_lower for word in analyze_keywords):
            return "analyzer_workflow"
        
        # ⚠️ EVALUATOR tiene prioridad sobre GENERATOR para términos de calidad
        # Primero verificar términos específicos de evaluación
        evaluate_keywords = ["evaluar", "evalúa", "evaluación", "utilidad", "métricas"]
        evaluate_phrases = ["calidad de datos", "calidad sintéticos", "métricas de utilidad"]
        
        if any(word in user_lower for word in evaluate_keywords):
            return "evaluator_workflow"
        if any(phrase in user_lower for phrase in evaluate_phrases):
            return "evaluator_workflow"
        # Si contiene "calidad" + términos de evaluación, va a evaluator
        if "calidad" in user_lower and any(term in user_lower for term in ["datos", "sintéticos", "utilidad", "métricas"]):
            return "evaluator_workflow"
            
        generate_keywords = ["generar", "sintético", "sintéticos", "genera", "crear", "ctgan", "tvae", "sdv"]
        if any(word in user_lower for word in generate_keywords):
            return "generator_workflow"
            
        validate_keywords = ["validar", "valida", "validación", "verificar", "verifica", "comprobar"]
        if any(word in user_lower for word in validate_keywords):
            return "validator_workflow"
            
        simulate_keywords = ["simular", "simula", "paciente", "evolución", "temporal"]
        if any(word in user_lower for word in simulate_keywords):
            return "simulator_workflow"
        
        # 💬 TODO LO DEMÁS va a respuesta directa
        return "direct_llm_response"
    
    def _get_direct_llm_response(self, user_input: str, context: dict = None) -> Dict[str, Any]:
        """
        Respuesta directa del LLM sin pasar por agentes.
        Optimizada para velocidad (2-5 segundos).
        """
        start_time = time.time()
        print(f"⚡ [{datetime.datetime.now().strftime('%H:%M:%S')}] Respuesta directa LLM para: {user_input[:50]}...")
        
        context = context or {}
        has_dataset = context.get("dataset_uploaded", False)
        
        try:
            # FORZAR GROQ TEMPORALMENTE - DEBUG
            use_groq_force = os.getenv('FORCE_GROQ', 'true').lower() == 'true'
            
            if use_groq_force and os.getenv('GROQ_API_KEY'):
                print(f"🚀 [DEBUG] Forzando uso de Groq para respuesta directa...")
                from ..config.llm_config import GroqProvider
                
                groq_provider = GroqProvider()
                if groq_provider.available:
                    llm = groq_provider.create_llm(temperature=0.1, max_tokens=200)
                    max_words = 80
                    provider_name = "groq"
                    print(f"✅ [DEBUG] Usando Groq con modelo: {groq_provider.model}")
                else:
                    print(f"❌ [DEBUG] Groq no disponible, usando configuración unificada")
                    use_groq_force = False
            
            if not use_groq_force and LLM_AVAILABLE and unified_llm_config:
                # Configuración optimizada para velocidad según el modelo
                current_model = os.getenv('OLLAMA_MODEL', 'llama3.2:1b')
                
                if 'mistral' in current_model.lower():
                    # Configuración ultra-rápida para Mistral
                    llm = unified_llm_config.create_llm(
                        temperature=0.2,
                        max_tokens=200,  # Respuestas más cortas
                        top_p=0.9,
                        repeat_penalty=1.1
                    )
                    max_words = 50  # Mistral es muy conciso
                elif 'llama3.2:1b' in current_model.lower():
                    # Configuración para Llama 3.2 (razonamiento)
                    llm = unified_llm_config.create_llm(
                        temperature=0.1,
                        max_tokens=150,
                        top_p=0.8
                    )
                    max_words = 80
                else:
                    # Configuración por defecto
                    llm = unified_llm_config.create_llm(
                        temperature=0.1,
                        max_tokens=1000
                    )
                    max_words = 100
                
                # Prompt optimizado para respuestas rápidas y concisas
                system_prompt = f"""Eres un asistente de IA médica. Responde de forma CONCISA y DIRECTA.

REGLAS ESTRICTAS:
- Máximo {max_words} palabras por respuesta
- Respuestas directas sin explicaciones largas
- Si es saludo: responde amigablemente
- Si es pregunta médica: información básica + sugerencia de subir dataset
- NO analices datos ni generes contenido (eso requiere comandos específicos)

CAPACIDADES DISPONIBLES:
- Análisis: "analizar datos"
- Generación sintética: "generar sintéticos"  
- Validación: "validar datos"

Responde SOLO lo esencial."""

                # Añadir contexto del dataset si existe
                user_prompt = user_input
                if has_dataset:
                    filename = context.get("filename", "dataset")
                    rows = context.get("rows", 0)
                    cols = context.get("columns", 0)
                    user_prompt += f"\n\nCONTEXTO: Tengo cargado el dataset '{filename}' con {rows:,} filas y {cols} columnas."

                # Generar respuesta usando el método correcto según el proveedor
                if unified_llm_config.active_provider == "azure":
                    # Para Azure OpenAI, usar método chat
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    response = llm.invoke(messages)
                    content = response.content.strip()
                elif unified_llm_config.active_provider == "ollama":
                    # Para Ollama, usar prompt simple
                    full_prompt = f"{system_prompt}\n\nUsuario: {user_prompt}\nAsistente:"
                    response = llm.invoke(full_prompt)
                    content = response.strip()
                elif unified_llm_config.active_provider == "grok":
                    # Para Grok, usar método directo
                    full_prompt = f"{system_prompt}\n\nUsuario: {user_prompt}\nAsistente:"
                    content = llm.invoke(full_prompt)
                else:
                    # Fallback genérico
                    full_prompt = f"{system_prompt}\n\nUsuario: {user_prompt}\nAsistente:"
                    content = llm.invoke(full_prompt)
                
                end_time = time.time()
                print(f"✅ [{datetime.datetime.now().strftime('%H:%M:%S')}] LLM respondió en {end_time - start_time:.2f}s")
                
                # Filtrar etiquetas <think> y otros elementos no deseados
                clean_content = self._clean_llm_response(content)
                
                return {
                    "message": clean_content,
                    "agent": "llm_direct",
                    "response_time": f"{end_time - start_time:.2f}s",
                    "route": "direct"
                }
            else:
                # Fallback a respuestas predefinidas si no hay LLM
                return self._get_fallback_response(user_input, context)
                
        except Exception as e:
            print(f"❌ Error en LLM directo: {e}")
            # Fallback a respuestas predefinidas
            return self._get_fallback_response(user_input, context)
    
    def _get_fallback_response(self, user_input: str, context: dict = None) -> Dict[str, Any]:
        """
        Respuestas fallback inteligentes sin LLM.
        Basadas en patrones de reconocimiento.
        """
        user_lower = user_input.lower()
        context = context or {}
        has_dataset = context.get("dataset_uploaded", False)
        
        # Patrones de respuesta médica
        if any(term in user_lower for term in ["diabetes", "glucosa", "insulina"]):
            return {
                "message": "🩺 **Información sobre Diabetes**\n\nLa diabetes es una enfermedad crónica que afecta la forma en que el cuerpo procesa la glucosa. Los principales factores de riesgo incluyen:\n\n• **Tipo 1**: Factores genéticos e inmunológicos\n• **Tipo 2**: Sobrepeso, sedentarismo, historia familiar\n• **Gestacional**: Cambios hormonales durante el embarazo\n\n**Recomendación**: Para análisis detallado de datos diabéticos, sube un dataset y solicita un análisis específico.\n\n*Nota: Esta es información general. Consulta siempre con un profesional médico.*",
                "agent": "fallback_medical",
                "route": "fallback"
            }
            
        elif any(term in user_lower for term in ["covid", "coronavirus", "sars-cov"]):
            return {
                "message": "🦠 **Información sobre COVID-19**\n\nFactores de riesgo identificados en datos clínicos:\n\n• **Edad**: Pacientes > 65 años\n• **Comorbilidades**: Diabetes, hipertensión, EPOC\n• **Estado inmunológico**: Inmunosupresión\n• **Factores cardiovasculares**: Enfermedad cardíaca previa\n\n**Nuestro sistema** puede analizar datasets COVID-19 y generar datos sintéticos que preserven estos patrones epidemiológicos.\n\n*Datos basados en estudios internacionales publicados.*",
                "agent": "fallback_medical",
                "route": "fallback"
            }
            
        elif any(term in user_lower for term in ["hipertensión", "presión", "cardiovascular"]):
            return {
                "message": "❤️ **Factores de Riesgo Cardiovascular**\n\nPrincipales factores identificados en estudios clínicos:\n\n• **Modificables**: Tabaquismo, colesterol alto, sedentarismo\n• **No modificables**: Edad, sexo, historia familiar\n• **Metabólicos**: Diabetes, obesidad, síndrome metabólico\n• **Otros**: Estrés, apnea del sueño, enfermedad renal\n\n**¿Tienes datos cardiovasculares?** Puedo ayudarte a analizarlos y generar datasets sintéticos para investigación.\n\n*Información basada en guías clínicas internacionales.*",
                "agent": "fallback_medical",
                "route": "fallback"
            }
            
        elif any(term in user_lower for term in ["hola", "saludo", "buenos días", "buenas tardes", "como estas"]):
            dataset_msg = ""
            if has_dataset:
                filename = context.get("filename", "archivo")
                rows = context.get("rows", 0)
                dataset_msg = f"\n\n📊 **Dataset activo**: {filename} ({rows:,} registros)"
            
            return {
                "message": f"👋 **¡Hola!** Estoy muy bien, gracias por preguntar.\n\nSoy tu asistente de IA especializado en datos clínicos sintéticos.{dataset_msg}\n\n**¿En qué puedo ayudarte?**\n• Analizar datasets médicos\n• Generar datos sintéticos seguros\n• Responder preguntas sobre medicina\n• Validar coherencia clínica\n\n¡Pregúntame cualquier cosa sobre medicina o datos clínicos!",
                "agent": "fallback_greeting",
                "route": "fallback"
            }
            
        elif any(term in user_lower for term in ["ayuda", "help", "qué puedes hacer"]):
            return {
                "message": "📋 **Guía de Uso - Patient IA**\n\n**🤖 Comandos principales:**\n• `Analiza estos datos` - Explora patrones en tu dataset\n• `Genera 1000 muestras con CTGAN` - Crea datos sintéticos\n• `Valida la coherencia médica` - Verifica calidad clínica\n\n**🩺 Consultas médicas:**\n• Factores de riesgo cardiovascular\n• Información sobre diabetes, COVID-19\n• Análisis epidemiológico\n• Interpretación de biomarcadores\n\n**📊 Tipos de datos soportados:**\n• CSV, Excel (.xlsx, .xls)\n• Historiales clínicos\n• Datos de laboratorio\n• Registros epidemiológicos\n\n¿Hay algo específico en lo que te pueda ayudar?",
                "agent": "fallback_help",
                "route": "fallback"
            }
        else:
            # Respuesta general
            return {
                "message": f"🤔 **Consulta recibida**\n\nHe recibido tu consulta: *\"{user_input}\"*\n\n📚 Como asistente de IA médica, puedo ayudarte con:\n• Análisis de datasets clínicos\n• Información sobre enfermedades comunes\n• Interpretación de factores de riesgo\n• Generación de datos sintéticos\n\n**Para tareas específicas**, usa comandos como:\n• \"Analizar datos\" - Para explorar tu dataset\n• \"Generar sintéticos\" - Para crear datos sintéticos\n• \"Validar datos\" - Para verificar coherencia\n\n*Recuerda: Esta información es para fines educativos. Consulta siempre con profesionales médicos.*",
                "agent": "fallback_general",
                "route": "fallback"
            }
    
    def _clean_llm_response(self, content: str) -> str:
        """
        Limpia la respuesta del LLM eliminando etiquetas de razonamiento y formato no deseado.
        """
        if not content:
            return content
        
        # Eliminar etiquetas <think> completas
        content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Eliminar etiquetas <thinking> si existen
        content = re.sub(r'<thinking>.*?</thinking>\s*', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Eliminar etiquetas de razonamiento que no se cerraron
        content = re.sub(r'<think>.*$', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<thinking>.*$', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Limpiar múltiples espacios y saltos de línea
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'^\s+|\s+$', '', content)
        
        # Si la respuesta está vacía después de limpiar, devolver mensaje por defecto
        if not content.strip():
            content = "Lo siento, no pude generar una respuesta clara. ¿Puedes reformular tu pregunta?"
        
        return content

    async def _execute_agent_workflow(self, workflow_type: str, user_input: str, context: dict) -> Dict[str, Any]:
        """
        Ejecuta workflows específicos de agentes para tareas complejas.
        """
        start_time = time.time()
        print(f"🔧 [{datetime.datetime.now().strftime('%H:%M:%S')}] Ejecutando workflow: {workflow_type}")
        
        try:
            if workflow_type == "analyzer_workflow":
                print(f"🔍 [DEBUG] Buscando agente 'analyzer' en: {list(self.agents.keys())}")
                if "analyzer" in self.agents:
                    print(f"✅ [DEBUG] Agente analyzer encontrado: {type(self.agents['analyzer'])}")
                    
                    # Verificar si hay dataset en contexto
                    if context and context.get("dataframe") is not None:
                        print(f"📊 [DEBUG] Dataset encontrado para análisis")
                        
                        # Ejecutar análisis universal primero
                        try:
                            from ..adapters.universal_dataset_detector import UniversalDatasetDetector
                            detector = UniversalDatasetDetector()
                            df = context["dataframe"]
                            
                            print(f"🔬 [DEBUG] Ejecutando análisis universal...")
                            universal_analysis = detector.analyze_dataset(df)
                            
                            # Agregar el análisis universal al contexto
                            context["universal_analysis"] = universal_analysis
                            print(f"✅ [DEBUG] Análisis universal completado")
                            
                        except Exception as e:
                            print(f"❌ [DEBUG] Error en análisis universal: {e}")
                            return {
                                "message": f"❌ **Error en análisis universal**: {str(e)}\n\nNo se pudo completar el análisis automático del dataset.",
                                "agent": "system",
                                "route": "agent_error"
                            }
                    
                    # Usar el método process() estándar como los otros agentes
                    response = await self.agents["analyzer"].process(user_input, context)
                    response["route"] = "agent_workflow"
                    return response
                else:
                    print(f"❌ [DEBUG] Agente analyzer NO encontrado")
                    return {
                        "message": "📊 **Análisis de datos solicitado**\n\nPara realizar un análisis completo, necesito que el agente analizador esté disponible. Actualmente estoy en modo simplificado.\n\n**Capacidades de análisis:**\n• Estadísticas descriptivas\n• Detección de patrones médicos\n• Identificación de outliers\n• Correlaciones clínicas\n\nPor favor, verifica la configuración de los agentes.",
                        "agent": "system",
                        "route": "agent_fallback"
                    }
                    
            elif workflow_type == "generator_workflow":
                if "generator" in self.agents:
                    # Generar datos sintéticos
                    response = await self.agents["generator"].process(user_input, context)
                    
                    # Post-procesar datos sintéticos para validación JSON
                    if "synthetic_data" in response:
                        response = self._post_process_synthetic_data(response, context)
                    
                    response["route"] = "agent_workflow"
                    return response
                else:
                    return {
                        "message": "🏭 **Generación de datos sintéticos solicitada**\n\nPara generar datos sintéticos, necesito que el agente generador esté disponible. Actualmente estoy en modo simplificado.\n\n**Modelos disponibles:**\n• CTGAN - Para datos mixtos\n• TVAE - Para preservar distribuciones\n• SDV - Para máxima calidad\n\nPor favor, verifica la configuración de los agentes.",
                        "agent": "system",
                        "route": "agent_fallback"
                    }
                    
            elif workflow_type == "validator_workflow":
                if "validator" in self.agents:
                    response = await self.agents["validator"].process(user_input, context)
                    response["route"] = "agent_workflow"
                    return response
                else:
                    return {
                        "message": "🔍 **Validación de datos solicitada**\n\nPara validar la coherencia clínica, necesito que el agente validador esté disponible. Actualmente estoy en modo simplificado.\n\n**Validaciones disponibles:**\n• Coherencia médica\n• Rangos de valores\n• Patrones clínicos\n• Calidad de datos\n\nPor favor, verifica la configuración de los agentes.",
                        "agent": "system",
                        "route": "agent_fallback"
                    }
                    
            elif workflow_type == "evaluator_workflow":
                print(f"📊 [DEBUG] Buscando agente 'evaluator' en: {list(self.agents.keys())}")
                if "evaluator" in self.agents:
                    print(f"✅ [DEBUG] Agente evaluator encontrado: {type(self.agents['evaluator'])}")
                    
                    # Verificar que hay datos originales y sintéticos para evaluar
                    original_data = context.get("dataframe")
                    synthetic_data = context.get("synthetic_data")
                    
                    if original_data is None:
                        return {
                            "message": "❌ **Error**: No hay dataset original cargado.\n\nPara evaluar calidad, necesito:\n• Dataset original cargado\n• Datos sintéticos generados\n\n**Pasos sugeridos:**\n1. Sube un dataset\n2. Genera datos sintéticos\n3. Solicita evaluación",
                            "agent": "system",
                            "route": "evaluation_error"
                        }
                    
                    if synthetic_data is None:
                        return {
                            "message": "❌ **Error**: No hay datos sintéticos generados.\n\nPara evaluar calidad, necesito datos sintéticos previos.\n\n**Pasos sugeridos:**\n1. Genera datos sintéticos primero\n2. Luego solicita evaluación\n\n**Comando ejemplo:**\n`Genera 500 registros con CTGAN`",
                            "agent": "system",
                            "route": "evaluation_error"
                        }
                    
                    print(f"📊 [DEBUG] Datos disponibles para evaluación:")
                    print(f"   - Original: {len(original_data)} filas")
                    print(f"   - Sintético: {len(synthetic_data)} filas")
                    
                    response = await self.agents["evaluator"].process(user_input, context)
                    response["route"] = "agent_workflow"
                    return response
                else:
                    print(f"❌ [DEBUG] Agente evaluator NO encontrado")
                    return {
                        "message": "📊 **Evaluación de calidad solicitada**\n\nPara evaluar la utilidad de datos sintéticos, necesito que el agente evaluador esté disponible. Actualmente estoy en modo simplificado.\n\n**Métricas de evaluación:**\n• Fidelidad estadística\n• Utilidad para ML\n• Preservación de patrones\n• Score de privacidad\n• Calidad de entidades médicas\n\nPor favor, verifica la configuración de los agentes.",
                        "agent": "system",
                        "route": "agent_fallback"
                    }
                    
            elif workflow_type == "simulator_workflow":
                print(f"🏥 [DEBUG] Buscando agente 'simulator' en: {list(self.agents.keys())}")
                if "simulator" in self.agents:
                    print(f"✅ [DEBUG] Agente simulator encontrado: {type(self.agents['simulator'])}")
                    
                    # Verificar que hay datos validados para la simulación
                    synthetic_data = context.get("synthetic_data")
                    original_data = context.get("dataframe")
                    
                    # Priorizar datos sintéticos validados, pero permitir simulación con datos originales
                    simulation_data = synthetic_data if synthetic_data is not None else original_data
                    
                    if simulation_data is None:
                        return {
                            "message": "❌ **Error**: No hay datos disponibles para simulación.\n\nPara simular evolución de pacientes, necesito:\n• Dataset cargado (original o sintético)\n• Preferiblemente datos sintéticos validados\n\n**Pasos sugeridos:**\n1. Sube un dataset\n2. Opcionalmente genera datos sintéticos\n3. Valida los datos\n4. Solicita simulación\n\n**Comando ejemplo:**\n`Simula la evolución de estos pacientes`",
                            "agent": "system",
                            "route": "simulation_error"
                        }
                    
                    print(f"🏥 [DEBUG] Datos disponibles para simulación:")
                    print(f"   - Datos de simulación: {len(simulation_data)} filas")
                    print(f"   - Tipo de datos: {'sintéticos' if synthetic_data is not None else 'originales'}")
                    
                    # Actualizar contexto con los datos de simulación
                    context["synthetic_data"] = simulation_data
                    
                    response = await self.agents["simulator"].process(user_input, context)
                    response["route"] = "agent_workflow"
                    return response
                else:
                    print(f"❌ [DEBUG] Agente simulator NO encontrado")
                    return {
                        "message": "🏥 **Simulación de pacientes solicitada**\n\nPara simular la evolución temporal de pacientes, necesito que el agente simulador esté disponible. Actualmente estoy en modo simplificado.\n\n**Simulaciones disponibles:**\n• Evolución hospitalaria\n• Progresión clínica temporal\n• Simulación de visitas\n• Cambios en biomarcadores\n• Respuesta a tratamientos\n\n**Enfermedades soportadas:**\n• COVID-19 (simulación específica)\n• Casos generales (todas las especialidades)\n\nPor favor, verifica la configuración de los agentes.",
                        "agent": "system",
                        "route": "agent_fallback"
                    }
            else:
                return {
                    "message": f"❌ **Workflow no reconocido**: {workflow_type}\n\nWorkflows disponibles:\n• analyzer_workflow\n• generator_workflow\n• validator_workflow\n• evaluator_workflow\n• simulator_workflow",
                    "agent": "system",
                    "route": "error"
                }
                
        except Exception as e:
            end_time = time.time()
            print(f"❌ [{datetime.datetime.now().strftime('%H:%M:%S')}] Error en workflow {workflow_type} después de {end_time - start_time:.2f}s: {e}")
            return {
                "message": f"❌ **Error en {workflow_type}**: {str(e)}\n\nPor favor, intenta de nuevo o usa un comando más simple.",
                "agent": "system",
                "route": "error",
                "error": True
            }
    
    async def process_user_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Punto de entrada principal. Decide entre respuesta directa o workflow de agentes.
        """
        start_time = time.time()
        print(f"🎯 [{datetime.datetime.now().strftime('%H:%M:%S')}] FastOrchestrator procesando: {user_input[:50]}...")
        
        context = context or {}
        
        # 🔍 PASO 1: Detectar intención (SIN LLM, muy rápido)
        intention = self._detect_intention_fast(user_input, context)
        print(f"🎯 Intención detectada: {intention}")
        
        # 🚀 PASO 2: Ejecutar ruta correspondiente
        if intention == "direct_llm_response":
            # RUTA RÁPIDA: Respuesta directa del LLM
            response = self._get_direct_llm_response(user_input, context)
        else:
            # RUTA DE AGENTES: Workflow específico
            response = await self._execute_agent_workflow(intention, user_input, context)
        
        end_time = time.time()
        total_time = end_time - start_time
        response["total_time"] = f"{total_time:.2f}s"
        
        print(f"✅ [{datetime.datetime.now().strftime('%H:%M:%S')}] FastOrchestrator completado en {total_time:.2f}s")
        return response
    
    def process_user_input_sync(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Versión síncrona que usa el wrapper async-safe"""
        try:
            from ..utils.streamlit_async_wrapper import run_async_safe
            return run_async_safe(self.process_user_input, user_input, context)
        except ImportError:
            # Fallback simple si no hay wrapper
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.process_user_input(user_input, context))
                finally:
                    loop.close()
            except Exception as e:
                return {
                    "message": f"❌ Error ejecutando de forma síncrona: {str(e)}",
                    "agent": "system",
                    "error": True
                }
    
    def _post_process_synthetic_data(self, response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-procesa los datos sintéticos para prepararlos para validación JSON.
        Aplica el medical_data_adapter para convertir al formato esperado.
        """
        try:
            synthetic_data = response.get("synthetic_data")
            if synthetic_data is None:
                return response
            
            print("🔄 Post-procesando datos sintéticos para validación JSON...")
            
            # Importar el adaptador médico
            from ..adapters.medical_data_adapter import UniversalMedicalAdapter
            adapter = UniversalMedicalAdapter()
            
            # Obtener mapeos de campo del contexto si están disponibles
            analysis_results = context.get("analysis_results", {})
            field_mappings = analysis_results.get("field_mappings", {})
            
            # Preparar datos para validación JSON
            json_ready_data = adapter.prepare_for_json_validation(
                synthetic_data, 
                field_mappings
            )
            
            # Actualizar la respuesta con los datos procesados
            response["synthetic_data"] = json_ready_data
            
            # Agregar información sobre el post-procesamiento
            generation_info = response.get("generation_info", {})
            generation_info["json_processed"] = True
            generation_info["json_processing_timestamp"] = datetime.datetime.now().isoformat()
            response["generation_info"] = generation_info
            
            # Actualizar el mensaje para indicar el post-procesamiento
            if "message" in response:
                response["message"] += "\n\n🔧 **Post-procesamiento aplicado:** Datos preparados para validación JSON esquema médico."
            
            print("✅ Post-procesamiento completado - datos listos para validación JSON")
            return response
            
        except Exception as e:
            print(f"⚠️ Error en post-procesamiento de datos sintéticos: {e}")
            # Si falla el post-procesamiento, devolver los datos originales
            return response
