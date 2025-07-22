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
        Punto de entrada principal. Delega la detección de intención al Coordinador
        y luego enruta al workflow correspondiente.
        """
        start_time = time.time()
        print(f"🎯 [{datetime.datetime.now().strftime('%H:%M:%S')}] FastOrchestrator procesando: {user_input[:50]}...")
        
        context = context or {}
        
        # 🔍 PASO 1: Usar SIEMPRE el CoordinatorAgent para detectar la intención
        coordinator = self.agents.get("coordinator")
        if not coordinator:
            return {
                "message": "❌ Error crítico: Agente Coordinador no encontrado.",
                "agent": "system",
                "error": True
            }
            
        coordinator_response = await coordinator.process(user_input, context)
        intention = coordinator_response.get("intention")
        agent_name = coordinator_response.get("agent")
        
        print(f"🎯 Intención detectada por Coordinador: '{intention}' para agente '{agent_name}'")

        # 🚀 PASO 2: Ejecutar ruta correspondiente
        if intention == "conversacion":
            # RUTA RÁPIDA: Es una pregunta directa, devolver la respuesta del coordinador
            response = coordinator_response
            response["route"] = "direct_llm_response"
        
        elif intention == "comando" and agent_name != "coordinator":
            # RUTA DE AGENTES: Workflow específico
            workflow_name = f"{agent_name}_workflow"
            response = await self._execute_agent_workflow(workflow_name, user_input, context)
        
        else:
            # Fallback si la respuesta del coordinador no es clara
            response = {
                "message": coordinator_response.get("message", "No estoy seguro de cómo proceder. ¿Puedes reformular tu petición?"),
                "agent": "system",
                "route": "error"
            }

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
