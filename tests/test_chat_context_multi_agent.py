"""
Test de Chat Contextual Multi-Agente
=====================================

Este script prueba que el chat puede responder preguntas sobre
resultados de TODOS los agentes (análisis, generación, validación, etc.)

Casos de prueba:
1. Análisis → Preguntar sobre columnas, nulos, estadísticas
2. Generación → Preguntar sobre modelo usado, cantidad de datos
3. Validación → Preguntar sobre inconsistencias encontradas
4. Múltiples operaciones → Resumir todo lo ejecutado
"""

import requests
import json
import time
from typing import Dict, Any

# Configuración
BASE_URL = "http://localhost:8000/api/v1"
DATASET_ID = "test_dataset"  # Ajustar según tu dataset

class ChatContextTester:
    def __init__(self, base_url: str, dataset_id: str):
        self.base_url = base_url
        self.dataset_id = dataset_id
        self.session = requests.Session()
        
    def health_check(self) -> bool:
        """Verificar que el API esté funcionando"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ API está funcionando correctamente")
                return True
            else:
                print(f"❌ API respondió con status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error conectando al API: {e}")
            return False
    
    def simulate_analysis(self) -> Dict[str, Any]:
        """Simular resultado de análisis"""
        return {
            "type": "analysis",
            "markdown": """# 📊 Análisis Completo del Dataset

## Resumen General
- **Total de registros**: 1,234
- **Total de columnas**: 15
- **Valores nulos**: 234 (18.9%)

## Columnas Numéricas (8)
- edad: 45.2 ± 12.3 años
- presion_sistolica: 120 ± 15 mmHg
- presion_diastolica: 80 ± 10 mmHg
- glucosa: 95 ± 20 mg/dL

## Columnas Categóricas (7)
- diagnostico: 5 categorías únicas
- tratamiento: 12 tipos diferentes
- genero: 2 categorías
""",
            "meta": {
                "timestamp": "2024-01-15T10:30:00",
                "duration": "45s"
            }
        }
    
    def simulate_generation(self) -> Dict[str, Any]:
        """Simular resultado de generación"""
        return {
            "type": "generation",
            "markdown": """# 🎲 Generación de Datos Sintéticos

## Configuración
- **Modelo**: CTGAN
- **Registros generados**: 500
- **Tiempo de entrenamiento**: 120s

## Resultados
- Calidad de generación: 87%
- Preservación de distribuciones: 92%
- Correlaciones mantenidas: 85%
""",
            "meta": {
                "model_type": "CTGAN",
                "num_samples": 500,
                "training_epochs": 100
            }
        }
    
    def simulate_validation(self) -> Dict[str, Any]:
        """Simular resultado de validación"""
        return {
            "type": "validation",
            "markdown": """# ✅ Validación Médica

## Inconsistencias Detectadas
- **Total de errores**: 12
- **Categorías**:
  - Valores fuera de rango clínico: 5
  - Combinaciones incompatibles: 4
  - Fechas inconsistentes: 3

## Columnas con Problemas
- presion_sistolica: 3 valores >200 mmHg
- glucosa: 2 valores <20 mg/dL
- fecha_nacimiento: 3 fechas futuras
""",
            "meta": {
                "total_errors": 12,
                "severity": "medium"
            }
        }
    
    def send_chat_with_context(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enviar mensaje al chat con contexto"""
        try:
            payload = {
                "message": message,
                "context": context
            }
            
            response = self.session.post(
                f"{self.base_url}/chat/send",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Status {response.status_code}", "detail": response.text}
                
        except Exception as e:
            return {"error": str(e)}
    
    def build_enriched_context(self, dataset_info: Dict, recent_results: Dict) -> Dict[str, Any]:
        """Construir contexto enriquecido como lo hace el frontend"""
        return {
            "has_dataset": True,
            "dataset_id": self.dataset_id,
            "dataset_info": dataset_info,
            "columns": ["edad", "genero", "diagnostico", "presion_sistolica", "presion_diastolica", 
                       "glucosa", "tratamiento", "fecha_ingreso", "fecha_alta", "estado"],
            "statistics_summary": {
                "has_missing_values": True,
                "numerical_columns": ["edad", "presion_sistolica", "presion_diastolica", "glucosa"],
                "categorical_columns": ["genero", "diagnostico", "tratamiento", "estado"],
                "total_numerical": 4,
                "total_categorical": 4
            },
            "recent_results": recent_results,
            "active_operation": list(recent_results.keys())[0] if recent_results else "none"
        }
    
    def run_test_scenario_1(self):
        """Test 1: Análisis + Preguntas sobre el análisis"""
        print("\n" + "="*60)
        print("🧪 TEST 1: Análisis + Chat sobre Análisis")
        print("="*60)
        
        dataset_info = {
            "total_rows": 1234,
            "total_columns": 15,
            "null_count": 234,
            "selected_target": "diagnostico"
        }
        
        analysis_result = self.simulate_analysis()
        
        context = self.build_enriched_context(
            dataset_info=dataset_info,
            recent_results={"analysis": {
                "has_result": True,
                "summary": analysis_result["markdown"],
                "meta": analysis_result["meta"]
            }}
        )
        
        # Pregunta 1: ¿Cuántas columnas numéricas hay?
        print("\n📝 Pregunta: ¿Cuántas columnas numéricas hay?")
        response = self.send_chat_with_context("¿Cuántas columnas numéricas hay?", context)
        print(f"💬 Respuesta: {response.get('message', response)}\n")
        time.sleep(1)
        
        # Pregunta 2: ¿Cuántos valores nulos encontraste?
        print("📝 Pregunta: ¿Cuántos valores nulos encontraste?")
        response = self.send_chat_with_context("¿Cuántos valores nulos encontraste?", context)
        print(f"💬 Respuesta: {response.get('message', response)}\n")
        time.sleep(1)
    
    def run_test_scenario_2(self):
        """Test 2: Generación + Preguntas sobre generación"""
        print("\n" + "="*60)
        print("🧪 TEST 2: Generación + Chat sobre Generación")
        print("="*60)
        
        dataset_info = {
            "total_rows": 1234,
            "total_columns": 15,
            "null_count": 234,
            "selected_target": "diagnostico"
        }
        
        generation_result = self.simulate_generation()
        
        context = self.build_enriched_context(
            dataset_info=dataset_info,
            recent_results={"generation": {
                "has_result": True,
                "summary": generation_result["markdown"],
                "meta": generation_result["meta"]
            }}
        )
        
        # Pregunta 1: ¿Qué modelo usaste para generar?
        print("\n📝 Pregunta: ¿Qué modelo usaste para generar?")
        response = self.send_chat_with_context("¿Qué modelo usaste para generar?", context)
        print(f"💬 Respuesta: {response.get('message', response)}\n")
        time.sleep(1)
        
        # Pregunta 2: ¿Cuántos datos sintéticos generaste?
        print("📝 Pregunta: ¿Cuántos datos sintéticos generaste?")
        response = self.send_chat_with_context("¿Cuántos datos sintéticos generaste?", context)
        print(f"💬 Respuesta: {response.get('message', response)}\n")
        time.sleep(1)
    
    def run_test_scenario_3(self):
        """Test 3: Validación + Preguntas sobre validación"""
        print("\n" + "="*60)
        print("🧪 TEST 3: Validación + Chat sobre Validación")
        print("="*60)
        
        dataset_info = {
            "total_rows": 1234,
            "total_columns": 15,
            "null_count": 234,
            "selected_target": "diagnostico"
        }
        
        validation_result = self.simulate_validation()
        
        context = self.build_enriched_context(
            dataset_info=dataset_info,
            recent_results={"validation": {
                "has_result": True,
                "summary": validation_result["markdown"],
                "meta": validation_result["meta"]
            }}
        )
        
        # Pregunta 1: ¿Encontraste errores en la validación?
        print("\n📝 Pregunta: ¿Encontraste errores en la validación?")
        response = self.send_chat_with_context("¿Encontraste errores en la validación?", context)
        print(f"💬 Respuesta: {response.get('message', response)}\n")
        time.sleep(1)
        
        # Pregunta 2: ¿Qué columnas tienen problemas?
        print("📝 Pregunta: ¿Qué columnas tienen problemas?")
        response = self.send_chat_with_context("¿Qué columnas tienen problemas?", context)
        print(f"💬 Respuesta: {response.get('message', response)}\n")
        time.sleep(1)
    
    def run_test_scenario_4(self):
        """Test 4: Múltiples operaciones + Resumen"""
        print("\n" + "="*60)
        print("🧪 TEST 4: Múltiples Operaciones + Resumen General")
        print("="*60)
        
        dataset_info = {
            "total_rows": 1234,
            "total_columns": 15,
            "null_count": 234,
            "selected_target": "diagnostico"
        }
        
        analysis_result = self.simulate_analysis()
        generation_result = self.simulate_generation()
        validation_result = self.simulate_validation()
        
        context = self.build_enriched_context(
            dataset_info=dataset_info,
            recent_results={
                "analysis": {
                    "has_result": True,
                    "summary": analysis_result["markdown"],
                    "meta": analysis_result["meta"]
                },
                "generation": {
                    "has_result": True,
                    "summary": generation_result["markdown"],
                    "meta": generation_result["meta"]
                },
                "validation": {
                    "has_result": True,
                    "summary": validation_result["markdown"],
                    "meta": validation_result["meta"]
                }
            }
        )
        
        # Pregunta: Resume todas las operaciones que hice
        print("\n📝 Pregunta: Resume todas las operaciones que hice")
        response = self.send_chat_with_context("Resume todas las operaciones que hice", context)
        print(f"💬 Respuesta: {response.get('message', response)}\n")
        time.sleep(1)
    
    def run_all_tests(self):
        """Ejecutar todos los tests"""
        print("\n🚀 Iniciando Tests de Chat Contextual Multi-Agente\n")
        
        if not self.health_check():
            print("❌ El API no está disponible. Abortando tests.")
            return
        
        try:
            self.run_test_scenario_1()
            self.run_test_scenario_2()
            self.run_test_scenario_3()
            self.run_test_scenario_4()
            
            print("\n" + "="*60)
            print("✅ TODOS LOS TESTS COMPLETADOS")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrumpidos por el usuario")
        except Exception as e:
            print(f"\n\n❌ Error durante los tests: {e}")

def main():
    """Función principal"""
    tester = ChatContextTester(
        base_url=BASE_URL,
        dataset_id=DATASET_ID
    )
    tester.run_all_tests()

if __name__ == "__main__":
    main()
