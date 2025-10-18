"""
Test de mejoras al sistema de simulación
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.simulator_agent import PatientSimulatorAgent
from src.simulation.transition_model import LearnedTransitionModel
from src.simulation.progress_simulator import ProgressSimulator
from src.simulation.visualization import SimulationVisualizer
from src.utils.logging_config import get_logger
from src.utils.image_embedding import save_response_as_html

logger = get_logger(__name__)


def create_test_data(n_patients: int = 20) -> pd.DataFrame:
    """Crea datos de prueba realistas"""
    np.random.seed(42)
    
    data = {
        'patient_id': range(1, n_patients + 1),
        'age': np.random.randint(20, 90, n_patients),
        'gender': np.random.choice(['MALE', 'FEMALE'], n_patients),
        'diagnosis': ['COVID-19 - POSITIVO'] * n_patients,
        'medication': np.random.choice(['PARACETAMOL', 'REMDESIVIR', 'DEXAMETASONA'], n_patients),
        'icu_days': np.random.choice([0, 0, 0, 1, 2, 5, 10], n_patients),  # Mayoría sin UCI
        'temperature': np.random.uniform(36.5, 39.5, n_patients),
        'oxygen_saturation': np.random.uniform(88, 100, n_patients),
        'pcr_result': np.random.exponential(5, n_patients),  # Distribución exponencial más realista
        'discharge_motive': np.random.choice(['Domicilio', 'Traslado', 'Alta médica'], n_patients)
    }
    
    return pd.DataFrame(data)


def test_transition_model():
    """Test del modelo de transición aprendido"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 1: Modelo de Transición Aprendido")
    logger.info("="*60)
    
    # Crear datos de prueba
    data = create_test_data(50)
    
    # Entrenar modelo
    model = LearnedTransitionModel()
    model.fit(data, disease_type="covid19")
    
    # Hacer predicciones
    logger.info("\n📊 Predicciones de ejemplo:")
    
    test_cases = [
        (95.0, 'oxygen_saturation', 1, 'moderate'),
        (38.5, 'temperature', 3, 'mild'),
        (15.0, 'pcr_result', 5, 'severe')
    ]
    
    for current_val, param, visit, severity in test_cases:
        new_val, improved = model.predict_next_value(current_val, param, visit, severity)
        status = "✅ Mejoró" if improved else "⚠️ Deterioró"
        logger.info(f"  {param}: {current_val:.1f} → {new_val:.1f} ({status}) [Visita {visit}, {severity}]")
    
    logger.info("\n✅ Test de modelo de transición completado\n")


def test_improved_simulator():
    """Test del simulador mejorado"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 2: Simulador Mejorado")
    logger.info("="*60)
    
    # Crear datos
    data = create_test_data(10)
    
    # Simulador con modelo aprendido
    logger.info("\n🎯 Probando con modelo aprendido...")
    sim_learned = ProgressSimulator(data, "covid19", use_learned_model=True)
    evolved_learned, stats_learned = sim_learned.simulate_batch_evolution(data.head(5))
    
    logger.info(f"\n📈 Resultados con modelo aprendido:")
    logger.info(f"  - Total visitas: {stats_learned['total_visits']}")
    logger.info(f"  - Promedio visitas/paciente: {stats_learned['avg_visits_per_patient']:.1f}")
    logger.info(f"  - Pacientes con mejoría: {stats_learned['patients_with_improvement']}")
    logger.info(f"  - Pacientes con deterioro: {stats_learned['patients_with_deterioration']}")
    
    # Simulador simple (fallback)
    logger.info("\n🔄 Probando con modelo simple (fallback)...")
    sim_simple = ProgressSimulator(data, "covid19", use_learned_model=False)
    evolved_simple, stats_simple = sim_simple.simulate_batch_evolution(data.head(5))
    
    logger.info(f"\n📈 Resultados con modelo simple:")
    logger.info(f"  - Total visitas: {stats_simple['total_visits']}")
    logger.info(f"  - Promedio visitas/paciente: {stats_simple['avg_visits_per_patient']:.1f}")
    logger.info(f"  - Pacientes con mejoría: {stats_simple['patients_with_improvement']}")
    logger.info(f"  - Pacientes con deterioro: {stats_simple['patients_with_deterioration']}")
    
    logger.info("\n✅ Test de simulador mejorado completado\n")
    
    return evolved_learned


def test_visualizations(evolved_data: pd.DataFrame):
    """Test de visualizaciones"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 3: Visualizaciones")
    logger.info("="*60)
    
    visualizer = SimulationVisualizer()
    
    # Timeline
    logger.info("\n📊 Generando gráfico de timeline...")
    timeline_path = visualizer.create_patient_timeline_plot(
        evolved_data,
        params=['oxygen_saturation', 'temperature', 'pcr_result'],
        max_patients=3
    )
    
    if timeline_path:
        logger.info(f"  ✅ Timeline guardado en: {timeline_path}")
    
    # Heatmap
    logger.info("\n🔥 Generando heatmap...")
    heatmap_path = visualizer.create_summary_heatmap(
        evolved_data,
        params=['oxygen_saturation', 'temperature', 'pcr_result']
    )
    
    if heatmap_path:
        logger.info(f"  ✅ Heatmap guardado en: {heatmap_path}")
    
    # Markdown summary
    logger.info("\n📝 Generando resumen Markdown...")
    stats = {
        'total_patients': 10,
        'total_visits': 50,
        'avg_visits_per_patient': 5.0,
        'patients_with_improvement': 7,
        'patients_with_deterioration': 3
    }
    
    markdown = visualizer.create_markdown_summary(stats, timeline_path, heatmap_path)
    logger.info("\n" + "-"*60)
    logger.info(markdown)
    logger.info("-"*60)
    
    logger.info("\n✅ Test de visualizaciones completado\n")


async def test_full_agent():
    """Test completo del agente de simulación"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST 4: Agente de Simulación Completo")
    logger.info("="*60)
    
    # Crear datos
    data = create_test_data(15)
    
    # Crear agente
    agent = PatientSimulatorAgent(enable_visualizations=True)
    
    # Preparar contexto
    context = {
        'dataframe': data,
        'universal_analysis': {
            'dataset_type': 'COVID-19'
        }
    }
    
    # Ejecutar simulación
    logger.info("\n🚀 Ejecutando agente de simulación...")
    result = await agent.process("Simula la evolución temporal de estos pacientes", context)
    
    if result.get('error'):
        logger.error(f"❌ Error: {result.get('message')}")
    else:
        logger.info("\n" + "="*60)
        logger.info("📄 INFORME GENERADO:")
        logger.info("="*60)
        logger.info(result.get('message', ''))
        logger.info("="*60)
        
        logger.info(f"\n📊 Timeline: {result.get('timeline_path', 'No generado')}")
        logger.info(f"🔥 Heatmap: {result.get('heatmap_path', 'No generado')}")
        logger.info(f"📈 Stats: {result.get('simulation_stats', {})}")
        
        # 🎨 Verificar si las imágenes fueron embebidas
        if result.get('images_embedded'):
            logger.info("\n✅ Las imágenes fueron embebidas en base64 en el mensaje")
            message = result.get('message', '')
            if 'data:image' in message:
                logger.info("✅ Confirmado: Imágenes base64 detectadas en el markdown")
            else:
                logger.warning("⚠️ No se detectaron imágenes base64 en el mensaje")
        
        # 🌐 Generar y guardar HTML
        logger.info("\n📂 Generando y guardando reporte HTML...")
        html_path = f"reporte_simulacion_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
        try:
            save_response_as_html(result, html_path)
            logger.info(f"  ✅ Reporte HTML guardado en: {html_path}")
        except Exception as e:
            logger.error(f"  ❌ Error al guardar el reporte HTML: {e}")
    
    logger.info("\n✅ Test de agente completo finalizado\n")


def main():
    """Ejecuta todos los tests"""
    logger.info("\n" + "="*80)
    logger.info("🔬 INICIANDO TESTS DE MEJORAS AL SISTEMA DE SIMULACIÓN")
    logger.info("="*80)
    
    try:
        # Test 1: Modelo de transición
        test_transition_model()
        
        # Test 2: Simulador mejorado
        evolved_data = test_improved_simulator()
        
        # Test 3: Visualizaciones
        test_visualizations(evolved_data)
        
        # Test 4: Agente completo (async)
        asyncio.run(test_full_agent())
        
        logger.info("\n" + "="*80)
        logger.info("✅ TODOS LOS TESTS COMPLETADOS EXITOSAMENTE")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n❌ Error durante los tests: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
