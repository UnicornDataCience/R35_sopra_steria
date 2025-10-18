"""
Visualización eficiente de evolución temporal de pacientes
"""

import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidores
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from io import BytesIO
import base64
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Configurar estilo
sns.set_theme(style="whitegrid", palette="muted")


class SimulationVisualizer:
    """Generador de visualizaciones para simulaciones temporales"""
    
    def __init__(self, output_dir: str = "temp_generations/simulations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_patient_timeline_plot(
        self,
        evolved_data: pd.DataFrame,
        patient_ids: Optional[List[str]] = None,
        params: List[str] = None,
        max_patients: int = 5
    ) -> str:
        """
        Crea gráfico de línea para evolución temporal de pacientes
        
        Args:
            evolved_data: DataFrame con datos evolutivos
            patient_ids: IDs específicos a visualizar (None = aleatorios)
            params: Parámetros a graficar
            max_patients: Máximo de pacientes a mostrar
            
        Returns:
            Path al archivo PNG generado
        """
        if params is None:
            params = ['oxygen_saturation', 'temperature', 'pcr_result']
        
        # Filtrar parámetros existentes
        params = [p for p in params if p in evolved_data.columns]
        
        if not params:
            logger.warning("⚠️ No hay parámetros válidos para visualizar")
            return None
        
        # Extraer patient_id original y visit_number
        if 'patient_id' not in evolved_data.columns:
            logger.warning("⚠️ No hay columna patient_id")
            return None
        
        # Parsear patient_id para extraer ID original y número de visita
        def parse_patient_id(pid):
            if isinstance(pid, str) and '_V' in pid:
                parts = pid.split('_V')
                return parts[0], int(parts[1]) if len(parts) > 1 else 0
            return str(pid), 0
        
        evolved_data['original_patient_id'] = evolved_data['patient_id'].apply(lambda x: parse_patient_id(x)[0])
        evolved_data['visit_number'] = evolved_data['patient_id'].apply(lambda x: parse_patient_id(x)[1])
        
        # Seleccionar pacientes
        unique_patients = evolved_data['original_patient_id'].unique()
        if patient_ids is None:
            # Seleccionar pacientes con más visitas
            patient_counts = evolved_data['original_patient_id'].value_counts()
            selected_patients = patient_counts.head(max_patients).index.tolist()
        else:
            selected_patients = patient_ids[:max_patients]
        
        # Filtrar datos
        plot_data = evolved_data[evolved_data['original_patient_id'].isin(selected_patients)].copy()
        
        # Crear subplots
        n_params = len(params)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 4 * n_params))
        
        if n_params == 1:
            axes = [axes]
        
        # Mapeo de nombres a etiquetas legibles
        param_labels = {
            'oxygen_saturation': 'Saturación de O₂ (%)',
            'temperature': 'Temperatura (°C)',
            'pcr_result': 'PCR (mg/L)',
            'heart_rate': 'Frecuencia Cardíaca (bpm)',
            'blood_pressure': 'Presión Arterial (mmHg)'
        }
        
        for idx, param in enumerate(params):
            ax = axes[idx]
            
            # Graficar cada paciente
            for patient_id in selected_patients:
                patient_data = plot_data[plot_data['original_patient_id'] == patient_id].sort_values('visit_number')
                
                if len(patient_data) > 0:
                    ax.plot(
                        patient_data['visit_number'],
                        patient_data[param],
                        marker='o',
                        label=f'Paciente {patient_id}',
                        linewidth=2,
                        markersize=6
                    )
            
            # Añadir líneas de referencia según el parámetro
            if param == 'oxygen_saturation':
                ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='Normal (≥95%)')
                ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='Límite crítico (90%)')
            elif param == 'temperature':
                ax.axhline(y=37, color='green', linestyle='--', alpha=0.5, label='Normal (37°C)')
                ax.axhline(y=38, color='orange', linestyle='--', alpha=0.5, label='Fiebre (38°C)')
            elif param == 'pcr_result':
                ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='Elevado (10 mg/L)')
            
            ax.set_xlabel('Número de Visita', fontsize=11)
            ax.set_ylabel(param_labels.get(param, param), fontsize=11)
            ax.set_title(f'Evolución: {param_labels.get(param, param)}', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar
        output_path = self.output_dir / f"timeline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Gráfico de timeline guardado: {output_path}")
        return str(output_path)
    
    def create_summary_heatmap(
        self,
        evolved_data: pd.DataFrame,
        params: List[str] = None
    ) -> str:
        """
        Crea heatmap de resumen con promedio de parámetros por visita
        
        Args:
            evolved_data: DataFrame con datos evolutivos
            params: Parámetros a incluir
            
        Returns:
            Path al archivo PNG
        """
        if params is None:
            params = ['oxygen_saturation', 'temperature', 'pcr_result']
        
        params = [p for p in params if p in evolved_data.columns]
        
        if not params or 'patient_id' not in evolved_data.columns:
            logger.warning("⚠️ Datos insuficientes para heatmap")
            return None
        
        # Parsear visit_number
        def parse_visit(pid):
            if isinstance(pid, str) and '_V' in pid:
                return int(pid.split('_V')[1])
            return 0
        
        evolved_data['visit_number'] = evolved_data['patient_id'].apply(parse_visit)
        
        # Calcular promedios por visita
        summary = evolved_data.groupby('visit_number')[params].mean()
        
        # Normalizar para mejor visualización
        summary_norm = (summary - summary.min()) / (summary.max() - summary.min())
        
        # Crear heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.heatmap(
            summary_norm.T,
            annot=summary.T,
            fmt='.1f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Valor Normalizado'},
            ax=ax
        )
        
        ax.set_xlabel('Número de Visita', fontsize=12)
        ax.set_ylabel('Parámetro Clínico', fontsize=12)
        ax.set_title('Evolución Promedio de Parámetros Clínicos', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"heatmap_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"🔥 Heatmap guardado: {output_path}")
        return str(output_path)
    
    def plot_to_base64(self, fig) -> str:
        """Convierte figura matplotlib a string base64 para embedding"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_str
    
    def create_markdown_summary(
        self,
        stats: Dict[str, Any],
        timeline_path: Optional[str] = None,
        heatmap_path: Optional[str] = None
    ) -> str:
        """
        Crea resumen en Markdown con enlaces a visualizaciones
        
        Args:
            stats: Diccionario de estadísticas
            timeline_path: Path al gráfico de timeline
            heatmap_path: Path al heatmap
            
        Returns:
            String con Markdown formateado
        """
        md = "# 📊 Resumen de Simulación Temporal\n\n"
        
        md += "## 📈 Estadísticas Generales\n\n"
        md += f"- **Total de pacientes simulados**: {stats.get('total_patients', 0)}\n"
        md += f"- **Total de visitas generadas**: {stats.get('total_visits', 0)}\n"
        md += f"- **Promedio de visitas por paciente**: {stats.get('avg_visits_per_patient', 0):.1f}\n"
        md += f"- **Pacientes con mejoría**: {stats.get('patients_with_improvement', 0)} "
        md += f"({stats.get('patients_with_improvement', 0) / stats.get('total_patients', 1) * 100:.1f}%)\n"
        md += f"- **Pacientes con deterioro**: {stats.get('patients_with_deterioration', 0)} "
        md += f"({stats.get('patients_with_deterioration', 0) / stats.get('total_patients', 1) * 100:.1f}%)\n\n"
        
        if 'model_quality' in stats:
            md += "## 🎯 Calidad del Modelo de Transición\n\n"
            for param, quality in stats['model_quality'].items():
                md += f"- **{param}**: Realismo {quality:.2%}\n"
            md += "\n"
        
        if timeline_path:
            md += "## 📉 Visualización de Evolución Temporal\n\n"
            md += f"![Evolución Temporal]({timeline_path})\n\n"
        
        if heatmap_path:
            md += "## 🔥 Mapa de Calor de Evolución\n\n"
            md += f"![Heatmap de Evolución]({heatmap_path})\n\n"
        
        md += "---\n\n"
        md += "*Generado por el Sistema de Simulación de Evolución Temporal*\n"
        
        return md
