"""
Agente Analizador Clínico - Especializado en la interpretación y generación de informes.

Mejoras Fase 2:
- Caché de análisis por hash de dataset
- Performance tracking de etapas
- Métricas de análisis
- Logging estructurado mejorado
"""

import json
import time
from typing import Dict, Any
from src.utils.logging_config import get_logger
from src.utils.optimization_utils import CacheManager, PerformanceTracker, get_dataframe_hash
from .base_agent import BaseLLMAgent, BaseAgentConfig

logger = get_logger(__name__)

# 📊 Fase 2: Métricas del Analizador
class AnalyzerMetrics:
    """Métricas de performance del analizador"""
    def __init__(self):
        self.total_analyses = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_rows_analyzed = 0
        self.total_columns_analyzed = 0
        self.avg_analysis_time = 0.0
        self._analysis_times = []
    
    def record_cache_hit(self, rows: int, cols: int):
        self.cache_hits += 1
        self.total_analyses += 1
        self.total_rows_analyzed += rows
        self.total_columns_analyzed += cols
    
    def record_cache_miss(self, rows: int, cols: int):
        self.cache_misses += 1
        self.total_analyses += 1
        self.total_rows_analyzed += rows
        self.total_columns_analyzed += cols
    
    def record_analysis_time(self, time_s: float):
        self._analysis_times.append(time_s)
        if self._analysis_times:
            self.avg_analysis_time = sum(self._analysis_times) / len(self._analysis_times)
    
    def get_stats(self) -> Dict[str, Any]:
        cache_hit_rate = (self.cache_hits / self.total_analyses * 100) if self.total_analyses > 0 else 0
        return {
            "total_analyses": self.total_analyses,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "total_rows_analyzed": self.total_rows_analyzed,
            "total_columns_analyzed": self.total_columns_analyzed,
            "avg_analysis_time_s": f"{self.avg_analysis_time:.2f}"
        }
    
    def reset(self):
        """Reset todas las métricas"""
        self.__init__()

ANALYZER_SYSTEM_PROMPT = """
Eres un Científico de Datos especializado en salud. Tu tarea es recibir un análisis técnico de un dataset (en JSON) y redactar un informe de análisis exploratorio (EDA) en Markdown.

**ENTRADA (JSON):**
Recibirás un JSON con la estructura:

```json
{{
  "dataset_type": "<tipo>",
  "column_mapping": {{"age_col": "<col>", ...}},
  "basic_stats": {{"rows": <num>, ...}},
  "missing_values": {{"total_missing": <num>, ...}},
  "column_analysis": {{"<col_1>": {{"type": "...", ...}}}}
}}
```

**TAREA (MARKDOWN):**
Tu ÚNICA salida debe ser un informe en MARKDOWN con estas secciones:

1.  **`### 📝 Resumen Ejecutivo`**: Párrafo con los hallazgos clave.
2.  **`### 📊 Análisis Descriptivo`**: Características del dataset.
3.  **`### 🩺 Calidad de los Datos`**: Evaluación de nulos y duplicados.
4.  **`### 🔬 Análisis de Variables Clave`**: Descripción de las variables más importantes (edad, género, diagnóstico).
5.  **`### 💡 Conclusiones y Recomendaciones`**: Idoneidad del dataset para IA y posibles sesgos.

Basa todas tus afirmaciones en los datos del JSON. Sé profesional y objetivo.
"""

class ClinicalAnalyzerConfig(BaseAgentConfig):
    name: str = "Analizador Clínico"
    description: str = "Especialista en análisis estadístico y exploratorio de datos médicos."
    system_prompt: str = ANALYZER_SYSTEM_PROMPT
    max_tokens: int = 2500

class ClinicalAnalyzerAgent(BaseLLMAgent):
    def __init__(self):
        super().__init__(ClinicalAnalyzerConfig(), tools=[])  # Explícitamente sin herramientas
        
        # 🚀 Fase 2: Inicializar caché y métricas
        self.cache = CacheManager(cache_dir='cache/analyzer')
        self.metrics = AnalyzerMetrics()
        self.performance_tracker = PerformanceTracker('analyzer')
        logger.info("✅ Analizador inicializado con caché y métricas")

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if context and context.get("universal_analysis"):
            return await self.analyze_dataset(None, context)
        return {"message": "El analizador necesita un análisis universal previo.", "agent": self.name, "error": True}

    async def analyze_dataset(self, dataframe, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analiza el dataset y genera informe EDA.
        
        Mejoras Fase 2:
        - Caché de análisis completo por hash de dataset
        - Performance tracking
        - Métricas de análisis
        """
        # 📊 Fase 2: Iniciar tracking
        start_time = time.time()
        
        universal_analysis_result = context.get("universal_analysis")
        if not universal_analysis_result:
            return {"message": "Error: No se encontró el resultado del análisis universal.", "agent": self.name, "error": True}
        
        # Extraer info básica para métricas
        basic_info = universal_analysis_result.get("basic_info", {})
        rows = basic_info.get("rows", 0)
        cols = basic_info.get("columns", 0)
        
        # 🚀 Fase 2: Verificar caché de análisis
        # Solo cachear si hay dataframe (para calcular hash)
        cache_key = None
        if context and context.get("dataframe") is not None:
            df = context["dataframe"]
            try:
                df_hash = get_dataframe_hash(df)
                cache_key = f"analysis_{df_hash}"
                
                # Intentar cargar desde caché
                cached_analysis = self.cache.load(cache_key, 'analyses')
                if cached_analysis:
                    elapsed = time.time() - start_time
                    self.metrics.record_cache_hit(rows, cols)
                    self.metrics.record_analysis_time(elapsed)
                    
                    logger.info(
                        f"� Cache HIT - Análisis servido en {elapsed*1000:.0f}ms | "
                        f"{rows} filas x {cols} cols"
                    )
                    logger.debug(f"📊 Métricas actuales: {self.metrics.get_stats()}")
                    
                    return cached_analysis
                
                # Si no hay caché, continuar con análisis normal
                logger.info(
                    f"🔍 Cache MISS - Generando análisis | "
                    f"{rows} filas x {cols} cols"
                )
                self.metrics.record_cache_miss(rows, cols)
                
            except Exception as e:
                logger.warning(f"⚠️ Error calculando hash para caché: {e}")
                cache_key = None
        
        # �🔥 OPTIMIZACIÓN: Reducir el tamaño del contexto para evitar exceder límites de tokens
        # Especialmente importante para datasets grandes (>1000 filas o >50 columnas)
        summary_analysis = self._summarize_analysis_for_llm(universal_analysis_result)
        
        prompt_input = json.dumps(summary_analysis, indent=2)
        
        # 🔥 LÍMITE ADICIONAL: Truncar el JSON si es muy grande (>8000 caracteres)
        MAX_JSON_CHARS = 8000
        if len(prompt_input) > MAX_JSON_CHARS:
            logger.warning("JSON muy grande (%d chars), truncando a %d chars", len(prompt_input), MAX_JSON_CHARS)
            prompt_input = prompt_input[:MAX_JSON_CHARS] + "\n... (truncado por límite de tokens)"
        
        prompt_variables = {"input": prompt_input}
        if self.memory.chat_memory.messages:
            prompt_variables["chat_history"] = self.memory.chat_memory.messages
        
        logger.info("Generando informe EDA con LLM (%s) - Tamaño JSON: %d chars", 
                   type(self.llm).__name__ if self.llm else "mock", len(prompt_input))
        
        llm_response = await self.agent_executor.ainvoke(prompt_variables)
        response_text = self._extract_response_text(llm_response)
        
        # Preparar resultado
        result = {
            "message": response_text,
            "agent": self.name,
            "analysis_complete": True
        }
        
        # 🚀 Fase 2: Guardar en caché si es posible
        if cache_key:
            try:
                self.cache.save(cache_key, result, 'analyses')
                logger.debug(f"💾 Análisis guardado en caché: {cache_key[:16]}...")
            except Exception as e:
                logger.warning(f"⚠️ Error guardando en caché: {e}")
        
        # 📊 Fase 2: Registrar métricas
        elapsed = time.time() - start_time
        self.metrics.record_analysis_time(elapsed)
        
        logger.info(
            f"✅ Análisis completado en {elapsed:.2f}s | "
            f"{rows} filas x {cols} cols | "
            f"Informe: {len(response_text)} chars"
        )
        logger.debug(f"📊 Métricas actuales: {self.metrics.get_stats()}")
        
        return result
    
    def _summarize_analysis_for_llm(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resumir análisis para evitar exceder límites de tokens del LLM
        
        Mejora Fase 2: Balance entre completitud y tamaño de tokens
        - Datasets pequeños (<50 cols): análisis completo
        - Datasets medianos (50-100 cols): análisis detallado con límites
        - Datasets grandes (>100 cols): análisis resumido
        """
        summary = {}
        
        # Información de muestreo (si aplica)
        if "sampling_info" in analysis:
            summary["sampling_info"] = analysis["sampling_info"]
        
        # Resumen básico
        if "basic_info" in analysis:
            summary["basic_info"] = {
                "rows": analysis["basic_info"].get("rows", 0),
                "columns": analysis["basic_info"].get("columns", 0),
                "memory_usage": analysis["basic_info"].get("memory_usage", "N/A")
            }
        
        # Dominio médico
        if "medical_domain" in analysis:
            summary["medical_domain"] = analysis["medical_domain"]
        
        # 🚀 Fase 2: Determinar nivel de detalle según tamaño del dataset
        num_columns = summary.get("basic_info", {}).get("columns", 0)
        
        # Columnas: Incluir información estadística detallada
        if "columns" in analysis:
            cols = analysis["columns"]
            
            # Determinar cuántas columnas incluir en detalle
            if num_columns <= 50:
                # Dataset pequeño: incluir TODAS las columnas con detalles
                max_detailed = len(cols)
            elif num_columns <= 100:
                # Dataset mediano: incluir primeras 30 con detalles
                max_detailed = 30
            else:
                # Dataset grande: incluir primeras 20 con detalles
                max_detailed = 20
            
            # Separar por tipo
            numerical = [c for c in cols if c.get("type") == "numerical"]
            categorical = [c for c in cols if c.get("type") == "categorical"]
            datetime_cols = [c for c in cols if c.get("type") == "datetime"]
            
            summary["columns_summary"] = {
                "total": len(cols),
                "numerical_count": len(numerical),
                "categorical_count": len(categorical),
                "datetime_count": len(datetime_cols),
            }
            
            # 🚀 Incluir estadísticas detalladas de columnas numéricas importantes
            if numerical:
                numerical_details = []
                for col in numerical[:max_detailed]:
                    col_detail = {
                        "name": col.get("name"),
                        "type": "numerical"
                    }
                    # Agregar estadísticas si están disponibles
                    if "stats" in col:
                        stats = col["stats"]
                        col_detail["stats"] = {
                            "mean": stats.get("mean"),
                            "std": stats.get("std"),
                            "min": stats.get("min"),
                            "max": stats.get("max"),
                            "median": stats.get("50%"),
                            "missing": col.get("missing_count", 0)
                        }
                    numerical_details.append(col_detail)
                
                summary["numerical_columns"] = numerical_details
            
            # 🚀 Incluir detalles de columnas categóricas importantes
            if categorical:
                categorical_details = []
                for col in categorical[:max_detailed]:
                    col_detail = {
                        "name": col.get("name"),
                        "type": "categorical",
                        "unique_values": col.get("unique_count", 0),
                        "missing": col.get("missing_count", 0)
                    }
                    # Agregar top valores si están disponibles
                    if "top_values" in col:
                        col_detail["top_values"] = col["top_values"][:5]
                    categorical_details.append(col_detail)
                
                summary["categorical_columns"] = categorical_details
            
            # Incluir nombres de columnas datetime
            if datetime_cols:
                summary["datetime_columns"] = [c.get("name") for c in datetime_cols[:10]]
        
        # 🚀 Valores nulos: Incluir detalles por columna
        if "missing_values" in analysis:
            mv = analysis["missing_values"]
            total_cells = summary["basic_info"]["rows"] * summary["basic_info"]["columns"]
            missing_pct = (mv.get("total_missing", 0) / total_cells * 100) if total_cells > 0 else 0
            
            summary["missing_values"] = {
                "total_missing": mv.get("total_missing", 0),
                "percentage": round(missing_pct, 2),
                "columns_affected": len(mv.get("columns_with_missing", {}))
            }
            
            # Incluir detalles de columnas con más valores nulos
            if "columns_with_missing" in mv:
                cols_missing = mv["columns_with_missing"]
                # Ordenar por cantidad de nulos descendente y tomar top 10
                sorted_missing = sorted(cols_missing.items(), key=lambda x: x[1], reverse=True)[:10]
                summary["missing_values"]["top_missing_columns"] = [
                    {"column": col, "missing_count": count} 
                    for col, count in sorted_missing
                ]
        
        # 🚀 Correlaciones: Incluir más correlaciones si el dataset lo permite
        if "correlations" in analysis and "high_correlations" in analysis["correlations"]:
            high_corr = analysis["correlations"]["high_correlations"]
            # Incluir top 10 correlaciones en lugar de solo 5
            summary["high_correlations"] = high_corr[:10] if isinstance(high_corr, list) else []
        
        # Patrones médicos: incluir más detalles
        if "medical_patterns" in analysis:
            mp = analysis["medical_patterns"]
            summary["medical_patterns"] = {
                "has_patient_id": mp.get("has_patient_id", False),
                "has_age": mp.get("has_age", False),
                "has_dates": mp.get("has_dates", False),
                "has_diagnoses": mp.get("has_diagnoses", False),
                "key_medical_columns": mp.get("key_medical_columns", [])[:15]  # Aumentado de 10 a 15
            }
        
        return summary

    def get_metrics(self) -> Dict[str, Any]:
        """
        🚀 Fase 2: Obtener métricas actuales del analizador
        
        Returns:
            Dict con estadísticas de performance y caché
        """
        stats = self.metrics.get_stats()
        
        return {
            "analysis_metrics": stats,
            "cache_info": {
                "cache_dir": str(self.cache.cache_dir),
                "enabled": True
            },
            "status": "healthy" if stats["total_analyses"] > 0 else "idle"
        }
    
    def reset_metrics(self):
        """
        🚀 Fase 2: Resetear métricas (útil para testing)
        """
        self.metrics.reset()
        self.performance_tracker = PerformanceTracker('analyzer')
        logger.info("🔄 Métricas del analizador reseteadas")
    
    def clear_cache(self, category: str = 'analyses'):
        """
        🚀 Fase 2: Limpiar caché de análisis
        
        Args:
            category: Categoría de caché a limpiar
        """
        self.cache.clear(category)
        logger.info(f"🗑️ Caché del analizador limpiado: {category}")