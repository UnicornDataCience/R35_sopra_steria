# 🧪 Tests del Sistema Patient-IA

Esta carpeta contiene todos los tests automatizados del sistema, incluyendo tests de las fases de optimización.

## 📋 Estructura de Tests

### Tests de Optimización - Fase 1 (Coordinador)
- `test_coordinator_improvements.py` - Suite completa de tests del coordinador
  - Test de caché de respuestas comunes
  - Test de métricas de coordinación
  - Test de performance tracking

### Tests de Optimización - Fase 2 (Analizador)
- `test_analyzer_improvements.py` - Suite completa de tests del analizador
  - Test de caché de análisis completos
  - Test de métricas de análisis
  - Test de dataset modificado
  - Test de limpieza de caché

- `test_analyzer_summary_optimization.py` - Tests del resumen optimizado
  - Validación de estadísticas incluidas
  - Verificación de correlaciones
  - Análisis de patrones médicos
  - Completitud del resumen

- `test_analyzer_summary.py` - Verificación del análisis EDA completo
  - Validación de estadísticas descriptivas
  - Verificación del tamaño del JSON
  - Análisis de columnas detalladas

### Tests de Optimización - Fase 3 (Generador) 🟡
- `test_generator_improvements.py` - Suite completa de tests del generador
  - Test de funcionalidad de caché (❌ pendiente integración)
  - Test de métricas de calidad (✅ funcionando)
  - Test de métricas de rendimiento
  - Test de calidad de datos generados

### Tests de Contexto y Chat
- `test_chat_context_multi_agent.py` - Tests del chat contextual multi-agente

### Tests de Debugging
- `debug_detector.py` - Script de debug del UniversalDatasetDetector
- `test_complete_system.py` - Test integral del sistema

### Tests en Subcarpeta
- `results/` - Resultados de tests ejecutados
- `COMO_PROBAR_CHAT_CONTEXTUAL.md` - Guía de testing del chat

## 🚀 Ejecución de Tests

### Tests de Fase 1 (Coordinador)
```bash
uv run python tests/test_coordinator_improvements.py
```

### Tests de Fase 2 (Analizador)
```bash
# Suite completa
uv run python tests/test_analyzer_improvements.py

# Resumen optimizado
uv run python tests/test_analyzer_summary_optimization.py

# Análisis EDA completo
uv run python tests/test_analyzer_summary.py
```

### Test Integral del Sistema
```bash
uv run python validate_system.py
```

## ✅ Estado de Tests

| Test | Estado | Última Ejecución |
|------|--------|------------------|
| `test_coordinator_improvements.py` | ✅ Pasando | 15 Oct 2025 |
| `test_analyzer_improvements.py` | ✅ Pasando | 16 Oct 2025 |
| `test_analyzer_summary_optimization.py` | ✅ Pasando | 16 Oct 2025 |
| `test_analyzer_summary.py` | ✅ Pasando | 16 Oct 2025 |
| `test_chat_context_multi_agent.py` | ⏳ Pendiente | - |

## 📊 Cobertura

- **Coordinador**: 100% - Caché, métricas, performance
- **Analizador**: 100% - Caché, métricas, resumen, EDA
- **Generador**: Pendiente (Fase 3)
- **Validador**: Pendiente (Fase 4)
- **Evaluador**: Pendiente (Fase 5)
- **Simulador**: Pendiente (Fase 6)

## 🔧 Debugging

Para debugging de componentes específicos:
```bash
# Debug del detector de datasets
uv run python tests/debug_detector.py

# Debug del sistema completo
uv run python tests/test_complete_system.py
```

## 📝 Convenciones

- Todos los tests usan `uv run` para gestión de dependencias
- Los tests imprimen resultados detallados con emojis para mejor legibilidad
- Cada test valida métricas específicas y las compara con umbrales esperados
- Los tests de optimización validan que no hay breaking changes

---

**Última actualización**: 16 de Octubre, 2025
