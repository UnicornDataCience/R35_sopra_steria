# 🧪 Cómo Probar el Chat Contextual Multi-Agente

## 📋 Requisitos Previos

1. **Backend funcionando**:
   ```bash
   python run_api.py
   ```

2. **Frontend abierto** en navegador:
   ```
   http://localhost:8000
   ```

3. **Dataset cargado** en la aplicación

---

## 🎯 Test Manual (Recomendado)

### Paso 1: Ejecutar Análisis
1. Cargar un dataset (ej: diabetes.csv)
2. Seleccionar target (ej: "Outcome")
3. Hacer clic en **"Analizar Datos"**
4. Esperar a que aparezca el resultado del análisis

### Paso 2: Preguntar en el Chat
En el panel de chat (derecha), hacer preguntas como:

```
¿Cuántas columnas numéricas tiene el dataset?
```
**Esperado**: Respuesta con el número exacto del análisis

```
¿Cuántos valores nulos hay?
```
**Esperado**: Respuesta con el número exacto del análisis

```
¿Cuál es la edad promedio de los pacientes?
```
**Esperado**: Respuesta con el valor exacto (si existe esa columna)

```
¿Qué columnas categóricas detectaste?
```
**Esperado**: Lista de columnas categóricas del análisis

### Paso 3: Generar Datos Sintéticos
1. Hacer clic en **"Generar Datos Sintéticos"**
2. Seleccionar modelo (ej: CTGAN)
3. Configurar cantidad de muestras (ej: 500)
4. Esperar resultado

### Paso 4: Preguntar sobre Generación
En el chat:

```
¿Qué modelo usaste para generar los datos?
```
**Esperado**: "CTGAN"

```
¿Cuántos datos sintéticos generaste?
```
**Esperado**: "500" (o el número que configuraste)

```
¿Cuál fue la calidad de la generación?
```
**Esperado**: Respuesta con las métricas del resultado

### Paso 5: Validar Datos
1. Hacer clic en **"Validar Datos"**
2. Esperar resultado de validación

### Paso 6: Preguntar sobre Validación
En el chat:

```
¿Encontraste inconsistencias en la validación?
```
**Esperado**: Respuesta con el número y tipo de inconsistencias

```
¿Qué columnas tienen problemas médicos?
```
**Esperado**: Lista de columnas con problemas detectados

### Paso 7: Resumen Multi-Operación
Después de ejecutar varias operaciones, preguntar:

```
Resume todas las operaciones que hice
```
**Esperado**: Lista completa de:
- Análisis realizado
- Datos generados
- Validación ejecutada
- Otros resultados disponibles

---

## 🤖 Test Automatizado (Opcional)

### Opción 1: Script Python

1. **Asegúrate de que el API esté corriendo**:
   ```bash
   python run_api.py
   ```

2. **Ejecuta el script de test**:
   ```bash
   python tests/test_chat_context_multi_agent.py
   ```

Este script:
- ✅ Verifica que el API esté funcionando
- ✅ Simula resultados de análisis, generación, validación
- ✅ Hace preguntas al chat con contexto enriquecido
- ✅ Verifica que las respuestas sean coherentes

### Opción 2: Test desde Frontend (Consola del Navegador)

Abre la consola del navegador (F12) y ejecuta:

```javascript
// Verificar que el contexto se esté construyendo correctamente
const context = dashboard.buildEnrichedChatContext();
console.log('📋 Contexto del Chat:', context);

// Verificar qué resultados están disponibles
console.log('📊 Resultados disponibles:', 
    Object.keys(context.recent_results).filter(k => context.recent_results[k].has_result)
);

// Ver el contexto completo de la última operación
console.log('🔍 Última operación:', context.active_operation);
console.log('📄 Detalle:', context.recent_results[context.active_operation]);
```

---

## 📊 Ejemplo de Flujo Completo

```
1. Usuario: [Carga dataset diabetes.csv]
   → Dashboard muestra estadísticas básicas

2. Usuario: [Hace clic en "Analizar Datos"]
   → Sistema ejecuta análisis completo
   → Aparece resultado en panel central

3. Usuario: "¿Cuántas columnas numéricas hay?"
   → Chat consulta context.recent_results.analysis
   → Responde: "El dataset tiene 8 columnas numéricas: Glucose, BloodPressure, Insulin..."

4. Usuario: [Hace clic en "Generar Datos" → CTGAN → 500 muestras]
   → Sistema genera datos sintéticos
   → Aparece resultado en panel central

5. Usuario: "¿Qué modelo usaste?"
   → Chat consulta context.recent_results.generation
   → Responde: "Utilicé el modelo CTGAN para generar 500 registros sintéticos..."

6. Usuario: [Hace clic en "Validar Datos"]
   → Sistema valida coherencia médica
   → Aparece resultado en panel central

7. Usuario: "¿Encontraste problemas?"
   → Chat consulta context.recent_results.validation
   → Responde: "Encontré 12 inconsistencias: 5 valores fuera de rango clínico..."

8. Usuario: "Resume todo"
   → Chat consulta todos los recent_results
   → Responde con resumen de análisis, generación y validación
```

---

## ✅ Checklist de Verificación

Marca cada item cuando lo pruebes:

- [ ] **Análisis**
  - [ ] Ejecutar análisis
  - [ ] Preguntar sobre columnas
  - [ ] Preguntar sobre valores nulos
  - [ ] Preguntar sobre estadísticas

- [ ] **Generación**
  - [ ] Generar datos sintéticos
  - [ ] Preguntar qué modelo se usó
  - [ ] Preguntar cuántos datos se generaron
  - [ ] Preguntar sobre calidad

- [ ] **Validación**
  - [ ] Ejecutar validación
  - [ ] Preguntar sobre errores encontrados
  - [ ] Preguntar qué columnas tienen problemas

- [ ] **Multi-Operación**
  - [ ] Ejecutar 2-3 operaciones seguidas
  - [ ] Preguntar por resumen general
  - [ ] Verificar que mencione todas las operaciones

---

## 🐛 Troubleshooting

### Problema: Chat no responde con datos específicos
**Solución**: 
1. Verifica en consola del navegador:
   ```javascript
   console.log(dashboard.resultContexts);
   ```
2. Debe haber objetos con `markdown` y `meta` para cada operación ejecutada

### Problema: Chat dice "No hay dataset activo"
**Solución**: 
1. Asegúrate de haber cargado un dataset
2. Verifica que `dashboard.currentDatasetId` no sea null
3. Refresca las estadísticas

### Problema: Respuestas genéricas sin contexto
**Solución**: 
1. Abre DevTools → Network
2. Busca la llamada a `/chat/send`
3. Verifica que el payload incluya `context.recent_results`

### Problema: Backend no recibe contexto
**Solución**:
1. Revisa logs del backend:
   ```
   ✅ Prompt enriquecido: 1234 registros, 15 columnas, operaciones=['analysis', 'generation']
   ```
2. Si no aparece, verifica que `buildEnrichedChatContext()` se esté llamando

---

## 📈 Métricas de Éxito

| Métrica | Target | Estado |
|---------|--------|--------|
| Chat responde con datos específicos | 100% | ⬜ |
| Identifica operación activa | 100% | ⬜ |
| Incluye contexto de múltiples operaciones | ✅ | ⬜ |
| Respuestas precisas (no genéricas) | 90%+ | ⬜ |
| Tiempo de respuesta | <3s | ⬜ |

---

## 🎯 Resultado Esperado

**Antes**:
```
Usuario: ¿Cuántas columnas tiene?
Chat: Los datasets médicos suelen tener entre 10 y 50 columnas... [respuesta genérica]
```

**Después**:
```
Usuario: ¿Cuántas columnas tiene?
Chat: El dataset actual tiene exactamente 15 columnas: edad, genero, diagnostico... ✅
```

---

## 📝 Notas

- El contexto se construye **automáticamente** después de cada operación
- No necesitas configurar nada manualmente
- El chat **siempre** usa el resultado más reciente de cada agente
- Si ejecutas la misma operación dos veces, el contexto se actualiza al resultado más reciente

¡Disfruta de tu chat contextual inteligente! 🚀
