// Dashboard functionality for Patient-IA
class PatientIADashboard {
    // API base and state
    apiBase = (window.API_BASE_URL || `${window.location.origin}/api/v1`);
    currentDatasetId = null;
    // Store latest results rendered in the central panel to pass as chat context
    resultContexts = { analysis: null, generation: null, validation: null, evaluation: null, simulation: null };
    activeResultType = null;
    includeChartsFlag = false;
    targetSelections = {};

    constructor() {
        this._eventsWired = false;
        // Initialize Notyf for notifications
        this.notyf = new Notyf({
            duration: 4000,
            position: { x: 'right', y: 'top' },
            types: [
                {
                    type: 'success',
                    background: '#38a169',
                    icon: { className: 'fas fa-check', tagName: 'i', color: 'white' }
                },
                {
                    type: 'error',
                    background: '#e53e3e',
                    icon: { className: 'fas fa-times', tagName: 'i', color: 'white' }
                },
                {
                    type: 'info',
                    background: '#319795',
                    icon: { className: 'fas fa-info', tagName: 'i', color: 'white' }
                },
                {
                    type: 'warning',
                    background: '#d69e2e',
                    icon: { className: 'fas fa-exclamation-triangle', tagName: 'i', color: 'white' }
                }
            ]
        });
        // Load saved target selections
        try { this.targetSelections = JSON.parse(localStorage.getItem('targetSelections') || '{}') || {}; } catch { this.targetSelections = {}; }
        console.log('🚀 Patient-IA Dashboard initialized');
        console.log('📡 API Base:', this.apiBase);
        this.init();
    }

    // Persist last central results to build chat context
    setResultContext(type, payload) {
        try {
            if (!type) return;
            this.resultContexts = this.resultContexts || {};
            this.resultContexts[type] = payload || null;
            this.activeResultType = type;
        } catch {}
    }

    buildChatContext() {
        const base = this.currentDatasetId ? { dataset_id: this.currentDatasetId } : {};
        const central_results = [];
        try {
            const keys = Object.keys(this.resultContexts || {});
            for (const k of keys) {
                const ctx = this.resultContexts[k];
                if (!ctx) continue;
                central_results.push({
                    type: k,
                    title: (ctx.meta && ctx.meta.title) || k,
                    markdown: (ctx.markdown || '').toString().slice(0, 6000),
                    tablePreview: Array.isArray(ctx.tablePreview) ? ctx.tablePreview.slice(0, 10) : undefined,
                    meta: ctx.meta || {}
                });
            }
        } catch {}
        return {
            ...base,
            ui_mode: 'qa_over_results',
            active_result_type: this.activeResultType,
            central_results,
            selected_target: this.getTargetSelection(this.currentDatasetId)
        };
    }

    // NEW: Build enriched context specifically for chat with ALL dataset info + ALL agent results
    buildEnrichedChatContext() {
        if (!this.currentDatasetId) {
            return { 
                has_dataset: false,
                message: "No hay dataset activo. Por favor, carga un dataset primero."
            };
        }

        const stats = this.lastStats || {};
        const target = this.getTargetSelection(this.currentDatasetId);
        
        // Extract column names from stats
        const columns = stats.columns || [];
        const numerical_cols = stats.numerical_columns || [];
        const categorical_cols = stats.categorical_columns || [];

        const context = {
            has_dataset: true,
            dataset_id: this.currentDatasetId,
            
            // Dataset basic info
            dataset_info: {
                total_rows: stats.overview?.patients || stats.overview?.total_rows || null,
                total_columns: stats.overview?.total_columns || columns.length || null,
                average_age: stats.overview?.average_age ? Math.round(stats.overview.average_age) : null,
                null_count: stats.overview?.total_nulls || 0,
                selected_target: target || "Sin definir"
            },
            
            // Column information - CRITICAL for answering questions about columns
            columns: columns,
            
            // Statistical summary
            statistics_summary: {
                has_missing_values: (stats.overview?.total_nulls || 0) > 0,
                numerical_columns: numerical_cols,
                categorical_columns: categorical_cols,
                total_numerical: numerical_cols.length,
                total_categorical: categorical_cols.length
            },
            
            // 🔥 NEW: Include ALL agent results for contextual Q&A
            recent_results: {}
        };

        // Extract all available agent results
        const agentTypes = ['analysis', 'generation', 'validation', 'evaluation', 'simulation'];
        
        for (const agentType of agentTypes) {
            const result = this.resultContexts[agentType];
            if (result && result.markdown) {
                context.recent_results[agentType] = {
                    has_result: true,
                    summary: result.markdown.substring(0, 2000), // First 2000 chars for context
                    meta: result.meta || {},
                    tablePreview: result.tablePreview ? result.tablePreview.slice(0, 5) : null
                };
            } else {
                context.recent_results[agentType] = { has_result: false };
            }
        }
        
        // Identify most recent operation for context priority
        context.active_operation = this.activeResultType || 'none';
        
        // Debug log to verify context
        console.log('🔍 Enhanced Chat Context Built:', {
            dataset: context.dataset_id,
            rows: context.dataset_info.total_rows,
            columns: context.columns.length,
            target: context.dataset_info.selected_target,
            recent_operations: Object.keys(context.recent_results).filter(k => context.recent_results[k].has_result)
        });
        
        return context;
    }

    init() {
        this.setupCharts();
        this.setupEventListeners();
        
        // Debug: Check DOM structure
        setTimeout(() => {
            const statNumbers = document.querySelectorAll('.stat-number');
            console.log('🔍 DOM Debug - Found .stat-number elements:', statNumbers.length);
            statNumbers.forEach((el, i) => {
                console.log(`  - KPI ${i}:`, el.textContent, el);
            });
        }, 500);
        
        // Load list of datasets on startup
        this.loadDatasetsList();
        // Restore dataset id from localStorage and load preview
        try {
            const savedId = localStorage.getItem('currentDatasetId');
            console.log('📂 Restored dataset ID from localStorage:', savedId);
            console.log('📦 Available target selections:', this.targetSelections);
            if (savedId) {
                this.currentDatasetId = savedId;
                const savedTarget = this.getTargetSelection(savedId);
                console.log('🎯 Saved target for restored dataset:', savedTarget);
                this.loadDatasetPreview(10);
            }
        } catch {}
        this.loadInitialData();
    }

    setupCharts() {
        this.createTemporalChart();
        this.createDiagnosticsChart();
    }

    createTemporalChart(labels = null, data = null, yAxisLabel = 'Frecuencia', xAxisLabel = 'Categorías / Rangos') {
        const ctx = document.getElementById('temporalChart');
        if (!ctx) return;

        // Destruir gráfica anterior si existe
        if (this.temporalChartInstance) {
            this.temporalChartInstance.destroy();
        }

        const defaultLabels = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun'];
        const defaultData = [130, 195, 160, 180, 260, 220];

        const actualLabels = labels || defaultLabels;
        const actualData = data || defaultData;

        this.temporalChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: actualLabels,
                datasets: [{
                    label: yAxisLabel,
                    data: actualData,
                    borderColor: '#4facfe',
                    backgroundColor: 'rgba(79, 172, 254, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#4facfe',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#4a5568',
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.parsed.y;
                                const total = actualData.reduce((a, b) => a + b, 0);
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                return `${yAxisLabel}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: yAxisLabel,
                            color: '#4a5568',
                            font: {
                                size: 13,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: '#e0e4e7'
                        },
                        ticks: {
                            color: '#718096',
                            precision: 0
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: xAxisLabel,
                            color: '#4a5568',
                            font: {
                                size: 13,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#718096',
                            maxRotation: 45,
                            minRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 10
                        }
                    }
                },
                elements: {
                    point: {
                        hoverRadius: 8
                    }
                }
            }
        });
    }

    createDiagnosticsChart(labels = null, data = null) {
        const ctx = document.getElementById('diagnosticsChart');
        if (!ctx) return;

        // Destruir gráfica anterior si existe
        if (this.diagnosticsChartInstance) {
            this.diagnosticsChartInstance.destroy();
        }

        const defaultLabels = ['Cardiología', 'Diabetes', 'Neurología', 'Oncología', 'Otros'];
        const defaultData = [35, 25, 20, 15, 5];

        const actualLabels = labels || defaultLabels;
        const actualData = data || defaultData;

        // Calcular total para porcentajes
        const total = actualData.reduce((a, b) => a + b, 0);

        this.diagnosticsChartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: actualLabels,
                datasets: [{
                    data: actualData,
                    backgroundColor: [
                        '#43e97b',
                        '#f093fb',
                        '#4facfe',
                        '#f6ad55',
                        '#fc8181',
                        '#ffd700',
                        '#ff69b4',
                        '#00ced1'
                    ],
                    borderWidth: 2,
                    borderColor: '#ffffff',
                    hoverBorderWidth: 3,
                    hoverBorderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: {
                        right: 10 // Más espacio a la derecha para la leyenda
                    }
                },
                plugins: {
                    legend: {
                        position: 'right',
                        align: 'start', // Alinear arriba para mejor distribución
                        maxWidth: 180, // Limitar ancho máximo de la leyenda
                        labels: {
                            padding: 6, // Reducir más el padding entre labels
                            usePointStyle: true,
                            color: '#4a5568',
                            font: {
                                size: 9.5, // Reducir más el tamaño de fuente
                                weight: '500'
                            },
                            boxWidth: 8,
                            boxHeight: 8,
                            generateLabels: function(chart) {
                                const data = chart.data;
                                if (data.labels.length && data.datasets.length) {
                                    return data.labels.map((label, i) => {
                                        const value = data.datasets[0].data[i];
                                        const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                        // Truncar labels muy largos - ajustar según ancho de pantalla
                                        const windowWidth = window.innerWidth;
                                        let maxLength = 12; // Por defecto para pantallas estrechas
                                        if (windowWidth > 1400) maxLength = 18;
                                        else if (windowWidth > 1200) maxLength = 15;
                                        else if (windowWidth > 992) maxLength = 13;
                                        
                                        const shortLabel = label.length > maxLength ? label.substring(0, maxLength - 2) + '...' : label;
                                        return {
                                            text: `${shortLabel} (${percentage}%)`,
                                            fillStyle: data.datasets[0].backgroundColor[i],
                                            hidden: false,
                                            index: i
                                        };
                                    });
                                }
                                return [];
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                                return `${label}: ${value} registros (${percentage}%)`;
                            }
                        }
                    }
                },
                cutout: '65%'
            }
        });
    }

    updateInitialCharts(stats) {
        if (!stats) {
            console.log('⚠️ updateInitialCharts called with no stats');
            return;
        }
        
        console.log('📊 Updating initial charts with stats:', stats);
        console.log('📊 Stats keys:', Object.keys(stats));
        console.log('📊 Numeric columns:', stats.numeric_columns);
        console.log('📊 Categorical columns:', stats.categorical_columns);
        
        // 1. Actualizar gráfica temporal - Buscar la mejor visualización disponible
        this.updateTemporalChart(stats);
        
        // 2. Actualizar gráfica de diagnósticos/target
        this.updateDiagnosticsChart(stats);
    }

    // Helper function para detectar columnas tipo ID
    isIdColumn(columnName) {
        if (!columnName) return false;
        const lowerName = columnName.toLowerCase().trim();
        
        // Patrones comunes de columnas ID
        const idPatterns = [
            'id',
            '_id',
            'patient_id',
            'patient id',
            'patientid',
            'identificador',
            'identifier',
            'codigo',
            'code',
            'num_paciente',
            'numero_paciente',
            'nro_paciente',
            'numero',
            'cedula',
            'dni',
            'document',
            'documento'
        ];
        
        // Verificar si el nombre de la columna coincide con algún patrón
        return idPatterns.some(pattern => {
            // Coincidencia exacta
            if (lowerName === pattern) return true;
            // Comienza con el patrón
            if (lowerName.startsWith(pattern + '_') || lowerName.startsWith(pattern + ' ')) return true;
            // Termina con el patrón
            if (lowerName.endsWith('_' + pattern) || lowerName.endsWith(' ' + pattern)) return true;
            // Contiene el patrón como palabra completa
            if (lowerName === pattern || lowerName.includes('_' + pattern + '_') || lowerName.includes(' ' + pattern + ' ')) return true;
            return false;
        });
    }

    updateTemporalChart(stats) {
        try {
            let labels = null;
            let data = null;
            let chartType = 'unknown';
            let selectedCol = null;
            
            console.log('🔍 updateTemporalChart - Analyzing stats...');
            
            // Prioridad de visualización:
            // 1. Si hay columnas numéricas, mostrar distribución de la más variable
            // 2. Si hay columnas categóricas, mostrar frecuencia de la más diversa
            // 3. Fallback: usar histogramas si están disponibles
            
            if (stats.numeric_columns && Object.keys(stats.numeric_columns).length > 0) {
                console.log('📊 Found numeric_columns:', Object.keys(stats.numeric_columns));
                // Buscar la columna numérica con mayor variación (std más alto)
                const numericCols = stats.numeric_columns;
                let maxStd = 0;
                
                for (const [colName, colStats] of Object.entries(numericCols)) {
                    // 🔒 EXCLUIR columnas tipo ID
                    if (this.isIdColumn(colName)) {
                        console.log(`⚠️ Skipping ID column: ${colName}`);
                        continue;
                    }
                    
                    if (colStats.std && colStats.std > maxStd) {
                        maxStd = colStats.std;
                        selectedCol = colName;
                    }
                }
                
                if (selectedCol && numericCols[selectedCol].distribution) {
                    const dist = numericCols[selectedCol].distribution;
                    labels = dist.bins || Object.keys(dist);
                    data = dist.counts || Object.values(dist);
                    chartType = 'numeric_distribution';
                    console.log(`✅ Using numeric distribution from column: ${selectedCol}`);
                } else if (maxStd === 0) {
                    console.log('⚠️ No suitable numeric columns found (all might be ID columns)');
                }
            }
            
            if (!labels && stats.categorical_columns && Object.keys(stats.categorical_columns).length > 0) {
                console.log('📊 Found categorical_columns:', Object.keys(stats.categorical_columns));
                // Buscar columna categórica con mayor número de categorías únicas (más diversidad)
                const catCols = stats.categorical_columns;
                let maxUnique = 0;
                
                for (const [colName, colStats] of Object.entries(catCols)) {
                    // 🔒 EXCLUIR columnas tipo ID
                    if (this.isIdColumn(colName)) {
                        console.log(`⚠️ Skipping ID column: ${colName}`);
                        continue;
                    }
                    
                    if (colStats.unique_count && colStats.unique_count > maxUnique && colStats.unique_count <= 20) {
                        maxUnique = colStats.unique_count;
                        selectedCol = colName;
                    }
                }
                
                if (selectedCol && catCols[selectedCol].value_counts) {
                    const valueCounts = catCols[selectedCol].value_counts;
                    const entries = Object.entries(valueCounts).slice(0, 10); // Top 10
                    labels = entries.map(e => String(e[0]));
                    data = entries.map(e => e[1]);
                    chartType = 'categorical_frequency';
                    console.log(`✅ Using categorical frequency from column: ${selectedCol}`);
                } else if (maxUnique === 0) {
                    console.log('⚠️ No suitable categorical columns found (all might be ID columns)');
                }
            }
            
            // Fallback: Intentar usar histogramas antiguos
            if (!labels && stats.histograms && stats.histograms.length > 0) {
                console.log('📊 Using fallback histograms');
                const hist = stats.histograms[0];
                labels = hist.bins;
                data = hist.counts;
                chartType = 'histogram_fallback';
                selectedCol = hist.column;
            }
            
            // Si aún no hay datos, no actualizar
            if (!labels || !data || labels.length === 0) {
                console.log('⚠️ No suitable data for temporal chart - keeping default');
                return;
            }
            
            console.log(`✅ Updating temporal chart with ${labels.length} data points`);
            
            // Limitar a máximo 10 puntos para evitar saturación
            if (labels.length > 10) {
                console.log(`⚠️ Limiting from ${labels.length} to 10 data points`);
                labels = labels.slice(0, 10);
                data = data.slice(0, 10);
            }
            
            // Determinar los labels de los ejes
            let yAxisLabel = 'Frecuencia';
            let xAxisLabel = 'Categorías';
            
            if (chartType === 'numeric_distribution') {
                yAxisLabel = 'Cantidad de registros';
                xAxisLabel = 'Rangos de valores';
            } else if (chartType === 'categorical_frequency') {
                yAxisLabel = 'Frecuencia';
                xAxisLabel = 'Categorías';
            }
            
            // Actualizar la gráfica
            this.createTemporalChart(labels, data, yAxisLabel, xAxisLabel);
            
            // Actualizar el título de la gráfica para que sea descriptivo
            const chartTitle = document.getElementById('temporal-chart-title');
            if (chartTitle) {
                if (chartType === 'numeric_distribution') {
                    chartTitle.textContent = `Distribución: ${selectedCol}`;
                } else if (chartType === 'categorical_frequency') {
                    chartTitle.textContent = `Frecuencia: ${selectedCol}`;
                } else if (chartType === 'histogram_fallback') {
                    chartTitle.textContent = `Distribución: ${selectedCol}`;
                } else {
                    chartTitle.textContent = 'Distribución de datos';
                }
            }
        } catch (e) {
            console.error('❌ Error updating temporal chart:', e);
        }
    }

    updateDiagnosticsChart(stats) {
        try {
            let labels = null;
            let data = null;
            let selectedCol = null;
            
            console.log('🔍 updateDiagnosticsChart - Analyzing stats...');
            
            // Buscar el target o la columna más relevante
            const target = stats.target && stats.target.selected;
            
            // Intentar primero con el target
            if (target && stats.categorical_columns && stats.categorical_columns[target]) {
                const valueCounts = stats.categorical_columns[target].value_counts;
                if (valueCounts && Object.keys(valueCounts).length > 0) {
                    const entries = Object.entries(valueCounts);
                    labels = entries.map(e => String(e[0]));
                    data = entries.map(e => e[1]);
                    selectedCol = target;
                    console.log(`✅ Using target distribution: ${target} (${labels.length} categories)`);
                }
            }
            
            // Si no hay target o el target no tiene datos, buscar otra columna
            if (!labels && stats.categorical_columns && Object.keys(stats.categorical_columns).length > 0) {
                console.log('📊 Searching for best categorical column...');
                const catCols = stats.categorical_columns;
                let bestScore = 0;
                
                for (const [colName, colStats] of Object.entries(catCols)) {
                    const uniqueCount = colStats.unique_count || 0;
                    // Score: preferir columnas con 2-15 valores únicos
                    if (uniqueCount >= 2 && uniqueCount <= 15) {
                        const score = uniqueCount;
                        if (score > bestScore) {
                            bestScore = score;
                            selectedCol = colName;
                        }
                    }
                }
                
                if (selectedCol && catCols[selectedCol].value_counts) {
                    const valueCounts = catCols[selectedCol].value_counts;
                    const entries = Object.entries(valueCounts);
                    labels = entries.map(e => String(e[0]));
                    data = entries.map(e => e[1]);
                    console.log(`✅ Using categorical column: ${selectedCol} (${labels.length} categories)`);
                }
            }
            
            // Si no hay datos reales, no actualizar
            if (!labels || !data || labels.length === 0) {
                console.log('⚠️ No suitable data for diagnostics chart - keeping default');
                return;
            }
            
            // 🔥 Limitar a máximo 6 categorías para evitar saturación visual
            const MAX_CATEGORIES = 6;
            if (labels.length > MAX_CATEGORIES) {
                console.log(`⚠️ Too many categories (${labels.length}), limiting to top ${MAX_CATEGORIES}`);
                // Ordenar por valor descendente
                const combined = labels.map((label, i) => ({ label, value: data[i] }));
                combined.sort((a, b) => b.value - a.value);
                
                // Tomar las top (MAX_CATEGORIES - 1) y agrupar el resto en "Otros"
                const topN = combined.slice(0, MAX_CATEGORIES - 1);
                const others = combined.slice(MAX_CATEGORIES - 1);
                const othersSum = others.reduce((sum, item) => sum + item.value, 0);
                
                labels = topN.map(item => item.label);
                data = topN.map(item => item.value);
                
                if (othersSum > 0) {
                    labels.push('Otros');
                    data.push(othersSum);
                }
            }
            
            // 🔥 Para datos binarios (0/1, Yes/No, etc.), formatear labels mejor
            if (labels.length === 2) {
                const mappings = {
                    '0': 'No (0)',
                    '1': 'Sí (1)',
                    'true': 'Verdadero',
                    'false': 'Falso',
                    'yes': 'Sí',
                    'no': 'No',
                    '0.0': 'No (0)',
                    '1.0': 'Sí (1)'
                };
                labels = labels.map(label => {
                    const lower = String(label).toLowerCase().trim();
                    return mappings[lower] || label;
                });
                console.log(`✅ Binary data detected - formatted labels:`, labels);
            }
            
            console.log(`✅ Updating diagnostics chart with ${labels.length} categories`);
            
            // Actualizar la gráfica
            this.createDiagnosticsChart(labels, data);
            
            // Actualizar el título
            const chartTitle = document.getElementById('diagnostics-chart-title');
            if (chartTitle) {
                if (selectedCol) {
                    chartTitle.textContent = `Distribución: ${selectedCol}`;
                } else {
                    chartTitle.textContent = 'Distribución de categorías';
                }
            }
        } catch (e) {
            console.error('❌ Error updating diagnostics chart:', e);
        }
    }

    // Target selection helpers
    getTargetSelection(datasetId) {
        try {
            const target = this.targetSelections && datasetId ? this.targetSelections[datasetId] : null;
            if (target) {
                console.log(`🎯 Retrieved saved target for dataset ${datasetId}:`, target);
            }
            return target;
        } catch (e) {
            console.error('❌ Error retrieving target selection:', e);
            return null;
        }
    }
    
    setTargetSelection(datasetId, col) {
        try {
            if (!datasetId || !col) {
                console.warn('⚠️ Cannot set target: missing datasetId or column');
                return;
            }
            this.targetSelections = this.targetSelections || {};
            this.targetSelections[datasetId] = col;
            localStorage.setItem('targetSelections', JSON.stringify(this.targetSelections));
            console.log(`✅ Target saved for dataset ${datasetId}:`, col);
            console.log('📦 Full targetSelections:', this.targetSelections);
        } catch (e) {
            console.error('❌ Error saving target selection:', e);
        }
    }

    async promptTargetSelection(candidates) {
        if (!Array.isArray(candidates) || !candidates.length) return null;
        
        // Check if already prompted for this dataset to avoid loops
        const promptKey = `target_prompted_${this.currentDatasetId}`;
        if (sessionStorage.getItem(promptKey)) {
            // Already prompted in this session, just return first candidate
            return candidates[0];
        }
        
        const options = candidates.map((c, i) => `
            <label style="display:flex; gap:8px; align-items:center; margin-bottom:6px;">
                <input type="radio" name="target_col" value="${c}" ${i===0?'checked':''}/> ${c}
            </label>
        `).join('');
        const bodyHtml = `<form id="target-form">
            <div style="margin-bottom:8px; color:#4a5568;">Selecciona la columna objetivo (target)</div>
            ${options}
        </form>`;
        
        // Mark as prompted
        sessionStorage.setItem(promptKey, 'true');
        
        const res = await this.showModal({ title: 'Seleccionar Target', bodyHtml });
        const selected = res && res.target_col ? res.target_col : candidates[0];
        if (selected) this.setTargetSelection(this.currentDatasetId, selected);
        return selected;
    }

    setupEventListeners() {
        if (this._eventsWired) return; // prevent duplicate wiring
        this._eventsWired = true;
        this.setupChatInput();
        // New: wire sidebar and upload
        this.setupUploadButton();
        this.setupSidebarNavigation();
        // Responsive sidebar toggle
        this.setupResponsiveBehavior();
        // Modal wiring
        this.setupModal();
        // Select target button
        const selectBtn = document.getElementById('select-target-btn');
        if (selectBtn) {
            selectBtn.addEventListener('click', async () => {
                if (!this.currentDatasetId) { this.showNotification('Primero selecciona un dataset', 'warning'); return; }
                try {
                    const res = await fetch(`${this.apiBase}/datasets/${this.currentDatasetId}/columns`);
                    const json = await res.json();
                    if (!res.ok || !json.success) throw new Error(json.error || json.message || 'No se pudieron obtener columnas');
                    const cols = (json.data && json.data.columns && Array.isArray(json.data.columns.columns)) ? json.data.columns.columns : (json.data && json.data.columns ? Object.keys(json.data.columns) : []);
                    if (!cols.length) { this.showNotification('No hay columnas para seleccionar', 'warning'); return; }
                    const opts = cols.map((c, i) => `<option value="${c}">${c}</option>`).join('');
                    const bodyHtml = `<form id="pick-target"><div style="margin-bottom:8px; color:#4a5568;">Elegir cualquier columna como target</div><select name="target_col" style="width:100%; padding:8px 10px; border:1px solid #e2e8f0; border-radius:8px;">${opts}</select></form>`;
                    const resSel = await this.showModal({ title: 'Seleccionar Target', bodyHtml });
                    const chosen = resSel && resSel.target_col ? resSel.target_col : null;
                    if (chosen) {
                        console.log('🎯 User selected target:', chosen);
                        this.setTargetSelection(this.currentDatasetId, chosen);
                        
                        // Force immediate KPI update
                        const statNumbers = document.querySelectorAll('.stat-number');
                        console.log('🔍 Immediate update: Found', statNumbers.length, 'stat elements');
                        if (statNumbers.length >= 4) {
                            const targetEl = statNumbers[3];
                            const oldValue = targetEl.textContent;
                            targetEl.textContent = chosen;
                            targetEl.setAttribute('data-value', chosen);
                            // Force repaint
                            targetEl.style.display = 'none';
                            targetEl.offsetHeight; // Trigger reflow
                            targetEl.style.display = '';
                            console.log(`✅ Immediately updated Target KPI: "${oldValue}" → "${chosen}"`);
                            console.log('🔍 Target element:', targetEl);
                        } else {
                            console.warn('⚠️ Not enough stat elements found:', statNumbers.length);
                        }
                        
                        // Then refresh all stats from backend
                        await this.updateStatsFromAPI(true); // Skip prompt to avoid re-showing popup
                        this.showNotification(`Target seleccionado: ${chosen}`, 'success');
                    }
                } catch (e) {
                    this.showNotification(`Error cargando columnas: ${e.message || e}`, 'error');
                }
            });
        }
    }

    // Upload wiring: open file chooser and handle upload
    setupUploadButton() {
        const uploadBtn = document.querySelector('#upload-btn');
        const uploadSidebarBtn = document.querySelector('#upload-sidebar-btn');
        const fileInput = document.querySelector('#file-upload');

        if (uploadBtn && fileInput) {
            uploadBtn.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', async (e) => {
                const file = e.target.files && e.target.files[0];
                if (file) await this.handleFileUpload(file);
                fileInput.value = '';
            });
        }
        if (uploadSidebarBtn && fileInput) {
            uploadSidebarBtn.addEventListener('click', () => uploadBtn ? uploadBtn.click() : fileInput.click());
        }
    }

    // Replace notifications with Notyf
    showNotification(message, type = 'success') {
        switch(type) {
            case 'success':
                this.notyf.success(message);
                break;
            case 'error':
                this.notyf.error(message);
                break;
            case 'info':
                this.notyf.open({ type: 'info', message });
                break;
            case 'warning':
                this.notyf.open({ type: 'warning', message });
                break;
            default:
                this.notyf.success(message);
        }
    }

    async handleFileUpload(file) {
        try {
            this.showLoadingState();
            const form = new FormData();
            form.append('file', file, file.name);
            const res = await fetch(`${this.apiBase}/datasets/upload`, { method: 'POST', body: form });
            const json = await res.json();
            if (!res.ok || !json.success) throw new Error(json.error || json.message || 'Error al subir');

            const info = json.data && (json.data.dataset_info || json.data.dataset || json.data.info);
            this.currentDatasetId = info && (info.id || info.dataset_id);
            try { if (this.currentDatasetId) localStorage.setItem('currentDatasetId', String(this.currentDatasetId)); } catch {}
            this.showNotification(`Archivo ${info && info.filename ? info.filename : file.name} cargado exitosamente`, 'success');
            // Refresh dataset list and load preview + stats
            await this.loadDatasetsList();
            this.highlightActiveDataset(this.currentDatasetId);
            await this.loadDatasetPreview();
        } catch (err) {
            console.error(err);
            this.showNotification(`Error al subir dataset: ${err.message || err}`, 'error');
        } finally {
            this.hideLoadingState();
        }
    }

    async loadDatasetPreview(rows = 10) {
        if (!this.currentDatasetId) return;
        try {
            const res = await fetch(`${this.apiBase}/datasets/${this.currentDatasetId}/preview?rows=${rows}`);
            const json = await res.json();
            if (!res.ok || !json.success) throw new Error(json.error || json.message || 'Error obteniendo preview');
            // API returns data.preview as an object with { preview: [...], summary: {...} }
            const dataPreview = json.data && json.data.preview;
            const records = (dataPreview && (Array.isArray(dataPreview) ? dataPreview : dataPreview.preview)) || [];
            this.renderDatasetPreview(records);
            // Load real statistics from API instead of computing from preview
            await this.updateStatsFromAPI();
        } catch (err) {
            console.error(err);
            this.showNotification(`Error cargando preview: ${err.message || err}`, 'error');
        }
    }

    async updateStatsFromAPI(skipPrompt = false) {
        if (!this.currentDatasetId) return;
        try {
            const res = await fetch(`${this.apiBase}/datasets/${this.currentDatasetId}/statistics`);
            const json = await res.json();
            if (!res.ok || !json.success) throw new Error(json.error || json.message || 'Error estadísticas');
            const stats = json.data && json.data.statistics;
            
            // Apply user-selected target if available (highest priority)
            const savedTarget = this.getTargetSelection(this.currentDatasetId);
            let target = 'Sin definir';
            
            if (savedTarget) {
                // User manually selected a target - use it
                target = savedTarget;
                // Also update stats object for consistency
                if (stats) {
                    stats.target = stats.target || {};
                    stats.target.selected = savedTarget;
                }
                console.log('✅ Using saved target from localStorage:', savedTarget);
            } else if (stats && stats.target && stats.target.selected) {
                // Backend provided a target
                target = stats.target.selected;
                console.log('Using target from backend:', target);
            }
            
            // Order: Pacientes, Edad media, Nulos, Target
            const vals = [
                stats && stats.overview ? (stats.overview.patients ?? '-') : '-',
                stats && stats.overview && stats.overview.average_age != null ? Math.round(stats.overview.average_age) : '-',
                stats && stats.overview ? (stats.overview.total_nulls ?? '-') : '-',
                target
            ];
            
            const statNumbers = document.querySelectorAll('.stat-number');
            console.log('🔍 Found stat elements:', statNumbers.length);
            console.log('📊 Values to update:', vals);
            
            statNumbers.forEach((el, i) => { 
                const newVal = String(vals[i] ?? '-');
                const oldVal = el.textContent;
                
                // Force update - always set the value
                el.textContent = newVal;
                el.setAttribute('data-value', newVal);
                
                console.log(`📊 KPI ${i}: "${oldVal}" → "${newVal}" (element exists: ${!!el})`);
                
                // Force repaint
                el.style.display = 'none';
                el.offsetHeight; // Trigger reflow
                el.style.display = '';
            });
            
            // Show hint if target is not defined (only once per dataset)
            if (target === 'Sin definir' && !skipPrompt) {
                const hintKey = `target_hint_shown_${this.currentDatasetId}`;
                if (!sessionStorage.getItem(hintKey)) {
                    sessionStorage.setItem(hintKey, 'true');
                    setTimeout(() => {
                        this.showNotification('💡 Puedes seleccionar un target usando el botón "Seleccionar Target"', 'info');
                    }, 1000);
                }
            }
            
            // Store stats for charts
            this.lastStats = stats || {};
            
            // 🆕 Actualizar las gráficas iniciales con datos reales
            this.updateInitialCharts(stats);
            
            // If charts are enabled, render them in the dedicated section
            if (this.includeChartsFlag) this.renderAnalysisCharts();
        } catch (e) {
            console.error('Error updating KPIs:', e);
            this.showNotification(`Error cargando estadísticas: ${e.message || e}`, 'error');
        }
    }

    renderDatasetPreview(records) {
        const container = document.getElementById('dataset-preview');
        if (!container) return;
        const rows = Array.isArray(records) ? records.slice(0, 10) : [];
        if (!rows.length) {
            container.innerHTML = '<p>No hay vista previa disponible.</p>';
            return;
        }
        const cols = Object.keys(rows[0]);
        const header = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>`;
        const body = `<tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${r[c] ?? ''}</td>`).join('')}</tr>`).join('')}</tbody>`;
        container.innerHTML = `<div class="table-wrapper"><table class="preview-table">${header}${body}</table></div>`;
    }

    // Load dataset list from API and render sidebar entries
    async loadDatasetsList() {
        try {
            const list = document.getElementById('dataset-list');
            if (!list) return;
            const res = await fetch(`${this.apiBase}/datasets`);
            const json = await res.json();
            if (!res.ok || !json.success) throw new Error(json.error || json.message || 'Error listando datasets');
            const datasets = (json.data && json.data.datasets) || [];
            list.innerHTML = '';
            if (!datasets.length) {
                const empty = document.createElement('li');
                empty.className = 'dataset-list-item';
                empty.innerHTML = '<span style="color:#718096">No hay datasets cargados.</span>';
                list.appendChild(empty);
                return;
            }
            datasets.forEach(ds => this.updateDatasetList({ id: ds.id, filename: ds.filename }));
            // If there's a current dataset, ensure it is selected in UI
            if (this.currentDatasetId) {
                this.highlightActiveDataset(this.currentDatasetId);
            }
        } catch (e) {
            console.error('Error loading datasets:', e);
        }
    }

    highlightActiveDataset(datasetId) {
        try {
            const list = document.getElementById('dataset-list');
            if (!list) return;
            list.querySelectorAll('li[data-id]').forEach(li => {
                if (li.getAttribute('data-id') === String(datasetId)) {
                    li.classList.add('active');
                } else {
                    li.classList.remove('active');
                }
            });
        } catch {}
    }

    // 🆕 Obtener el nombre del dataset por su ID
    getDatasetNameById(datasetId) {
        try {
            const list = document.getElementById('dataset-list');
            if (!list || !datasetId) return null;
            const item = list.querySelector(`li[data-id="${datasetId}"]`);
            if (!item) return null;
            // Extraer el texto del span (ignorar el botón de eliminar)
            const span = item.querySelector('span');
            return span ? span.textContent.trim() : null;
        } catch {
            return null;
        }
    }

    // Add/refresh dataset sidebar entry and click handler
    updateDatasetList(info) {
        try {
            const list = document.getElementById('dataset-list');
            if (!list || !info || !info.id) return;
            // Remove placeholder empty items
            list.querySelectorAll('li').forEach(li => {
                if (!li.getAttribute('data-id') && li.textContent && li.textContent.includes('No hay datasets')) li.remove();
            });
            // Remove existing entry with same id
            list.querySelectorAll('li[data-id]').forEach(li => { if (li.getAttribute('data-id') === String(info.id)) li.remove(); });
            const li = document.createElement('li');
            li.className = 'dataset-list-item';
            li.setAttribute('data-id', String(info.id));
            li.title = info.filename || info.id;
            // Add delete button
            li.innerHTML = `<i class="fas fa-database"></i><span style="flex:1">${(info.filename || info.id)}</span><button class="btn-delete" title="Eliminar"><i class="fas fa-trash"></i></button>`;
            li.querySelector('.btn-delete').addEventListener('click', async (e) => {
                e.stopPropagation();
                await this.deleteDataset(info.id);
            });
            li.addEventListener('click', async () => {
                const oldDatasetId = this.currentDatasetId;
                this.currentDatasetId = String(info.id);
                try { localStorage.setItem('currentDatasetId', this.currentDatasetId); } catch {}
                
                // Clear session flags if switching to a different dataset
                if (oldDatasetId !== this.currentDatasetId) {
                    try {
                        sessionStorage.removeItem(`target_prompted_${this.currentDatasetId}`);
                        sessionStorage.removeItem(`target_hint_shown_${this.currentDatasetId}`);
                    } catch {}
                }
                
                this.highlightActiveDataset(this.currentDatasetId);
                await this.loadDatasetPreview(10);
                this.addChatMessage(`Dataset activo: ${(info.filename || info.id)}`, 'assistant');
            });
            list.prepend(li);
        } catch (e) { /* noop */ }
    }

    async deleteDataset(datasetId) {
        try {
            const res = await fetch(`${this.apiBase}/datasets/${datasetId}`, { method: 'DELETE' });
            const json = await res.json();
            if (!res.ok || !json.success) throw new Error(json.error || json.message || 'No se pudo eliminar');
            this.showNotification('Dataset eliminado', 'success');
            
            // Clean up session storage for this dataset
            try {
                sessionStorage.removeItem(`target_prompted_${datasetId}`);
                sessionStorage.removeItem(`target_hint_shown_${datasetId}`);
            } catch {}
            
            // If deleted current, clear state
            if (String(this.currentDatasetId) === String(datasetId)) {
                this.currentDatasetId = null;
                try { localStorage.removeItem('currentDatasetId'); } catch {}
                // Clear preview and KPIs
                const container = document.getElementById('dataset-preview');
                if (container) container.innerHTML = '<p>No hay vista previa disponible.</p>';
                const statNumbers = document.querySelectorAll('.stat-number');
                statNumbers.forEach((el) => { el.textContent = '-'; });
            }
            // Refresh list
            await this.loadDatasetsList();
        } catch (e) {
            this.showNotification(`Error eliminando dataset: ${e.message || e}`, 'error');
        }
    }

    updateStatsFromPreview(records) {
        try {
            const statNumbers = document.querySelectorAll('.stat-number');
            const totals = {
                patients: records.length || '-',
                age: '-',
                events: '-',
                records: records.length || '-'
            };
            // Simple heuristic for age if present
            const first = records[0] || {};
            const ageKey = Object.keys(first).find(k => /edad|age/i.test(k));
            if (ageKey) {
                const ages = records.map(r => Number(r[ageKey])).filter(v => !isNaN(v));
                if (ages.length) totals.age = Math.round(ages.reduce((a,b)=>a+b,0)/ages.length);
            }
            // Apply to UI (order: Pacientes, Edad media, Eventos, Registros)
            const vals = [totals.patients, totals.age, totals.events, totals.records];
            statNumbers.forEach((el, i) => { el.textContent = String(vals[i] ?? '-'); });
        } catch {}
    }

    // Sidebar navigation -> trigger API calls
    setupSidebarNavigation() {
        const sidebarItems = document.querySelectorAll('.sidebar-item');
        sidebarItems.forEach(item => {
            item.addEventListener('click', async () => {
                const label = item.textContent.trim();
                if (label.includes('Analizar')) {
                    await this.callAnalyze();
                } else if (label.includes('Generar')) {
                    await this.callGeneration();
                } else if (label.includes('Evaluar')) {
                    await this.callEvaluation();
                } else if (label.toLowerCase().includes('validar')) {
                    await this.callValidate();
                } else if (label.includes('Simular')) {
                    await this.callSimulation();
                }
            });
        });
    }

    async callAnalyze() {
        if (!this.currentDatasetId) return this.showNotification('Sube un dataset primero', 'warning');
        // Popup: analysis configuration (SIMPLIFIED - solo tipo de análisis)
        const bodyHtml = `
            <form id="analysis-form">
                <div style="display:flex; gap:12px; align-items:center; margin-bottom:10px;">
                    <label style="min-width:160px; color:#4a5568;">Tipo de análisis</label>
                    <select name="analysis_type" style="flex:1; padding:8px 10px; border:1px solid #e2e8f0; border-radius:8px;">
                        <option value="basic" selected>Básico (rápido, resumen general)</option>
                        <option value="comprehensive">Integral (detallado, análisis completo)</option>
                    </select>
                </div>
                <div style="margin-top:12px; padding:10px; background:#f7fafc; border-radius:6px; font-size:12px; color:#4a5568;">
                    ℹ️ El análisis se mostrará en formato texto en el panel central.
                </div>
            </form>`;
        const choice = await this.showModal({ title: 'Configurar análisis', bodyHtml });
        if (!choice) return; // cancelled
        this.includeChartsFlag = false; // Deshabilitado por ahora
        
        try {
            this.showLoadingState();
            const res = await fetch(`${this.apiBase}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_id: this.currentDatasetId, analysis_type: (choice && choice.analysis_type) || 'basic' })
            });
            const json = await res.json();
            console.log('📊 Analysis response:', json);
            
            // Extract response from various possible structures
            let reply = '';
            
            // Prioridad 1: Buscar en messages array (formato LangGraph)
            if (json.data && json.data.messages && Array.isArray(json.data.messages) && json.data.messages.length) {
                const lastMsg = json.data.messages[json.data.messages.length - 1];
                reply = lastMsg.message || lastMsg.content || lastMsg.response || '';
                console.log('✅ Respuesta extraída de messages array');
            }
            // Prioridad 2: Buscar en analysis_response
            else if (json.data && json.data.analysis_response) {
                reply = json.data.analysis_response.message || 
                       json.data.analysis_response.response || 
                       json.data.analysis_response.answer || '';
                console.log('✅ Respuesta extraída de analysis_response');
            }
            // Prioridad 3: Buscar directamente en data
            else if (json.data) {
                reply = json.data.message || json.data.response || json.data.answer || '';
                console.log('✅ Respuesta extraída de data');
            }
            // Prioridad 4: Buscar en nivel raíz
            else if (json.message) {
                reply = json.message;
                console.log('✅ Respuesta extraída de message raíz');
            }
            
            // Verificar si es un error REAL (no solo menciones de "error")
            const isRealError = !reply || 
                               reply.toLowerCase().includes('no se encontró') ||
                               reply.toLowerCase().includes('no se pudo') ||
                               reply.toLowerCase().includes('falló') ||
                               (reply.toLowerCase().includes('error') && reply.length < 100);
            
            if (isRealError) {
                console.error('❌ Error en análisis:', reply || 'Sin respuesta');
                reply = reply || '⚠️ El análisis se completó pero no se recibió resultado. Por favor, intenta de nuevo.';
            }
            
            console.log('📝 Respuesta final (primeros 200 chars):', reply.substring(0, 200));
            
            // Mostrar resultado en panel central
            const agentName = (json.data && json.data.messages && json.data.messages.length && json.data.messages[json.data.messages.length - 1].agent) ||
                            (json.data && json.data.analysis_response && json.data.analysis_response.agent) || 
                            (json.data && json.data.agent) || 'Analizador Clínico';
            
            const analysisType = (choice && choice.analysis_type) || 'basic';
            const analysisTypeLabel = analysisType === 'comprehensive' ? 'integral' : 'básico';
            
            // Mostrar resultado en panel central
            this.renderAgentResultCard({
                title: '📊 Análisis del Dataset',
                subtitle: `Agente: ${agentName} | Tipo: ${analysisTypeLabel}`,
                contentMarkdown: reply,
                footer: `📅 ${new Date().toLocaleString()}`,
                type: 'analysis'
            });
            
            // Notificar en chat con contexto
            if (isRealError) {
                this.addChatMessage('⚠️ Análisis completado con errores. Revisa el panel central.', 'assistant');
                this.showNotification('Análisis completado con advertencias', 'warning');
            } else {
                this.addChatMessage('✅ Análisis completado. Resultado disponible en el panel central.', 'assistant');
                this.addChatMessage('💬 Ahora puedes hacerme preguntas sobre el análisis, por ejemplo:', 'assistant');
                this.addChatMessage('  • "¿Cuál es el porcentaje de nulos?"', 'system');
                this.addChatMessage('  • "¿Qué dominio médico detectaste?"', 'system');
                this.addChatMessage('  • "Dame un resumen del análisis"', 'system');
                this.showNotification('Análisis completado exitosamente', 'success');
            }
        } catch (e) {
            this.showNotification(`Error analizando: ${e.message || e}`, 'error');
        } finally { this.hideLoadingState(); }
    }

    async callGeneration() {
        if (!this.currentDatasetId) return this.showNotification('Sube un dataset primero', 'warning');
        
        // Cargar columnas del dataset antes de mostrar el modal
        try {
            const res = await fetch(`${this.apiBase}/datasets/${this.currentDatasetId}/columns`);
            const json = await res.json();
            if (!res.ok || !json.success) throw new Error(json.error || json.message || 'No se pudieron obtener columnas');
            
            const cols = (json.data && json.data.columns && Array.isArray(json.data.columns.columns)) 
                ? json.data.columns.columns 
                : (json.data && json.data.columns ? Object.keys(json.data.columns) : []);
            
            if (!cols.length) {
                this.showNotification('No hay columnas disponibles', 'warning');
                return;
            }
            
            // Mostrar modal de configuración con columnas
            this.showGenerationModal(cols);
        } catch (e) {
            this.showNotification(`Error cargando columnas: ${e.message || e}`, 'error');
        }
    }

    showGenerationModal(columns = []) {
        const modalBody = document.getElementById('modal-body');
        const modalTitle = document.getElementById('modal-title');
        
        modalTitle.textContent = 'Generar Datos Sintéticos';
        
        // Generar opciones de columnas
        const columnCheckboxes = columns.length > 0 ? `
            <div class="form-group">
                <label><strong>Seleccionar Columnas:</strong></label>
                <small class="form-text" style="display: block; margin-bottom: 8px;">
                    ${columns.length} columnas disponibles. Selecciona las más relevantes para la generación (recomendado: 10-20 columnas).
                </small>
                <div class="column-selector-controls" style="margin-bottom: 8px; display: flex; gap: 8px;">
                    <button type="button" class="btn-select-all" style="padding: 4px 10px; border: 1px solid #e2e8f0; border-radius: 4px; background: #f7fafc; cursor: pointer; font-size: 12px;">
                        ✓ Seleccionar Todas
                    </button>
                    <button type="button" class="btn-deselect-all" style="padding: 4px 10px; border: 1px solid #e2e8f0; border-radius: 4px; background: #f7fafc; cursor: pointer; font-size: 12px;">
                        ✗ Deseleccionar Todas
                    </button>
                    <button type="button" class="btn-select-numeric" style="padding: 4px 10px; border: 1px solid #e2e8f0; border-radius: 4px; background: #f7fafc; cursor: pointer; font-size: 12px;">
                        🔢 Solo Numéricas
                    </button>
                </div>
                <div class="column-selector" style="max-height: 300px; overflow-y: auto; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; background: #f7fafc;">
                    ${columns.map((col, idx) => `
                        <label class="column-checkbox-label" style="display: flex; align-items: center; gap: 8px; padding: 6px; margin-bottom: 4px; cursor: pointer; border-radius: 4px; transition: background 0.2s;" 
                               onmouseover="this.style.background='#edf2f7'" 
                               onmouseout="this.style.background='transparent'">
                            <input type="checkbox" name="selected_columns" value="${col}" 
                                   ${idx < 15 ? 'checked' : ''} 
                                   style="cursor: pointer;">
                            <span style="font-size: 13px; color: #4a5568;">${col}</span>
                        </label>
                    `).join('')}
                </div>
                <small class="form-text" id="column-count-info" style="display: block; margin-top: 8px; font-weight: 500;">
                    ${Math.min(15, columns.length)} columnas seleccionadas
                </small>
            </div>
        ` : '';
        
        modalBody.innerHTML = `
            <div class="generation-config">
                <div class="form-group">
                    <label for="model-select"><strong>Modelo de Generación:</strong></label>
                    <select id="model-select" class="form-control">
                        <option value="CTGAN">CTGAN - Redes GAN condicionales (recomendado para datos complejos)</option>
                        <option value="TVAE">TVAE - Autoencoder variacional (mejor para datos mixtos)</option>
                        <option value="SDV">SDV - Synthetic Data Vault (rápido, balanceado)</option>
                    </select>
                    <small class="form-text">
                        <strong>CTGAN:</strong> Genera datos complejos con buenas correlaciones.<br>
                        <strong>TVAE:</strong> Mejor para datos con tipos mixtos (numéricos + categóricos).<br>
                        <strong>SDV:</strong> Más rápido, bueno para prototipado.
                    </small>
                </div>
                
                <div class="form-group">
                    <label for="num-samples"><strong>Cantidad de Muestras:</strong></label>
                    <input type="number" id="num-samples" class="form-control" 
                           value="1000" min="100" max="10000" step="100">
                    <small class="form-text">Entre 100 y 10,000 registros sintéticos</small>
                </div>
                
                ${columnCheckboxes}
            </div>
        `;
        
        // Agregar event listeners para los botones de selección
        if (columns.length > 0) {
            setTimeout(() => {
                const updateCount = () => {
                    const checked = modalBody.querySelectorAll('input[name="selected_columns"]:checked').length;
                    const countInfo = document.getElementById('column-count-info');
                    if (countInfo) {
                        countInfo.textContent = `${checked} columnas seleccionadas`;
                        countInfo.style.color = checked === 0 ? '#e53e3e' : checked > 30 ? '#d69e2e' : '#38a169';
                    }
                };
                
                // Actualizar contador al cambiar checkboxes
                modalBody.querySelectorAll('input[name="selected_columns"]').forEach(cb => {
                    cb.addEventListener('change', updateCount);
                });
                
                // Botón seleccionar todas
                const btnSelectAll = modalBody.querySelector('.btn-select-all');
                if (btnSelectAll) {
                    btnSelectAll.addEventListener('click', () => {
                        modalBody.querySelectorAll('input[name="selected_columns"]').forEach(cb => cb.checked = true);
                        updateCount();
                    });
                }
                
                // Botón deseleccionar todas
                const btnDeselectAll = modalBody.querySelector('.btn-deselect-all');
                if (btnDeselectAll) {
                    btnDeselectAll.addEventListener('click', () => {
                        modalBody.querySelectorAll('input[name="selected_columns"]').forEach(cb => cb.checked = false);
                        updateCount();
                    });
                }
                
                // Botón seleccionar solo numéricas
                const btnSelectNumeric = modalBody.querySelector('.btn-select-numeric');
                if (btnSelectNumeric) {
                    btnSelectNumeric.addEventListener('click', () => {
                        // Patrones comunes para columnas numéricas
                        const numericPatterns = /age|edad|dias|days|temp|temperatura|sat|peso|weight|altura|height|fc|hr|ta|presion|pressure|glu|glucose|id|number|num|count|cantidad/i;
                        modalBody.querySelectorAll('input[name="selected_columns"]').forEach(cb => {
                            cb.checked = numericPatterns.test(cb.value);
                        });
                        updateCount();
                    });
                }
            }, 100);
        }
        
        // Mostrar modal
        const modalOverlay = document.getElementById('modal-overlay');
        modalOverlay.classList.remove('hidden');
        
        // Configurar botones
        const confirmBtn = document.getElementById('modal-confirm');
        const cancelBtn = document.getElementById('modal-cancel');
        const closeBtn = document.getElementById('modal-close');
        
        // Remover listeners anteriores
        const newConfirmBtn = confirmBtn.cloneNode(true);
        confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);
        
        newConfirmBtn.onclick = async () => {
            const modelType = document.getElementById('model-select').value;
            const numSamples = parseInt(document.getElementById('num-samples').value);
            
            // Obtener columnas seleccionadas
            const selectedColumns = Array.from(
                modalBody.querySelectorAll('input[name="selected_columns"]:checked')
            ).map(cb => cb.value);
            
            if (selectedColumns.length === 0) {
                this.showNotification('⚠️ Debes seleccionar al menos una columna', 'warning');
                return;
            }
            
            if (selectedColumns.length > 50) {
                this.showNotification('⚠️ Recomendamos seleccionar máximo 50 columnas para mejor rendimiento', 'warning');
            }
            
            modalOverlay.classList.add('hidden');
            await this.executeGeneration(modelType, numSamples, selectedColumns);
        };
        
        cancelBtn.onclick = () => modalOverlay.classList.add('hidden');
        closeBtn.onclick = () => modalOverlay.classList.add('hidden');
    }

    async executeGeneration(modelType, numSamples, selectedColumns = []) {
        try {
            this.showLoadingState();
            
            const columnInfo = selectedColumns.length > 0 
                ? ` (${selectedColumns.length} columnas)` 
                : '';
            this.showNotification(`Generando ${numSamples} registros con ${modelType}${columnInfo}...`, 'info');
            
            console.log('🎯 Generación iniciada:', {
                model: modelType,
                samples: numSamples,
                columns: selectedColumns.length,
                columnList: selectedColumns
            });
            
            const res = await fetch(`${this.apiBase}/generation/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    dataset_id: this.currentDatasetId, 
                    model_type: modelType.toLowerCase(), 
                    num_samples: numSamples, 
                    selected_columns: selectedColumns, 
                    parameters: {} 
                })
            });
            
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            
            const json = await res.json();
            const info = json.data || {};
            
            this.addChatMessage(`Generación completada: ${numSamples} registros con ${modelType}. Consulta el resultado en el panel central.`, 'assistant');
            
            // Render result in central panel
            const agentName = (json.data && json.data.agent) || 'generator';
            const reply = json.data && json.data.message ? json.data.message : 
                `# Datos Sintéticos Generados\n\n` +
                `**Modelo:** ${modelType}\n` +
                `**Registros:** ${numSamples}\n` +
                `**Estado:** Completado\n\n` +
                `Los datos sintéticos han sido generados exitosamente.`;
            
            // Incluir preview y datos completos si están disponibles
            const extraData = {};
            if (json.data && json.data.synthetic_data_preview) {
                extraData.synthetic_data_preview = json.data.synthetic_data_preview;
            }
            if (json.data && json.data.synthetic_data) {
                extraData.synthetic_data_full = json.data.synthetic_data;
            }
            
            this.renderAgentResultCard({
                title: `Generación con ${modelType}`,
                subtitle: `Agente: ${agentName} | ${numSamples} registros`,
                contentMarkdown: reply,
                footer: new Date().toLocaleString(),
                type: 'generation',
                meta: { model_type: modelType, num_samples: numSamples },
                ...extraData
            });
            
            this.showNotification(`✅ ${numSamples} registros generados con ${modelType}`, 'success');
        } catch (e) {
            this.showNotification(`Error generando: ${e.message || e}`, 'error');
            console.error('Error en generación:', e);
        } finally { 
            this.hideLoadingState(); 
        }
    }

    async callEvaluation() {
        if (!this.currentDatasetId) return this.showNotification('Sube un dataset primero', 'warning');
        try {
            this.showLoadingState();
            const res = await fetch(`${this.apiBase}/evaluation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_id: this.currentDatasetId })
            });
            const json = await res.json();
            const reply = (json.data && json.data.evaluation_response && (json.data.evaluation_response.response || json.data.evaluation_response.message)) || json.message || 'Evaluación completada';
            this.addChatMessage('Evaluación completada. Consulta el resultado en el panel central.', 'assistant');
            // Render result in central panel
            const agentName = (json.data && json.data.evaluation_response && json.data.evaluation_response.agent) || 'evaluator';
            this.renderAgentResultCard({
                title: 'Evaluación de calidad',
                subtitle: `Agente: ${agentName}`,
                contentMarkdown: reply,
                footer: new Date().toLocaleString(),
                type: 'evaluation'
            });
            this.showNotification('Evaluación de calidad completada', 'success');
        } catch (e) {
            this.showNotification(`Error evaluando: ${e.message || e}`, 'error');
        } finally { this.hideLoadingState(); }
    }

    async callValidate() {
        if (!this.currentDatasetId) return this.showNotification('Sube un dataset primero', 'warning');
        try {
            this.showLoadingState();
            const res = await fetch(`${this.apiBase}/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_id: this.currentDatasetId })
            });
            const json = await res.json();
            const reply = (json.data && json.data.agent_response && (json.data.agent_response.response || json.data.agent_response.message)) || json.message || 'Validación completada';
            this.addChatMessage('Validación completada. Consulta el resultado en el panel central.', 'assistant');
            // Render result in central panel
            const agentName = (json.data && json.data.agent_response && json.data.agent_response.agent) || 'validator';
            this.renderAgentResultCard({
                title: 'Validación médica',
                subtitle: `Agente: ${agentName}`,
                contentMarkdown: reply,
                footer: new Date().toLocaleString(),
                type: 'validation'
            });
            this.showNotification('Validación médica completada', 'success');
        } catch (e) {
            this.showNotification(`Error validando: ${e.message || e}`, 'error');
        } finally { this.hideLoadingState(); }
    }

    async callSimulation() {
        if (!this.currentDatasetId) return this.showNotification('Sube un dataset primero', 'warning');
        try {
            this.showLoadingState();
            const res = await fetch(`${this.apiBase}/simulation`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_id: this.currentDatasetId })
            });
            const json = await res.json();
            const reply = (json.data && json.data.simulation_response && (json.data.simulation_response.response || json.data.simulation_response.message)) || json.message || 'Simulación completada';
            this.addChatMessage('Simulación completada. Consulta el resultado en el panel central.', 'assistant');
            // Render result in central panel
            const agentName = (json.data && json.data.simulation_response && json.data.simulation_response.agent) || 'simulator';
            this.renderAgentResultCard({
                title: 'Simulación de pacientes',
                subtitle: `Agente: ${agentName}`,
                contentMarkdown: reply,
                footer: new Date().toLocaleString(),
                type: 'simulation'
            });
            this.showNotification('Simulación de pacientes completada', 'success');
        } catch (e) {
            this.showNotification(`Error simulando: ${e.message || e}`, 'error');
        } finally { this.hideLoadingState(); }
    }

    addChatMessage(message, sender) {
        const chatContainer = document.querySelector('.chat-container');
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}`;
        
        if (sender === 'user') {
            messageDiv.innerHTML = `
                <div class="message-content user-message">
                    ${message}
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-icon">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    ${message}
                </div>
            `;
        }
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    setupChatInput() {
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        if (!input || !sendBtn) return;

        const send = async () => {
            const text = (input.value || '').trim();
            if (!text) return;
            this.addChatMessage(text, 'user');
            input.value = '';
            try {
                this.showLoadingState();
                // Build enriched context with dataset information
                const enrichedContext = this.buildEnrichedChatContext();
                console.log('💬 Sending chat with enriched context:', enrichedContext);
                
                const res = await fetch(`${this.apiBase}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: text,
                        context: enrichedContext
                    })
                });
                const json = await res.json();
                console.log('Chat response:', json);
                let reply = 'Mensaje procesado';
                
                if (json.data) {
                    // Extract from various possible response structures
                    reply = json.data.response || 
                           json.data.answer ||
                           (json.data.chat_response && (json.data.chat_response.message || json.data.chat_response.response || json.data.chat_response.answer)) ||
                           (json.data.messages && json.data.messages.length && json.data.messages[json.data.messages.length - 1].content) ||
                           json.data.message ||
                           reply;
                } else if (json.message) {
                    reply = json.message;
                } else if (json.answer) {
                    reply = json.answer;
                }
                
                this.addChatMessage(reply, 'assistant');
                this.showNotification('Mensaje enviado correctamente', 'info');
            } catch (e) {
                this.showNotification(`Error enviando mensaje: ${e.message || e}`, 'error');
            } finally {
                this.hideLoadingState();
            }
        };

        sendBtn.addEventListener('click', () => { send(); });
        input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); send(); } });
    }

    showLoadingState() {
        // Add loading overlay or spinner
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading-overlay';
        loadingDiv.innerHTML = `
            <div class="loading-spinner">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Procesando...</p>
            </div>
        `;
        document.body.appendChild(loadingDiv);
    }

    hideLoadingState() {
        const loadingOverlay = document.querySelector('.loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.remove();
        }
    }

    updateStats() {
        // Animate stat numbers (but skip the 4th one which is the Target - it's not a number!)
        const statNumbers = document.querySelectorAll('.stat-number');
        
        statNumbers.forEach((stat, index) => {
            // Skip the Target KPI (index 3) as it's a string, not a number
            if (index === 3) {
                console.log('⏭️ Skipping Target KPI animation (index 3)');
                return;
            }
            
            const target = parseInt(stat.textContent.replace(/,/g, '').replace('k', '000'));
            
            // Only animate if it's a valid number
            if (isNaN(target)) {
                console.log(`⚠️ Skipping animation for KPI ${index}: not a number (${stat.textContent})`);
                return;
            }
            
            const increment = target / 100;
            let current = 0;
            
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                
                let displayValue = Math.floor(current);
                if (displayValue >= 1000) {
                    displayValue = (displayValue / 1000).toFixed(0) + 'k';
                } else {
                    displayValue = displayValue.toLocaleString();
                }
                
                stat.textContent = displayValue;
            }, 20);
        });
    }

    updateMainContent(section) {
        const header = document.querySelector('.header h1');
        const subtitle = document.querySelector('.header p');
        
        switch(section) {
            case 'Analizar Datos':
                header.textContent = 'Panel de Control';
                subtitle.textContent = 'Asistente para generación de datos clínicos sintéticos';
                break;
            case 'Generar Sintéticos':
                header.textContent = 'Generación de Datos Sintéticos';
                subtitle.textContent = 'Crea datos sintéticos basados en tu dataset';
                break;
            case 'Evaluar Calidad':
                header.textContent = 'Evaluación de Calidad';
                subtitle.textContent = 'Analiza la calidad de los datos sintéticos generados';
                break;
            case 'Simular Paciente':
                header.textContent = 'Simulación de Pacientes';
                subtitle.textContent = 'Simula evolución clínica de pacientes virtuales';
                break;
        }
    }

    adjustForMobile() {
        // Mobile-specific adjustments
        console.log('Adjusted for mobile view');
    }

    adjustForDesktop() {
        // Desktop-specific adjustments
        console.log('Adjusted for desktop view');
    }

    loadInitialData() {
        // Simulate initial data loading
        setTimeout(() => {
            this.updateStats();
        }, 1000);
    }

    // Placeholder para comportamiento responsive, evita errores si se llama
    setupResponsiveBehavior() {
        try {
            // Aquí puedes añadir lógica para mostrar/ocultar panel del asistente en móvil
        } catch (_) { /* noop */ }
    }

    // Modal helpers
    setupModal() {
        this.modal = {
            overlay: document.getElementById('modal-overlay'),
            title: document.getElementById('modal-title'),
            body: document.getElementById('modal-body'),
            btnClose: document.getElementById('modal-close'),
            btnCancel: document.getElementById('modal-cancel'),
            btnConfirm: document.getElementById('modal-confirm'),
            resolver: null,
        };
        const hide = () => this.hideModal();
        if (this.modal.btnClose) this.modal.btnClose.addEventListener('click', hide);
        if (this.modal.btnCancel) this.modal.btnCancel.addEventListener('click', hide);
        if (this.modal.overlay) this.modal.overlay.addEventListener('click', (e) => { if (e.target === this.modal.overlay) hide(); });
        if (this.modal.btnConfirm) this.modal.btnConfirm.addEventListener('click', () => { if (this.modal.resolver) { const res = this.collectModalForm(); this.modal.resolver(res); this.hideModal(); } });
    }

    showModal({ title, bodyHtml, onConfirm }) {
        if (!this.modal || !this.modal.overlay) return;
        this.modal.title.textContent = title || 'Configuración';
        this.modal.body.innerHTML = bodyHtml || '';
        this.modal.overlay.classList.remove('hidden');
        return new Promise(resolve => { this.modal.resolver = (vals) => { onConfirm && onConfirm(vals); resolve(vals); }; });
    }

    hideModal() {
        if (!this.modal || !this.modal.overlay) return;
        this.modal.overlay.classList.add('hidden');
        this.modal.body.innerHTML = '';
        this.modal.resolver = null;
    }

    collectModalForm() {
        const form = this.modal && this.modal.body ? this.modal.body.querySelector('form') : null;
        if (!form) return {};
        const data = new FormData(form);
        const obj = {};
        for (const [k, v] of data.entries()) obj[k] = v;
        // Also capture unchecked checkboxes
        form.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            if (!obj.hasOwnProperty(cb.name)) obj[cb.name] = cb.checked;
        });
        // convert numbers
        if (obj.num_samples) obj.num_samples = Number(obj.num_samples);
        return obj;
    }

    // Results: render a card in the center with markdown or table
    renderAgentResultCard({ title, subtitle, contentMarkdown, tableRecords, synthetic_data_preview, synthetic_data_full, footer, type, meta }) {
        const container = document.getElementById('agent-results');
        if (!container) return;
        
        // Show results section if hidden
        const resultsSection = document.getElementById('results-section');
        if (resultsSection) resultsSection.style.display = 'block';
        
        const card = document.createElement('div');
        card.className = 'result-card';
        const headerHtml = `<h4>${title || 'Resultado'}</h4><div class="meta">${subtitle || ''}</div>`;
        let bodyHtml = '';
        if (contentMarkdown) {
            console.log('Rendering markdown content:', contentMarkdown.substring(0, 200) + '...');
            const safeHtml = (window.marked ? window.marked.parse(contentMarkdown) : contentMarkdown.replace(/\n/g, '<br>'));
            bodyHtml += `<div class="content markdown-rendered">${safeHtml}</div>`;
        } else {
            console.warn('No markdown content to render');
        }
        
        // Render tabla de datos sintéticos si está disponible
        const previewData = synthetic_data_preview || tableRecords;
        if (Array.isArray(previewData) && previewData.length) {
            const cols = Object.keys(previewData[0]);
            const header = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>`;
            const body = `<tbody>${previewData.slice(0, 20).map(r => `<tr>${cols.map(c => `<td>${r[c] ?? ''}</td>`).join('')}</tr>`).join('')}</tbody>`;
            
            bodyHtml += `<div class="table-wrapper" style="margin-top:10px"><table class="preview-table">${header}${body}</table></div>`;
            
            // 🆕 Mensaje correcto: X visualizadas de Y generadas
            if (synthetic_data_preview && synthetic_data_full) {
                const totalGenerated = Array.isArray(synthetic_data_full) ? synthetic_data_full.length : (meta?.num_samples || previewData.length);
                const visualized = Math.min(20, previewData.length);
                bodyHtml += `<div class="meta" style="margin-top:8px">📊 Vista previa de las primeras ${visualized} filas de ${totalGenerated} registros generados</div>`;
            } else if (synthetic_data_preview) {
                bodyHtml += `<div class="meta" style="margin-top:8px">📊 Vista previa de las primeras ${Math.min(20, previewData.length)} filas</div>`;
            }
        }
        
        // 🆕 Botones de descarga para datos sintéticos
        if (synthetic_data_full && Array.isArray(synthetic_data_full) && synthetic_data_full.length > 0) {
            bodyHtml += `
                <div class="download-buttons" style="margin-top: 16px; display: flex; gap: 12px; justify-content: flex-start;">
                    <button class="btn-download btn-download-csv" data-type="csv">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 6px;">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                        Descargar CSV
                    </button>
                    <button class="btn-download btn-download-json" data-type="json">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 6px;">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                        Descargar JSON
                    </button>
                </div>
            `;
        }
        
        // Charts are no longer embedded in cards; they render in a separate section
        const footerHtml = footer ? `<div class="meta" style="margin-top:8px">${footer}</div>` : '';
        card.innerHTML = `${headerHtml}${bodyHtml}${footerHtml}`;
        
        // 🆕 Agregar event listeners para botones de descarga
        if (synthetic_data_full && Array.isArray(synthetic_data_full) && synthetic_data_full.length > 0) {
            const csvBtn = card.querySelector('.btn-download-csv');
            const jsonBtn = card.querySelector('.btn-download-json');
            
            if (csvBtn) {
                csvBtn.addEventListener('click', () => {
                    this.downloadSyntheticData(synthetic_data_full, 'csv', meta);
                });
            }
            
            if (jsonBtn) {
                jsonBtn.addEventListener('click', () => {
                    this.downloadSyntheticData(synthetic_data_full, 'json', meta);
                });
            }
        }
        
        // NUEVO: Si hay un resultado del mismo tipo, reemplazarlo en lugar de añadir
        if (type) {
            const existingCard = container.querySelector(`.result-card[data-type="${type}"]`);
            if (existingCard) {
                console.log(`🔄 Reemplazando resultado anterior de tipo: ${type}`);
                existingCard.replaceWith(card);
            } else {
                container.prepend(card);
            }
            // Marcar el tipo en el card para futuras búsquedas
            card.setAttribute('data-type', type);
        } else {
            container.prepend(card);
        }
        
        if (type) {
            this.setResultContext(type, {
                markdown: contentMarkdown || '',
                tablePreview: Array.isArray(tableRecords) ? tableRecords.slice(0, 20) : undefined,
                synthetic_data_full: synthetic_data_full,
                meta: { title: title || 'Resultado', subtitle: subtitle || '', footer: footer || '', created_at: new Date().toISOString(), ...meta }
            });
        }
        // After insertion, if requested, render analysis charts in dedicated section
        if (this.includeChartsFlag) this.renderAnalysisCharts();
    }

    // 🆕 Descargar datos sintéticos en formato CSV o JSON
    downloadSyntheticData(data, format, meta = {}) {
        if (!Array.isArray(data) || data.length === 0) {
            this.showNotification('No hay datos para descargar', 'warning');
            return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
        const modelType = meta?.model_type || 'synthetic';
        const numSamples = meta?.num_samples || data.length;
        const filename = `datos_sinteticos_${modelType}_${numSamples}_${timestamp}`;

        try {
            if (format === 'csv') {
                // Convertir a CSV
                const headers = Object.keys(data[0]);
                const csvRows = [
                    headers.join(','), // Header row
                    ...data.map(row => 
                        headers.map(header => {
                            const value = row[header];
                            // Escapar comillas y envolver en comillas si contiene comas o comillas
                            if (value === null || value === undefined) return '';
                            const stringValue = String(value);
                            if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
                                return `"${stringValue.replace(/"/g, '""')}"`;
                            }
                            return stringValue;
                        }).join(',')
                    )
                ];
                const csvContent = csvRows.join('\n');
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                this.triggerDownload(blob, `${filename}.csv`);
                this.showNotification(`✅ Descargado: ${filename}.csv`, 'success');
            } else if (format === 'json') {
                // Convertir a JSON
                const jsonContent = JSON.stringify(data, null, 2);
                const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
                this.triggerDownload(blob, `${filename}.json`);
                this.showNotification(`✅ Descargado: ${filename}.json`, 'success');
            }
        } catch (error) {
            console.error('Error al descargar datos:', error);
            this.showNotification(`❌ Error al descargar: ${error.message}`, 'error');
        }
    }

    // Helper para desencadenar la descarga del archivo
    triggerDownload(blob, filename) {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    }

    // New: render analysis charts in their own section
    renderAnalysisCharts() {
        try {
            const section = document.getElementById('analysis-charts');
            const sectionContainer = document.getElementById('analysis-charts-section');
            if (!section) return;
            // Show the section
            if (sectionContainer) sectionContainer.style.display = 'block';
            // Inject charts placeholders
            section.innerHTML = this.renderGenericChartsHTML();
            // Draw charts using the latest stats
            this.drawGenericChartsFromStats(section);
        } catch (_) { /* noop */ }
    }

    renderGenericChartsHTML() {
        return `
            <div class="charts-grid" style="margin-top:12px; display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px;">
                <div class="chart-container">
                    <h3 style="margin:0 0 8px 0">Matriz de correlación</h3>
                    <canvas class="corrCanvas" height="200"></canvas>
                </div>
                <div class="chart-container">
                    <h3 style="margin:0 0 8px 0">Histogramas</h3>
                    <div class="histograms"></div>
                </div>
                <div class="chart-container">
                    <h3 style="margin:0 0 8px 0">Resumen tipo boxplot</h3>
                    <div class="boxplots"></div>
                </div>
            </div>`;
    }

    drawGenericChartsFromStats(rootEl) {
        const stats = this.lastStats;
        if (!stats) return;
        const root = rootEl || document;
        // Correlation heatmap
        try {
            const corr = stats.correlation || { labels: [], matrix: [] };
            const canvas = root.querySelector('.corrCanvas');
            if (canvas && corr.labels.length && corr.matrix.length) {
                const ctx = canvas.getContext('2d');
                const data = [];
                for (let i = 0; i < corr.matrix.length; i++) {
                    for (let j = 0; j < corr.matrix[i].length; j++) {
                        data.push({ x: j+1, y: i+1, v: corr.matrix[i][j] });
                    }
                }
                new Chart(ctx, {
                    type: 'bubble',
                    data: {
                        datasets: [{
                            label: 'corr',
                            data: data.map(p => ({ x: p.x, y: p.y, r: Math.max(2, Math.abs(p.v) * 10) })),
                            backgroundColor: data.map(p => p.v >= 0 ? 'rgba(63, 131, 248, 0.6)' : 'rgba(244, 63, 94, 0.6)')
                        }]
                    },
                    options: {
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { ticks: { callback: (v) => corr.labels[v-1] || v } },
                            y: { ticks: { callback: (v) => corr.labels[v-1] || v } }
                        }
                    }
                });
            }
        } catch {}
        // Histograms
        try {
            const container = root.querySelector('.histograms');
            const hs = (stats.histograms || []).slice(0, 3);
            if (container && hs.length) {
                container.innerHTML = '';
                hs.forEach(h => {
                    const c = document.createElement('canvas');
                    c.height = 160;
                    container.appendChild(c);
                    new Chart(c.getContext('2d'), {
                        type: 'bar',
                        data: { labels: h.bins, datasets: [{ label: h.column, data: h.counts, backgroundColor: '#4facfe' }] },
                        options: { plugins: { legend: { display: false } }, scales: { x: { ticks: { maxRotation: 0, autoSkip: true } } } }
                    });
                });
            }
        } catch {}
        // Boxplots summary
        try {
            const container = root.querySelector('.boxplots');
            const bs = (stats.boxplots || []).slice(0, 3);
            if (container && bs.length) {
                container.innerHTML = bs.map(b => `
                    <div style="font-size:12px; color:#4a5568; margin-bottom:6px;">
                        <strong>${b.column}</strong>: min ${b.min.toFixed(2)} • Q1 ${b.q1.toFixed(2)} • med ${b.median.toFixed(2)} • Q3 ${b.q3.toFixed(2)} • max ${b.max.toFixed(2)}
                    </div>`).join('');
            }
        } catch {}
    }
}

// Ensure dashboard hooks run when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
    try {
        if (window.dashboard instanceof PatientIADashboard) {
            return;
        }
        window.dashboard = new PatientIADashboard();
        // Do not call setupEventListeners again to avoid duplicate listeners
    } catch (e) { console.error(e); }
});

// Add loading and notification styles
const additionalStyles = `
<style>
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

.loading-spinner {
    background: white;
    padding: 40px;
    border-radius: 12px;
    text-align: center;
    color: #4a5568;
}

.loading-spinner i {
    font-size: 32px;
    margin-bottom: 16px;
    color: #319795;
}

.user-message {
    background: #319795 !important;
    color: white !important;
    margin-left: auto;
    max-width: 80%;
}

.chat-message:has(.user-message) {
    justify-content: flex-end;
}

.table-wrapper { max-height: 320px; overflow: auto; border: 1px solid #e2e8f0; border-radius: 8px; }
.preview-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.preview-table th, .preview-table td { padding: 8px 10px; border-bottom: 1px solid #edf2f7; white-space: nowrap; }
.preview-table thead th { position: sticky; top: 0; background: #f7fafc; z-index: 1; }
.dataset-list-item { display: flex; align-items: center; gap: 8px; padding: 8px 10px; cursor: pointer; border-radius: 6px; }
.dataset-list-item:hover { background: #edf2f7; }
</style>
`;

document.head.insertAdjacentHTML('beforeend', additionalStyles);
