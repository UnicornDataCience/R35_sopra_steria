import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
import re

class DataAnalyzer:
    """Extractor de patrones clínicos de datasets médicos"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Columnas mínimas requeridas para cualquier dataset médico
        self.required_columns = {
            'patient_id': ['PATIENT_ID', 'ID_PACIENTE', 'PATIENT', 'ID', 'PACIENTE_ID', 'RECORD_ID'],
            'main_diagnosis': ['DIAGNOSTICO', 'DIAGNOSIS', 'DIAG', 'MAIN_DIAG', 'PRIMARY_DIAGNOSIS'],
            'age': ['EDAD', 'AGE', 'YEARS', 'ANOS'],
            'gender': ['SEXO', 'GENDER', 'SEX', 'GENERO']
        }
        
        # Contextos médicos que puede detectar
        self.medical_contexts = {
            'covid19': {
                'keywords': ['COVID', 'SARS', 'CORONAVIRUS', 'POSITIVO', 'NEGATIVO', 'UCI', 'ICU'],
                'column_patterns': ['DIAG', 'COVID', 'FARMACO', 'DRUG', 'UCI', 'ICU', 'TEMP', 'SAT_02'],
                'specialty': 'infectologia'
            },
            'cardiology': {
                'keywords': ['CORAZON', 'HEART', 'CARDIAC', 'CARDIO', 'PRESION', 'PRESSURE'],
                'column_patterns': ['ECG', 'HEART', 'CARDIAC', 'PRESSURE', 'CHOLESTEROL'],
                'specialty': 'cardiologia'
            },
            'diabetes': {
                'keywords': ['DIABETES', 'GLUCOSA', 'GLUCOSE', 'INSULIN', 'INSULINA'],
                'column_patterns': ['GLUCOSE', 'INSULIN', 'HBA1C', 'DIABETES'],
                'specialty': 'endocrinologia'
            },
            'oncology': {
                'keywords': ['CANCER', 'TUMOR', 'ONCOLOGY', 'CHEMO', 'RADIATION'],
                'column_patterns': ['TUMOR', 'CANCER', 'STAGE', 'METASTASIS'],
                'specialty': 'oncologia'
            }
        }
    
    def extract_patterns(self, df: pd.DataFrame) -> Dict:
        """Extrae patrones automáticamente CON PROTECCIÓN CONTRA ERRORES"""
        
        print(f"Analizando dataset: {len(df)} filas, {len(df.columns)} columnas")
        
        # VERIFICAR DATASET VÁLIDO
        if df.empty or len(df) == 0:
            print("Dataset vacio, devolviendo analisis basico")
            return self._create_empty_analysis_result(df)
        
        # VERIFICAR COLUMNAS DUPLICADAS ANTES DE PROCESAR
        original_columns = df.columns.tolist()
        unique_columns = list(dict.fromkeys(original_columns))
        
        if len(original_columns) != len(unique_columns):
            print(f"⚠️ Detectadas {len(original_columns) - len(unique_columns)} columnas duplicadas")
            df = df.loc[:, ~df.columns.duplicated()]
            print(f" Dataset limpiado: {len(df.columns)} columnas unicas")
        
        try:
            # 1. DETECTAR CONTEXTO MÉDICO (con protección)
            context_detection = self._detect_medical_context(df)
            
            # 2. EXTRAER PATRONES CON PROTECCIÓN
            patterns_result = self._extract_safe_patterns(df, context_detection)
            
            # 3. CREAR RESULTADO COMPLETO CON TODAS LAS CLAVES REQUERIDAS
            return {
                'medical_context': context_detection,
                'field_mappings': patterns_result.get('field_mappings', {}),
                'data_types': patterns_result.get('data_types', {}),
                'temporal_patterns': patterns_result.get('temporal_patterns', {}),
                'quality_metrics': patterns_result.get('quality_metrics', {
                    'completeness': 1.0,
                    'total_records': len(patterns_result.get('cleaned_dataframe', df)), # Usar el df del patterns_result
                    'total_fields': len(patterns_result.get('cleaned_dataframe', df).columns) # Usar el df del patterns_result
                }),
                'clinical_patterns': patterns_result.get('clinical_patterns', {}),
                'standardized_schema': patterns_result.get('standardized_schema', {}),
                'recommendations': patterns_result.get('recommendations', ["Análisis completado"]),
                'cleaned_dataframe': patterns_result.get('cleaned_dataframe', df), # AQUI ESTA EL CAMBIO CLAVE
                'is_covid_dataset': context_detection.get('is_covid_dataset', False),
                'statistics': patterns_result.get('statistics', {}),  # Asegurar clave 'statistics'
                'analysis_type': context_detection.get('primary_context', 'general'),  # Garantizar 'analysis_type'
                'filtered_file_path': patterns_result.get('filtered_file_path') # Propagar la ruta del archivo
            }
        
        except Exception as e:
            print(f"Error en extract_patterns: {e}")
            return self._create_fallback_analysis_result(df, str(e))
    
    def _detect_medical_context(self, df: pd.DataFrame) -> Dict:
        """Detecta el contexto médico del dataset usando múltiples especialidades"""
        
        context_scores = {}
        detected_features = []
        
        try:
            # Analizar cada contexto médico definido
            for context_name, context_config in self.medical_contexts.items():
                score = 0
                features = []
                
                # Puntuación por columnas
                for pattern in context_config['column_patterns']:
                    if any(pattern in str(col).upper() for col in df.columns):
                        score += 2
                        features.append(f"Columna relacionada con {pattern}")
                
                # Puntuación por contenido (si hay datos categóricos)
                try:
                    for col in df.select_dtypes(include=['object']).columns:
                        if df[col].dtype == 'object':
                            col_text = ' '.join(df[col].astype(str).str.upper().unique()[:100])  # Limitar para performance
                            for keyword in context_config['keywords']:
                                if keyword in col_text:
                                    score += 1
                                    features.append(f"Contenido relacionado con {keyword}")
                except Exception:
                    pass  # Ignorar errores de contenido
                
                context_scores[context_name] = {
                    'score': score,
                    'features': features,
                    'specialty': context_config['specialty']
                }
            
            # Determinar contexto principal
            if not context_scores or all(info['score'] == 0 for info in context_scores.values()):
                # No se detectó contexto específico
                primary_context = 'general'
                confidence = 0.0
                specialty = 'medicina_general'
                is_covid = False
            else:
                # Encontrar el contexto con mayor puntuación
                best_context = max(context_scores.items(), key=lambda x: x[1]['score'])
                primary_context = best_context[0]
                max_score = best_context[1]['score']
                
                # Calcular confianza (normalizada)
                confidence = min(max_score * 0.1, 1.0)  # 10% por punto, máximo 100%
                specialty = best_context[1]['specialty']
                is_covid = (primary_context == 'covid19')
                
                # Recopilar todas las características detectadas
                detected_features = best_context[1]['features']
            
            result = {
                'primary_context': primary_context,
                'context': primary_context,  # Mantener compatibilidad
                'confidence': confidence,
                'detected_features': detected_features,
                'clinical_specialty': specialty,
                'is_covid_dataset': is_covid,
                'all_context_scores': context_scores
            }
            
            self.logger.info(f"Contexto médico detectado: {primary_context} (confianza: {confidence:.2f})")
            return result
                    
        except Exception as e:
            self.logger.error(f"Error detectando contexto médico: {str(e)}")
            return {
                'primary_context': 'general',
                'context': 'general',
                'confidence': 0.0,
                'detected_features': [],
                'clinical_specialty': 'medicina_general',
                'is_covid_dataset': False,
                'all_context_scores': {}
            }
    
    def _extract_safe_patterns(self, df: pd.DataFrame, context: Dict) -> Dict:
        """Extrae patrones con protección contra errores matemáticos"""
        
        try:
            field_mappings = {}
            data_types = {}
            filtered_output_path = None  # Inicializar la ruta del archivo filtrado
            patterns_result = {}
            patterns_result = {}
            patterns_result = {}
            patterns_result = {}
            patterns_result = {}

            # CALCULAR COMPLETENESS CON PROTECCIÓN CONTRA DIVISION BY ZERO
            total_cells = len(df) * len(df.columns)
            if total_cells > 0:
                non_null_cells = df.count().sum()
                completeness = non_null_cells / total_cells
            else:
                completeness = 0.0
            
            quality_metrics = {
                'completeness': completeness,
                'total_records': len(df),
                'total_fields': len(df.columns)
            }
            
            # MAPEAR CAMPOS BÁSICOS
            for col in df.columns:
                try:
                    non_null_count = df[col].count()
                    total_count = len(df)
                    col_completeness = non_null_count / total_count if total_count > 0 else 0.0
                    
                    # Detectar tipo básico
                    if pd.api.types.is_numeric_dtype(df[col]):
                        detected_type = 'numerical'
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        detected_type = 'datetime'
                    else:
                        detected_type = 'categorical'
                    
                    field_mappings[col] = {
                        'original_name': col,
                        'standard_name': col.lower().replace(' ', '_').replace('/', '_'),
                        'confidence': col_completeness,
                        'data_type': detected_type
                    }
                    data_types[col] = detected_type
                
                except Exception as col_error:
                    print(f"⚠️ Error procesando columna {col}: {col_error}")
                    field_mappings[col] = {
                        'original_name': col,
                        'standard_name': col.lower().replace(' ', '_'),
                        'confidence': 0.5,
                        'data_type': 'categorical'
                    }
                    data_types[col] = 'categorical'
            
            # REDUCIR COLUMNAS PARA COVID-19 SI ES NECESARIO (SOLO PARA GENERACIÓN)
            # Mantener todas las columnas para análisis, filtrar solo para guardado
            if context.get('primary_context') == 'covid19':
                # Mapear columnas COVID-19 disponibles
                covid_priority_columns = [
                    'PATIENT_ID', 'PATIENT', 'ID_PACIENTE', 'ID',
                    'EDAD', 'AGE', 'YEARS', 'ANOS',
                    'SEXO', 'GENDER', 'SEX', 'GENERO',
                    'DIAGNOSTICO', 'DIAGNOSIS', 'DIAG',
                    'SAT_02', 'OXYGEN', 'O2',
                    'HIPER_ART', 'HYPERTENSION', 'PRESION',
                    'ENF_RESPIRA', 'RESPIRATORY', 'RESPIRATORIO',
                    'DIABETES', 'DIABETIC',
                    'FARMACO', 'DRUG', 'MEDICATION',
                    'UCI', 'ICU', 'INTENSIVE'
                ]
                
                # Encontrar columnas existentes que coincidan con las prioridades COVID
                existing_covid_cols = []
                for col in df.columns:
                    for priority in covid_priority_columns:
                        if priority.upper() in col.upper():
                            existing_covid_cols.append(col)
                            break
                
                # Si no hay suficientes columnas COVID, usar todas las disponibles
                if len(existing_covid_cols) < 4:
                    existing_covid_cols = df.columns.tolist()
                
                print(f"DEBUG: Columnas COVID seleccionadas: {existing_covid_cols}")
                
                # Crear dataset filtrado para generación sintética
                df_filtered = df[existing_covid_cols].copy()
                
                # Guardar dataset filtrado
                filtered_output_path = 'data/real/filtered_covid_dataset.csv'
                df_filtered.to_csv(filtered_output_path, index=False)
                print(f"Dataset COVID-19 filtrado guardado en: {filtered_output_path} ({len(df_filtered.columns)} columnas)")
                
                # MANTENER EL DATAFRAME ORIGINAL para análisis
                # No reemplazar df aquí para mantener todas las columnas en el análisis
                
            
            return {
                'field_mappings': field_mappings,
                'data_types': data_types,
                'quality_metrics': quality_metrics,
                'temporal_patterns': {},
                'clinical_patterns': {},
                'standardized_schema': {},
                'recommendations': [f"Análisis completado para {len(field_mappings)} campos"],
                'statistics': self._calculate_basic_statistics(df),
                'filtered_file_path': filtered_output_path,
                'cleaned_dataframe': df
                
            }
        
        except Exception as e:
            print(f"Error en _extract_safe_patterns: {e}")
            return self._create_minimal_patterns_result(df)
    
    def _validate_required_columns(self, df: pd.DataFrame) -> Dict:
        """
        Valida que el dataset tenga las columnas mínimas requeridas para análisis médico.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            Dict con resultado de validación
        """
        validation_result = {
            'valid': True,
            'missing_columns': [],
            'column_mapping': {},
            'suggestions': [],
            'error': None
        }
        
        # Buscar columnas requeridas
        found_columns = {}
        df_columns_upper = [col.upper() for col in df.columns]
        
        for required_type, possible_names in self.required_columns.items():
            found = False
            for possible_name in possible_names:
                for idx, col in enumerate(df_columns_upper):
                    if possible_name in col:
                        found_columns[required_type] = df.columns[idx]
                        found = True
                        break
                if found:
                    break
            
            if not found:
                validation_result['missing_columns'].append(required_type)
        
        # Determinar si el dataset es válido
        if len(validation_result['missing_columns']) > 0:
            validation_result['valid'] = False
            validation_result['error'] = f"Dataset no válido: faltan columnas requeridas: {', '.join(validation_result['missing_columns'])}"
            
            # Generar sugerencias
            suggestions = []
            for missing in validation_result['missing_columns']:
                suggestions.append(f"Para '{missing}': usar una de estas columnas: {', '.join(self.required_columns[missing])}")
            
            validation_result['suggestions'] = suggestions
        else:
            validation_result['column_mapping'] = found_columns
            self.logger.info(f"Columnas requeridas encontradas: {found_columns}")
        
        return validation_result

    def analyze_dataset(self, dataset_path: str) -> Dict:
        """
        Analiza un dataset médico y extrae sus características principales.
        
        Args:
            dataset_path: Ruta al archivo CSV del dataset
            
        Returns:
            Dict con el análisis del dataset
        """
        try:
            # Leer el dataset
            df = pd.read_csv(dataset_path)
            self.logger.info(f"Dataset cargado con {len(df)} filas y {len(df.columns)} columnas")
            
            # Verificar columnas mínimas requeridas
            validation_result = self._validate_required_columns(df)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'missing_columns': validation_result['missing_columns'],
                    'suggestions': validation_result['suggestions']
                }
            
            # Usar el método existente extract_patterns
            patterns = self.extract_patterns(df)
            
            # Extraer información básica
            result = {
                'success': True,
                'dataset_info': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'column_names': df.columns.tolist(),
                    'memory_usage': df.memory_usage(deep=True).sum()
                },
                'medical_context': patterns.get('medical_context', {}),
                'patterns': patterns,
                'data_quality': {
                    'completeness': patterns.get('quality_metrics', {}).get('completeness', 0),
                    'total_records': len(df),
                    'total_fields': len(df.columns)
                },
                'recommendations': patterns.get('recommendations', []),
                'column_mapping': validation_result['column_mapping']
            }
            
            self.logger.info(f"Análisis completado exitosamente. Contexto detectado: {patterns.get('medical_context', {}).get('primary_context', 'general')}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error en análisis del dataset: {str(e)}")
            return {
                'success': False,
                'error': f"Error al procesar el dataset: {str(e)}"
            }

    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Obtiene estadísticas básicas del dataset"""
        try:
            stats = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                'missing_values': df.isnull().sum().sum(),
                'duplicated_rows': df.duplicated().sum(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            return stats
        except Exception as e:
            self.logger.error(f"Error calculando estadísticas básicas: {str(e)}")
            return {}

    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """Analiza la calidad de los datos"""
        try:
            quality = {
                'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'uniqueness': len(df.drop_duplicates()) / len(df) * 100,
                'columns_with_nulls': df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
                'data_types': df.dtypes.value_counts().to_dict()
            }
            return quality
        except Exception as e:
            self.logger.error(f"Error en análisis de calidad: {str(e)}")
            return {}

    def _generate_recommendations(self, df: pd.DataFrame, medical_context: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        try:
            # Recomendaciones por contexto médico
            if medical_context.get('primary_context') == 'covid19':
                recommendations.append("Dataset COVID-19 detectado: considere usar todas las columnas para análisis exploratorio")
                recommendations.append("Para generación sintética, se recomienda filtrar a las 10 columnas más relevantes")
            elif medical_context.get('primary_context') == 'general':
                recommendations.append("Dataset médico general: verifique que las columnas principales estén correctamente mapeadas")
            
            # Recomendaciones por calidad de datos
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            if missing_pct > 10:
                recommendations.append(f"Alto porcentaje de valores faltantes ({missing_pct:.1f}%): considere técnicas de imputación")
            
            if df.duplicated().sum() > 0:
                recommendations.append(f"Se encontraron {df.duplicated().sum()} filas duplicadas")
            
            # Recomendaciones por tamaño
            if len(df) < 1000:
                recommendations.append("Dataset pequeño: considere técnicas de augmentación de datos")
            elif len(df) > 100000:
                recommendations.append("Dataset grande: considere técnicas de muestreo para entrenamiento")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generando recomendaciones: {str(e)}")
            return ["Error generando recomendaciones"]

    def _create_empty_analysis_result(self, df: pd.DataFrame) -> Dict:
        """Crea resultado para dataset vacío"""
        return {
            'medical_context': {
                'primary_context': 'empty',
                'confidence': 0.0,
                'detected_features': [],
                'clinical_specialty': 'none',
                'is_covid_dataset': False
            },
            'field_mappings': {},
            'data_types': {},
            'temporal_patterns': {},
            'quality_metrics': {'completeness': 0.0, 'total_records': 0, 'total_fields': len(df.columns)},
            'clinical_patterns': {},
            'standardized_schema': {},
            'recommendations': ["Dataset vacío"],
            'cleaned_dataframe': df,
            'is_covid_dataset': False,
            'statistics': {} # Asegurar que 'statistics' esté presente
        }
    
    def _create_fallback_analysis_result(self, df: pd.DataFrame, error_msg: str) -> Dict:
        """Crea resultado de fallback en caso de error"""
        
        basic_mappings = {}
        for col in df.columns:
            basic_mappings[col] = {
                'original_name': col,
                'standard_name': col.lower().replace(' ', '_'),
                'confidence': 0.3,
                'data_type': 'categorical'
            }
        
        return {
            'medical_context': {
                'primary_context': 'error_fallback',
                'confidence': 0.1,
                'detected_features': ['error_recovery'],
                'clinical_specialty': 'unknown',
                'is_covid_dataset': False
            },
            'field_mappings': basic_mappings,
            'data_types': {col: 'categorical' for col in df.columns},
            'temporal_patterns': {},
            'quality_metrics': {'completeness': 0.5, 'total_records': len(df), 'total_fields': len(df.columns)},
            'clinical_patterns': {},
            'standardized_schema': {},
            'recommendations': [f"Análisis de fallback debido a error: {error_msg}"],
            'cleaned_dataframe': df,
            'is_covid_dataset': 'COVID' in ' '.join(df.columns).upper(),
            'error_info': error_msg,
            'statistics': {} # Asegurar que 'statistics' esté presente
        }
    
    def _create_minimal_patterns_result(self, df: pd.DataFrame) -> Dict:
        """Crea resultado mínimo de patrones"""
        return {
            'field_mappings': {col: {
                'original_name': col,
                'standard_name': col.lower(),
                'confidence': 0.5,
                'data_type': 'categorical'
            } for col in df.columns},
            'data_types': {col: 'categorical' for col in df.columns},
            'quality_metrics': {'completeness': 0.5, 'total_records': len(df), 'total_fields': len(df.columns)},
            'temporal_patterns': {},
            'clinical_patterns': {},
            'standardized_schema': {},
            'recommendations': ["Análisis mínimo completado"],
            'statistics': {} # Asegurar que 'statistics' esté presente
        }

    def _calculate_basic_statistics(self, df: pd.DataFrame) -> Dict:
        """Calcula estadísticas básicas para el dataset."""
        stats_dict = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats_dict[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
            elif df[col].dtype == 'object': # Categóricas o texto
                stats_dict[col] = {
                    'unique_values': df[col].nunique(),
                    'top_5_values': df[col].value_counts().head(5).to_dict()
                }
        return stats_dict
    
    def generate_standard_schema(self, df: pd.DataFrame) -> Dict:
        """Genera un esquema estándar basado en las características del dataset."""
        schema = {
            'fields': {},
            'constraints': []
        }

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                schema['fields'][col] = {
                    'type': 'numerical',
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                schema['fields'][col] = {
                    'type': 'datetime',
                    'format': '%Y-%m-%d'
                }
            else:
                schema['fields'][col] = {
                    'type': 'categorical',
                    'unique_values': df[col].dropna().unique().tolist()
                }

        return schema
    
    def _extract_patterns(self, df: pd.DataFrame, medical_context: Dict) -> Dict:
        """Wrapper para mantener compatibilidad con el método extract_patterns existente"""
        # Usar el método existente extract_patterns
        return self.extract_patterns(df)