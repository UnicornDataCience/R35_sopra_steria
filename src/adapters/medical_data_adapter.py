from dataclasses import dataclass, field
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import re


@dataclass
class FieldMapping:
    """Representa el mapeo estandarizado de un campo médico."""
    original_name: str
    standard_name: str
    confidence: float
    data_type: str
    samples: List[Any] = field(default_factory=list)
    description: str = ""

    
    def __post_init__(self):
        if self.samples is None:
            self.samples = []

class UniversalMedicalAdapter:
    """Adaptador universal para cualquier estructura de datos médicos"""
    
    def __init__(self):
        # Patrones estándar para detectar campos médicos
        self.field_patterns = {
            # Identificación del paciente
            'patient_id': {
                'patterns': [r'patient.*id', r'id.*patient', r'numero.*paciente', r'nhc', r'historia.*clinica'],
                'type': 'identifier',
                'examples': ['PAT001', 'NH123456', '1001']
            },
            
            # Demografía
            'age': {
                'patterns': [r'edad', r'age', r'anos', r'years'],
                'type': 'numerical',
                'examples': [25, 65, 80]
            },
            'gender': {
                'patterns': [r'sexo', r'sex', r'gender', r'genero'],
                'type': 'categorical',
                'examples': ['M', 'F', 'MALE', 'FEMALE']
            },
            'birth_date': {
                'patterns': [r'fecha.*nacimiento', r'birth.*date', r'nacimiento', r'dob'],
                'type': 'date',
                'examples': ['1990-01-01', '01/01/1990']
            },
            
            # Clínicos
            'diagnosis': {
                'patterns': [r'diagnostico', r'diagnosis', r'diag', r'enfermedad', r'disease', r'patologia'],
                'type': 'categorical',
                'examples': ['COVID-19', 'Hipertensión', 'Diabetes']
            },
            'symptoms': {
                'patterns': [r'sintomas', r'symptoms', r'manifestaciones', r'signos'],
                'type': 'categorical',
                'examples': ['Fiebre', 'Tos', 'Dolor torácico']
            },
            
            # Vitales
            'temperature': {
                'patterns': [r'temp', r'temperatura', r'temperature'],
                'type': 'numerical',
                'examples': [36.5, 38.2, 37.0]
            },
            'heart_rate': {
                'patterns': [r'fc', r'heart.*rate', r'frecuencia.*cardiaca', r'pulso'],
                'type': 'numerical',
                'examples': [70, 85, 120]
            },
            'blood_pressure_sys': {
                'patterns': [r'tas', r'systolic', r'presion.*sistolica', r'bp.*sys'],
                'type': 'numerical',
                'examples': [120, 140, 160]
            },
            'blood_pressure_dia': {
                'patterns': [r'tad', r'diastolic', r'presion.*diastolica', r'bp.*dia'],
                'type': 'numerical',
                'examples': [80, 90, 100]
            },
            'oxygen_saturation': {
                'patterns': [r'sat.*o2', r'saturacion', r'oxygen.*saturation', r'spo2'],
                'type': 'numerical',
                'examples': [95, 98, 92]
            },
            
            # Laboratorio - Diabetes específico
            'glucose': {
                'patterns': [r'glucosa', r'glucose', r'glicemia'],
                'type': 'numerical',
                'examples': [90, 120, 200]
            },
            'pregnancies': {
                'patterns': [r'embarazos', r'pregnancies', r'gestaciones'],
                'type': 'numerical',
                'examples': [0, 1, 3]
            },
            'blood_pressure': {
                'patterns': [r'bloodpressure', r'presion.*arterial', r'bp'],
                'type': 'numerical',
                'examples': [80, 90, 100]
            },
            'skin_thickness': {
                'patterns': [r'skinthickness', r'grosor.*piel', r'triceps'],
                'type': 'numerical',
                'examples': [10, 20, 35]
            },
            'insulin': {
                'patterns': [r'insulin', r'insulina'],
                'type': 'numerical',
                'examples': [0, 80, 200]
            },
            'bmi': {
                'patterns': [r'bmi', r'imc', r'indice.*masa', r'body.*mass'],
                'type': 'numerical',
                'examples': [18.5, 25.0, 30.5]
            },
            'diabetes_pedigree': {
                'patterns': [r'diabetespedigreefunction', r'pedigri', r'herencia'],
                'type': 'numerical',
                'examples': [0.2, 0.5, 1.2]
            },
            'outcome': {
                'patterns': [r'outcome', r'resultado', r'target', r'clase', r'diabetes'],
                'type': 'categorical',
                'examples': [0, 1, 'Yes', 'No']
            },
            
            # Otros laboratorios
            'hemoglobin': {
                'patterns': [r'hemoglobina', r'hemoglobin', r'hb'],
                'type': 'numerical',
                'examples': [12.5, 14.0, 10.2]
            },
            
            # Medicación
            'medication': {
                'patterns': [r'medicamento', r'medication', r'farmaco', r'drug', r'tratamiento'],
                'type': 'categorical',
                'examples': ['Paracetamol', 'Ibuprofeno', 'Metformina']
            },
            
            # Hospitalización
            'icu_days': {
                'patterns': [r'dias.*uci', r'icu.*days', r'cuidados.*intensivos'],
                'type': 'numerical',
                'examples': [0, 3, 10]
            }
        }
    
    def _fix_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige columnas duplicadas SIN crear sufijos innecesarios"""
        
        columns = df.columns.tolist()
        
        # VERIFICAR SI REALMENTE HAY DUPLICADOS
        if len(columns) == len(set(columns)):
            print("✅ No hay columnas duplicadas reales")
            return df  # No hacer nada si no hay duplicados
        
        print(f"⚠️ Detectadas {len(columns) - len(set(columns))} columnas duplicadas")
        
        # Solo renombrar si hay duplicados REALES
        seen = {}
        new_columns = []
        
        for col in columns:
            if col in seen:
                seen[col] += 1
                new_name = f"{col}_{seen[col]}"
                new_columns.append(new_name)
                print(f"🔄 Renombrado duplicado: {col} → {new_name}")
            else:
                seen[col] = 0
                new_columns.append(col)
        
        df_fixed = df.copy()
        df_fixed.columns = new_columns
        
        return df_fixed

    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Analiza automáticamente cualquier dataset médico CON MEJOR MANEJO DE ERRORES"""
        
        try:
            print(f"🔍 Analizando dataset: {len(df)} filas, {len(df.columns)} columnas")
            
            # VERIFICACIONES INICIALES
            if df.empty:
                return self._fallback_analysis(df, "Dataset vacío")
            
            # LIMPIAR DATASET
            df_clean = self._fix_duplicate_columns(df)
            
            # USAR EXTRACTOR CON PROTECCIÓN
            try:
                from src.extraction.data_extractor import DataExtractor
                extractor = DataExtractor()
                
                # VERIFICAR QUE EL EXTRACTOR TENGA LOS MÉTODOS NECESARIOS
                if not hasattr(extractor, '_detect_medical_context'):
                    print("⚠️ DataExtractor no tiene _detect_medical_context, usando detección manual...")
                    # Detección manual como fallback
                    is_covid_dataset = self._detect_covid_dataset_manual(df_clean)
                    print(f"[DEBUG] Detección manual COVID-19: {is_covid_dataset}")
                    
                    # Create basic field mappings for fallback
                    field_mappings = self._create_basic_field_mappings(df_clean)

                    # NO filtrar columnas, solo registrar el tipo de dataset
                    extraction_results['is_covid_dataset'] = is_covid_dataset
                    extraction_results['cleaned_dataframe'] = df_clean

                    # Crear resultado manual
                    extraction_results = {
                        'medical_context': {
                            'primary_context': 'covid19' if is_covid_dataset else 'general',
                            'confidence': 0.8 if is_covid_dataset else 0.3,
                            'detected_features': ['covid_patterns'] if is_covid_dataset else [],
                            'clinical_specialty': 'infectologia' if is_covid_dataset else 'medicina_general',
                            'is_covid_dataset': is_covid_dataset
                        },
                        'field_mappings': field_mappings, # Use the generated mappings
                        'data_types': {col: 'categorical' for col in df_clean.columns}, # This needs to be updated based on filtered df
                        'temporal_patterns': {},
                        'quality_metrics': {
                            'completeness': 1.0,
                            'total_records': len(df_clean),
                            'total_fields': len(df_clean.columns)
                        },
                        'clinical_patterns': {},
                        'standardized_schema': {},
                        'recommendations': ["Análisis manual completado"],
                        'cleaned_dataframe': df_clean,
                        'is_covid_dataset': is_covid_dataset
                    }
                else:
                    # Usar extractor normal
                    print("[DEBUG] Usando DataExtractor normal...")
                    extraction_results = extractor.extract_patterns(df_clean)
                    is_covid_dataset = extraction_results.get('medical_context', {}).get('is_covid_dataset', False)
                    print(f"[DEBUG] Detección DataExtractor COVID-19: {is_covid_dataset}")
                    
                    # NO filtrar columnas, solo registrar el tipo de dataset
                    extraction_results['is_covid_dataset'] = is_covid_dataset
                    extraction_results['cleaned_dataframe'] = df_clean # Asegurarse de que el df limpio esté en los resultados

                # VERIFICAR RESULTADOS VÁLIDOS
                if not extraction_results or 'medical_context' not in extraction_results:
                    print("⚠️ Resultados del extractor inválidos, usando fallback...")
                    return self._fallback_analysis(df_clean)
                
                return extraction_results
            
            except Exception as extractor_error:
                print(f"❌ Error en extractor: {extractor_error}")
                return self._fallback_analysis(df_clean, str(extractor_error))
        
        except Exception as e:
            print(f"❌ Error general en análisis: {e}")
            return self._fallback_analysis(df, str(e))

    def _filter_covid_columns(self, df: pd.DataFrame, field_mappings: Dict[str, FieldMapping]) -> pd.DataFrame:
        """
        Filtra el DataFrame para quedarse solo con las columnas más relevantes para COVID-19.
        Utiliza los mapeos de campo para identificar las columnas.
        """
        print("Filtering COVID-19 columns...")
        
        # Desired standard names for COVID-19 relevant columns
        desired_standard_names = [
            'patient_id', 'age', 'gender', 'diagnosis', 'temperature',
            'oxygen_saturation', 'icu_days', 'medication', 'symptoms', 'outcome'
        ]
        
        # Map standard names back to original column names present in the DataFrame
        columns_to_keep = []
        for original_col, mapping in field_mappings.items():
            standard_name = getattr(mapping, 'standard_name', mapping.get('standard_name'))
            if standard_name in desired_standard_names and original_col in df.columns:
                columns_to_keep.append(original_col)
        
        # Ensure unique columns and maintain order if possible
        columns_to_keep = list(dict.fromkeys(columns_to_keep)) # Remove duplicates while preserving order
        
        if not columns_to_keep:
            print("⚠️ No se encontraron columnas COVID-19 relevantes. Manteniendo todas las columnas.")
            return df # Return original if no relevant columns found
        
        print(f"✅ Columnas COVID-19 seleccionadas: {len(columns_to_keep)} columnas: {columns_to_keep}")
        return df[columns_to_keep]

    def _detect_covid_dataset_manual(self, df: pd.DataFrame) -> bool:
        """Detección manual de datasets COVID-19"""
        try:
            # Buscar indicadores COVID en nombres de columnas
            covid_indicators = ['COVID', 'DIAG', 'FARMACO', 'UCI', 'ICU', 'PATIENT']
            column_score = sum(1 for indicator in covid_indicators 
                              if any(indicator in str(col).upper() for col in df.columns))
            
            # Buscar contenido COVID
            content_score = 0
            if len(df) > 0:
                sample_df = df.head(50)  # Muestrear solo 50 filas
                for col in sample_df.columns:
                    if sample_df[col].dtype == 'object':
                        try:
                            sample_values = sample_df[col].dropna().astype(str).head(5)
                            for val in sample_values:
                                if 'COVID' in str(val).upper():
                                    content_score += 1
                                    break
                        except:
                            continue
            
            is_covid = column_score >= 3 or content_score > 0
            print(f"🔍 Detección COVID-19: Score={column_score}, Contexto={'covid19' if is_covid else 'general'}, Confianza={100.0 if is_covid else 30.0}%")
            
            return is_covid
        
        except Exception as e:
            print(f"⚠️ Error en detección manual COVID: {e}")
            return False

    def _fallback_analysis(self, df: pd.DataFrame, error_message: str = "") -> Dict:
        """Análisis de fallback que SIEMPRE devuelve objetos FieldMapping para evitar errores."""
        print(f"🔄 Activado análisis de fallback: {error_message}")
        
        mappings = {}
        for col in df.columns:
            # Crea un objeto FieldMapping incluso en el modo de fallback
            mappings[col] = FieldMapping(
                original_name=col,
                standard_name=f"fallback_{col.lower().replace(' ', '_')}",
                confidence=0.3, # Asigna una confianza baja
                data_type='categorical',
                samples=df[col].dropna().head(3).astype(str).tolist() if not df.empty else [],
                description="Mapeo de fallback por error en análisis."
            )

        return {
            'field_mappings': mappings, # Ahora contiene objetos, no dicts
            'data_types': {m.standard_name: m.data_type for m in mappings.values()},
            'quality_metrics': {'completeness': 0.5, 'total_records': len(df)},
            'medical_context': {'primary_context': 'fallback', 'confidence': 0.1, 'is_covid_dataset': False},
            'recommendations': [f"Análisis de fallback: {error_message}"],
            'cleaned_dataframe': df,
            'is_covid_dataset': False
        }
    
    def _detect_medical_fields(self, df: pd.DataFrame) -> Dict[str, FieldMapping]:
        """Detecta automáticamente qué representa cada columna"""
        
        mappings = {}
        
        for col in df.columns:
            col_lower = col.lower().strip()
            col_clean = re.sub(r'[^\w\s]', '', col_lower)
            
            best_match = None
            best_confidence = 0.0
            
            # Buscar coincidencias con patrones conocidos
            for standard_field, config in self.field_patterns.items():
                confidence = 0.0
                
                # Verificar patrones de nombres
                for pattern in config['patterns']:
                    if re.search(pattern, col_clean):
                        confidence += 0.8
                        break
                
                # Verificar tipos de datos
                if self._check_data_type_compatibility(df[col], config['type']):
                    confidence += 0.3
                
                # Verificar ejemplos de valores
                if self._check_value_examples(df[col], config['examples']):
                    confidence += 0.4
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = standard_field
            
            # Solo mapear si hay confianza mínima
            if best_confidence >= 0.6:
                mappings[col] = FieldMapping(
                    original_name=col,
                    standard_name=best_match,
                    data_type=self.field_patterns[best_match]['type'],
                    confidence=best_confidence,
                    samples=df[col].dropna().astype(str).head(5).tolist()
                )
            else:
                # Campo no identificado - mantener original
                mappings[col] = FieldMapping(
                    original_name=col,
                    standard_name=f"custom_{col_clean}",
                    data_type=self._infer_data_type(df[col]),
                    confidence=0.3,
                    samples=df[col].dropna().astype(str).head(5).tolist()
                )
        
        return mappings
    
    def standardize_dataset(self, df: pd.DataFrame, field_mappings: Dict) -> pd.DataFrame:
        """
        Convierte el dataset a un esquema estándar solo renombrando columnas.
        La limpieza profunda de datos se delega al generador.
        """
        standardized_df = self._fix_duplicate_columns(df.copy())
        column_renames = {}

        for original_col, mapping in field_mappings.items():
            # Usar hasattr para manejar tanto objetos FieldMapping como dicts de fallback
            confidence = getattr(mapping, 'confidence', mapping.get('confidence', 0.5))
            standard_name = getattr(mapping, 'standard_name', mapping.get('standard_name', original_col))

            if confidence >= 0.6 and original_col in standardized_df.columns:
                # Renombrar a nombre estándar - EVITAR DUPLICADOS
                target_name = standard_name
                counter = 1
                while target_name in column_renames.values():
                    target_name = f"{standard_name}_{counter}"
                    counter += 1
                column_renames[original_col] = target_name
        
        if column_renames:
            standardized_df.rename(columns=column_renames, inplace=True)
            print(f"✅ {len(column_renames)} columnas renombradas según el mapeo.")

        return standardized_df

    def generate_synthetic_config(self, analysis_results: Dict) -> Dict:
        """Genera la configuración para el modelo de datos sintéticos."""
        
        # Extraer la información necesaria de los resultados del análisis
        data_types = analysis_results.get('data_types', {})
        
        # Crear la configuración del modelo
        model_config = {
            'method': 'CTGAN',  # O TVAE, etc.
            'table_name': 'medical_data',
            'primary_key': None, # Se puede detectar si hay un campo ID
            'sdtypes': data_types
        }
        
        # Buscar una posible clave primaria
        for standard_name, sdtype in data_types.items():
            if sdtype == 'id':
                model_config['primary_key'] = standard_name
                break
                
        return model_config

    def _infer_data_type(self, series: pd.Series) -> str:
        """Infiere el tipo de datos de una serie"""
        
        if pd.api.types.is_numeric_dtype(series):
            return 'numerical'
        elif series.dtype == 'object':
            # Verificar si es categórico o texto libre
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.5:
                return 'categorical'
            else:
                return 'categorical'  # Por defecto categórico para texto
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'date'
        else:
            return 'categorical'
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analiza los tipos de datos del DataFrame"""
        
        data_types = {}
        for col in df.columns:
            data_types[col] = {
                'pandas_dtype': str(df[col].dtype),
                'inferred_type': self._infer_data_type(df[col]),
                'null_count': df[col].isnull().sum(),
                'unique_values': df[col].nunique(),
                'unique_ratio': df[col].nunique() / len(df) if len(df) > 0 else 0
            }
        
        return data_types
    
    def _detect_temporal_patterns(self, df: pd.DataFrame, field_mappings: Dict) -> Dict:
        """Detecta patrones temporales en los datos"""
        
        temporal_info = {
            'has_dates': False,
            'date_columns': [],
            'time_span': None,
            'temporal_resolution': None
        }
        
        return temporal_info  # Simplificado por ahora
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """Evalúa la calidad de los datos"""
        
        total_cells = len(df) * len(df.columns)
        null_cells = df.isnull().sum().sum()
        
        quality_metrics = {
            'completeness': 1 - (null_cells / total_cells) if total_cells > 0 else 0,
            'total_records': len(df),
            'total_fields': len(df.columns),
            'null_percentage': (null_cells / total_cells * 100) if total_cells > 0 else 0,
            'duplicate_records': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0,
            'columns_with_nulls': df.isnull().any().sum(),
            'numeric_columns': len(df.select_dtypes(include=['number']).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }
        
        # Evaluar calidad general
        if quality_metrics['completeness'] >= 0.95:
            quality_metrics['quality_level'] = 'Excelente'
        elif quality_metrics['completeness'] >= 0.85:
            quality_metrics['quality_level'] = 'Buena'
        elif quality_metrics['completeness'] >= 0.70:
            quality_metrics['quality_level'] = 'Aceptable'
        else:
            quality_metrics['quality_level'] = 'Requiere limpieza'
        
        return quality_metrics
    
    def _identify_clinical_patterns(self, df: pd.DataFrame, field_mappings: Dict) -> Dict:
        """Identifica patrones clínicos en los datos"""
        
        clinical_patterns = {
            'medical_entities': [],
            'vital_signs': [],
            'lab_values': [],
            'demographics': [],
            'clinical_outcomes': []
        }
        
        return clinical_patterns  # Simplificado por ahora
    
    def _create_standard_schema(self, field_mappings: Dict) -> Dict:
        """Crea un esquema estándar basado en los mapeos"""
        
        schema = {
            'fields': {},
            'constraints': [],
            'relationships': []
        }
        
        for original_col, mapping in field_mappings.items():
            if isinstance(mapping, FieldMapping):
                schema['fields'][mapping.standard_name] = {
                    'original_column': original_col,
                    'data_type': mapping.data_type,
                    'confidence': mapping.confidence,
                    'samples': mapping.samples
                }
            else:
                schema['fields'][mapping.get('standard_name', original_col)] = {
                    'original_column': original_col,
                    'data_type': mapping.get('data_type', 'categorical'),
                    'confidence': mapping.get('confidence', 0.5),
                    'samples': mapping.get('samples', [])
                }
        
        return schema
    
    def _generate_recommendations(self, field_mappings: Dict, quality_metrics: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        
        recommendations = []
        
        # Recomendaciones de calidad
        if quality_metrics['completeness'] < 0.85:
            recommendations.append(f"📊 Completar valores faltantes ({quality_metrics['null_percentage']:.1f}% de datos nulos)")
        
        if quality_metrics['duplicate_percentage'] > 5:
            recommendations.append(f"🔄 Revisar {quality_metrics['duplicate_records']} registros duplicados")
        
        # Recomendaciones de mapeo
        low_confidence_fields = []
        for mapping in field_mappings.values():
            if isinstance(mapping, FieldMapping):
                if mapping.confidence < 0.6:
                    low_confidence_fields.append(mapping.original_name)
            else:
                if mapping.get('confidence', 0.5) < 0.6:
                    low_confidence_fields.append(mapping.get('original_name', 'unknown'))
        
        if low_confidence_fields:
            recommendations.append(f"🎯 Verificar mapeo de campos: {', '.join(low_confidence_fields[:3])}")
        
        # Recomendaciones técnicas
        if quality_metrics['numeric_columns'] > quality_metrics['categorical_columns']:
            recommendations.append("🔬 Dataset principalmente numérico - considerar SDV o TVAE")
        else:
            recommendations.append("📊 Dataset con muchas categorías - considerar CTGAN")
        
        return recommendations
    
    def generate_synthetic_config(self, analysis_results: Dict) -> Dict:
        """Genera configuración óptima para generación sintética con manejo robusto"""
        
        field_mappings = analysis_results.get('field_mappings', {})
        quality_metrics = analysis_results.get('quality_metrics', {})
        
        config = {
            'method': 'auto',
            'constraints': [],
            'privacy_level': 'medium',
            'sample_ratio': 1.0
        }
        
        if not field_mappings:
            return config
        
        # Determinar mejor método según características
        numeric_count = 0
        categorical_count = 0
        
        for mapping in field_mappings.values():
            try:
                if hasattr(mapping, 'data_type'):  # Objeto FieldMapping
                    data_type = mapping.data_type
                elif isinstance(mapping, dict):  # Diccionario
                    data_type = mapping.get('data_type', 'categorical')
                else:
                    data_type = 'categorical'
            
                if data_type in ['numeric', 'numerical']:
                    numeric_count += 1
                elif data_type == 'categorical':
                    categorical_count += 1
            except:
                categorical_count += 1  # Fallback
    
        total_fields = len(field_mappings)
        if total_fields > 0:
            numeric_ratio = numeric_count / total_fields
            categorical_ratio = categorical_count / total_fields
        
            # Elegir método basado en proporciones
            if categorical_ratio > 0.7:
                config['method'] = 'ctgan'
            elif numeric_ratio > 0.7:
                config['method'] = 'tvae'
            else:
                config['method'] = 'sdv'
    
        # Añadir restricciones médicas
        for col, mapping in field_mappings.items():
            try:
                if hasattr(mapping, 'standard_name'):
                    standard_name = mapping.standard_name
                elif isinstance(mapping, dict):
                    standard_name = mapping.get('standard_name', col)
                else:
                    standard_name = col
            
                # Restricciones médicas estándar
                if any(keyword in standard_name.lower() for keyword in ['age', 'edad']):
                    config['constraints'].append({
                        'column': standard_name,
                        'type': 'range',
                        'min': 0,
                        'max': 120
                    })
                elif any(keyword in standard_name.lower() for keyword in ['glucose', 'gluc']):
                    config['constraints'].append({
                        'column': standard_name,
                        'type': 'range',
                        'min': 50,
                        'max': 400
                    })
            except:
                continue
    
        return config
    
    
    def _safe_get_mapping_attr(self, mapping, attr_name, default=None):
        """Obtiene atributo de mapping de forma segura"""
        if isinstance(mapping, FieldMapping):
            return getattr(mapping, attr_name, default)
        elif isinstance(mapping, dict):
            return mapping.get(attr_name, default)
        else:
            return default
    
    
    def _safe_get_mapping_attr(self, mapping, attr_name, default=None):
        """Obtiene atributo de mapping de forma segura"""
        if isinstance(mapping, FieldMapping):
            return getattr(mapping, attr_name, default)
        elif isinstance(mapping, dict):
            return mapping.get(attr_name, default)
        else:
            return default
    
    
    def _safe_get_mapping_attr(self, mapping, attr_name, default=None):
        """Obtiene atributo de mapping de forma segura"""
        if isinstance(mapping, FieldMapping):
            return getattr(mapping, attr_name, default)
        elif isinstance(mapping, dict):
            return mapping.get(attr_name, default)
        else:
            return default
    
    
    def _safe_get_mapping_attr(self, mapping, attr_name, default=None):
        """Obtiene atributo de mapping de forma segura"""
        if isinstance(mapping, FieldMapping):
            return getattr(mapping, attr_name, default)
        elif isinstance(mapping, dict):
            return mapping.get(attr_name, default)
        else:
            return default
    
    def _detect_covid_dataset(self, df: pd.DataFrame) -> bool:
        """Detecta específicamente si es un dataset COVID-19"""
        
        covid_indicators = 0
        
        # Verificar columnas típicas COVID-19
        covid_columns = [
            'DIAG ING/INPAT',
            'FARMACO/DRUG_NOMBRE_COMERCIAL/COMERCIAL_NAME',
            'UCI_DIAS/ICU_DAYS',
            'TEMP_ING/INPAT',
            'SAT_02_ING/INPAT',
            'MOTIVO_ALTA/DESTINY_DISCHARGE_ING'
        ]
        
        for col in covid_columns:
            if col in df.columns:
                covid_indicators += 1
        
        # Verificar contenido COVID-19
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).str.upper().head(50)
                covid_content = sample_values.str.contains('COVID', na=False).sum()
                if covid_content > 0:
                    covid_indicators += 2
                    break
        
        is_covid = covid_indicators >= 3
        print(f"🦠 Detección COVID-19: {covid_indicators} indicadores → {'SÍ' if is_covid else 'NO'}")
        
        return is_covid

    def _create_basic_field_mappings(self, df: pd.DataFrame) -> Dict[str, FieldMapping]:
            """Crea mapeos básicos de campos usando objetos FieldMapping"""
            mappings = {}
            
            for col in df.columns:
                try:
                    non_null_count = df[col].count()
                    total_count = len(df)
                    completeness = non_null_count / total_count if total_count > 0 else 0.0
                    
                    try:
                        samples = df[col].dropna().astype(str).head(3).tolist()
                    except:
                        samples = ['N/A']
                    
                    mappings[col] = FieldMapping(
                        original_name=col,
                        standard_name=col.lower().replace(' ', '_').replace('/', '_'),
                        confidence=completeness,
                        data_type='categorical',
                        samples=samples
                    )
                except Exception as e:
                    print(f"⚠️ Error mapeando {col}: {e}")
                    mappings[col] = FieldMapping(
                        original_name=col,
                        standard_name=f"field_{len(mappings)}",
                        confidence=0.5,
                        data_type='categorical',
                        samples=['N/A']
                    )
            
            return mappings