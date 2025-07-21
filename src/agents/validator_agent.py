from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
from .base_agent import BaseLLMAgent, BaseAgentConfig
from src.validation.clinical_rules import validate_patient_case
from src.validation.json_schema import validate_medical_data, get_schema_for_domain, detect_domain_from_data

class MedicalValidatorAgent(BaseLLMAgent):
    """Agente especializado en validación médica de datos sintéticos"""
    
    def __init__(self):
        config = BaseAgentConfig(
            name="Validador Médico",
            description="Especialista en validación de coherencia médica y clínica de datos sintéticos",
            system_prompt="""Eres un agente experto en validación médica de datos sintéticos. Tu tarea es interpretar los resultados de validación y presentar un informe claro y conciso en Markdown, evaluando si los datos son aptos para investigación.

**Tu informe debe seguir exactamente esta estructura:**

### 📋 Informe de Validación Médica

**Tipo de datos analizados:** [Sintéticos/Originales]
**Puntuación General:** [XX.X%] - [Excelente/Buena/Aceptable/Deficiente]

#### 🔍 Análisis Detallado

**Coherencia Clínica ([XX.X%]):**
- [Descripción de la coherencia clínica encontrada]

**Calidad de Esquema ([XX.X%]):**
- [Descripción de la calidad de los datos y estructura]

**Validación Farmacológica ([XX.X%]):**
- [Descripción de la validación de tratamientos y medicamentos]

#### ⚠️ Puntos de Atención
[Lista de problemas encontrados, si los hay]

#### ✅ Recomendaciones
[Recomendaciones específicas basadas en los resultados]

**Conclusión:** Los datos son **[aptos/no aptos]** para su uso en **[investigación/entrenamiento de modelos/análisis clínico]**.

---
*Validación realizada el [fecha] con algoritmos de coherencia médica avanzada.*"""
        )
        super().__init__(config, tools=[])  # Explícitamente sin herramientas

    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Punto de entrada principal para la validación."""
        context = context or {}
        synthetic_data = context.get("synthetic_data")
        original_data = context.get("dataframe")
        validation_target = context.get("validation_target", "synthetic")

        # Determinar qué datos validar
        data_to_validate = None
        if validation_target == "synthetic" and synthetic_data is not None:
            data_to_validate = synthetic_data
        elif validation_target == "original" and original_data is not None:
            data_to_validate = original_data
        elif synthetic_data is not None:
            # Fallback: usar datos sintéticos si están disponibles
            data_to_validate = synthetic_data
            validation_target = "synthetic"
        elif original_data is not None:
            # Fallback: usar datos originales si están disponibles
            data_to_validate = original_data
            validation_target = "original"

        if data_to_validate is None or (hasattr(data_to_validate, 'empty') and data_to_validate.empty):
            return {
                "message": "❌ **Error de Validación**\n\nNo se encontraron datos para validar. Por favor, carga un dataset o genera datos sintéticos primero.", 
                "agent": self.name, 
                "error": True
            }

        try:
            # Determinar tipo de dataset para validaciones específicas
            is_covid = context.get('universal_analysis', {}).get('dataset_type') == 'COVID-19'
            
            # Realizar validaciones médicas completas
            validation_results = self._perform_comprehensive_medical_validations(
                data_to_validate, 
                is_covid, 
                validation_target
            )

            # Crear el prompt para el LLM con los resultados
            prompt = self._create_detailed_llm_prompt(validation_results, validation_target)

            # Obtener el informe del LLM
            llm_response = await self.agent_executor.ainvoke({
                "input": prompt, 
                "chat_history": self.memory.chat_memory.messages
            })

            return {
                "message": llm_response.content,
                "agent": self.name,
                "validation_results": validation_results,
                "validation_target": validation_target
            }
            
        except Exception as e:
            return {
                "message": f"❌ **Error durante la validación:** {str(e)}", 
                "agent": self.name, 
                "error": True
            }

    def _create_detailed_llm_prompt(self, results: Dict[str, Any], validation_target: str) -> str:
        """Crea un prompt detallado para el LLM a partir de los resultados de la validación."""
        data_type = "Sintéticos" if validation_target == "synthetic" else "Originales"
        
        # Formatear issues
        issues_list = results.get('issues', [])
        issues_formatted = "\n- ".join(issues_list) if issues_list else "No se detectaron problemas críticos."
        
        # Formatear recomendaciones
        recommendations = results.get('recommendations', [])
        recommendations_formatted = "\n- ".join(recommendations) if recommendations else "Los datos están listos para uso directo."
        
        # Clasificar el score general
        overall_score = results.get('overall_score', 0)
        if overall_score >= 0.9:
            score_category = "Excelente"
        elif overall_score >= 0.8:
            score_category = "Buena" 
        elif overall_score >= 0.7:
            score_category = "Aceptable"
        else:
            score_category = "Deficiente"
        
        prompt = f"""DATOS DE VALIDACIÓN MÉDICA:

Tipo de datos: {data_type}
Puntuación general: {overall_score:.1%} ({score_category})
Fecha de validación: {datetime.now().strftime('%d/%m/%Y %H:%M')}

SCORES DETALLADOS:
- Coherencia clínica: {results.get('clinical_coherence', 0):.1%}
- Calidad de esquema: {results.get('data_quality', 0):.1%}
- Validación farmacológica: {results.get('pharmacological_validation', 0):.1%}
- Total de registros analizados: {results.get('total_records', 0)}

PROBLEMAS DETECTADOS:
{issues_formatted}

RECOMENDACIONES:
{recommendations_formatted}

ESTADÍSTICAS ADICIONALES:
- Errores de esquema: {results.get('schema_errors', 0)}
- Inconsistencias clínicas: {results.get('clinical_inconsistencies', 0)}
- Problemas farmacológicos: {results.get('pharmacological_issues', 0)}

Por favor, genera el informe de validación médica en Markdown siguiendo exactamente la estructura especificada en tu prompt del sistema."""
        
        return prompt

    def _safe_numeric_conversion(self, series: pd.Series) -> pd.Series:
        """
        Convierte una serie a numérica de forma segura, manejando strings y valores faltantes.
        """
        try:
            return pd.to_numeric(series, errors='coerce')
        except Exception:
            return series.fillna(0)
    
    def _safe_between_validation(self, data: pd.DataFrame, column: str, min_val: float, max_val: float) -> tuple:
        """
        Realiza validación between de forma segura con conversión numérica.
        Retorna (porcentaje_válido, número_inválidos).
        """
        if column not in data.columns:
            return 1.0, 0
        
        try:
            # Conversión segura a numérico
            numeric_data = self._safe_numeric_conversion(data[column])
            # Excluir NaN y valores 0 que pueden ser faltantes
            valid_data = numeric_data[(numeric_data.notna()) & (numeric_data != 0)]
            
            if len(valid_data) == 0:
                return 1.0, 0
                
            valid_range = valid_data.between(min_val, max_val, inclusive='both')
            valid_percentage = valid_range.mean()
            invalid_count = (~valid_range).sum()
            
            return valid_percentage, invalid_count
        except Exception as e:
            print(f"⚠️ Error en validación segura de {column}: {e}")
            return 0.5, 0

    def _perform_comprehensive_medical_validations(self, data: pd.DataFrame, is_covid_dataset: bool, validation_target: str) -> Dict[str, Any]:
        """Realiza validaciones médicas completas y devuelve un diccionario de resultados."""
        results = {
            "issues": [],
            "recommendations": [],
            "total_records": len(data),
            "schema_errors": 0,
            "clinical_inconsistencies": 0,
            "pharmacological_issues": 0
        }
        
        # LÓGICA DE SELECCIÓN DE VALIDACIÓN BASADA EN EL TIPO DE DATOS
        if validation_target == "synthetic":
            # DATOS SINTÉTICOS → Convertir a formato JSON médico y usar validación JSON
            print("🔍 Procesando datos sintéticos: conversión a formato JSON médico")
            
            # Convertir datos sintéticos al formato médico estándar
            try:
                from src.validation.synthetic_data_converter import SyntheticDataConverter
                converter = SyntheticDataConverter()
                
                # Solo convertir si no están ya en formato médico
                if not converter.is_already_medical_format(data):
                    print("  ⚡ Convirtiendo datos sintéticos al formato médico estándar")
                    data = converter.convert_to_medical_json_format(data)
                    results["issues"].append("Datos sintéticos convertidos al formato médico estándar para validación completa")
                else:
                    print("  ✅ Datos ya están en formato médico estándar")
                
            except ImportError as e:
                print(f"  ⚠️ No se pudo importar SyntheticDataConverter: {e}")
                results["issues"].append("Conversión de formato no disponible, usando datos originales")
            
            print("🔍 Usando validación JSON para datos sintéticos")
            return self._validate_structured_data(data, is_covid_dataset, results)
        else:
            # DATOS ORIGINALES → Usar validación tabular (formato CSV/DataFrame)
            print("🔍 Usando validación tabular para datos originales")
            return self._validate_tabular_data(data, results)
    
    def _validate_tabular_data(self, data: pd.DataFrame, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validación específica para datos tabulares"""
        try:
            from src.validation.tabular_medical_validator import TabularMedicalValidator
            tabular_validator = TabularMedicalValidator()
            
            # 1. Validar calidad de datos
            quality_results = tabular_validator.validate_data_quality(data)
            base_results['data_quality'] = quality_results['overall_quality_score']
            base_results['issues'].extend(quality_results['issues'])
            
            # 2. Validar coherencia clínica
            clinical_results = tabular_validator.validate_clinical_coherence(data)
            base_results['clinical_coherence'] = (clinical_results['coherence_score'] + clinical_results['correlation_score']) / 2
            base_results['issues'].extend(clinical_results['issues'])
            base_results['clinical_inconsistencies'] = clinical_results['logic_violations']
            
            # 3. Validar completitud (como proxy de validación farmacológica)
            completeness_results = tabular_validator.validate_completeness(data)
            base_results['pharmacological_validation'] = completeness_results['completeness_score']
            base_results['issues'].extend(completeness_results['issues'])
            
            # 4. Generar recomendaciones específicas para datos tabulares
            if base_results['data_quality'] < 0.7:
                base_results['recommendations'].append("Revisar calidad de datos: verificar rangos válidos y tipos de columnas")
            
            if base_results['clinical_coherence'] < 0.6:
                base_results['recommendations'].append("Mejorar coherencia clínica: revisar correlaciones entre variables médicas")
            
            if not completeness_results['essential_columns_present']:
                base_results['recommendations'].append("Incluir columnas esenciales faltantes (edad, sexo) para análisis médicos completos")
            
            # 5. Calcular puntuación general
            base_results['overall_score'] = (
                base_results['data_quality'] * 0.4 +
                base_results['clinical_coherence'] * 0.4 +
                base_results['pharmacological_validation'] * 0.2
            )
            
            return base_results
            
        except ImportError:
            # Fallback si no se puede importar el validador tabular
            return self._validate_structured_data(data, False, base_results)
    
    def _validate_structured_data(self, data: pd.DataFrame, is_covid_dataset: bool, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validación original para datos estructurados/JSON usando el nuevo sistema genérico"""
        
        # AUTO-DETECTAR DOMINIO MÉDICO
        try:
            # Convertir algunas filas a dict para detección de dominio
            sample_data = []
            for _, row in data.head(5).iterrows():
                record = row.to_dict()
                clean_record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
                sample_data.append(clean_record)
            
            from src.validation.json_schema import detect_domain_from_data
            detected_domain = detect_domain_from_data(sample_data)
            
            print(f"🎯 Dominio médico detectado: {detected_domain}")
            
            # Si es_covid_dataset es True, forzar dominio covid
            if is_covid_dataset:
                detected_domain = "covid"
                print("🔄 Forzando dominio COVID-19 por parámetro")
                
        except Exception as e:
            print(f"⚠️ Error detectando dominio, usando genérico: {e}")
            detected_domain = "generic"
        
        # 1. VALIDACIÓN DE ESQUEMA JSON GENÉRICA
        try:
            schema_errors = 0
            total_validated = 0
            
            for _, row in data.iterrows():
                if self._validate_row_schema(row, detected_domain):
                    # Validación exitosa
                    pass
                else:
                    schema_errors += 1
                total_validated += 1
            
            base_results['schema_errors'] = schema_errors
            base_results['data_quality'] = max(0.0, (total_validated - schema_errors) / total_validated) if total_validated > 0 else 0.0
            
            if schema_errors > 0:
                error_percentage = (schema_errors / total_validated) * 100
                base_results['issues'].append(f"{schema_errors} registros ({error_percentage:.1f}%) no cumplen el esquema JSON médico para dominio '{detected_domain}'.")
                if error_percentage > 10:
                    base_results['recommendations'].append(f"Revisar la integridad de los datos para el dominio '{detected_domain}' y corregir errores de formato.")
                    
        except Exception as e:
            base_results['data_quality'] = 0.5
            base_results['issues'].append(f"Error en validación de esquema: {str(e)}")

        # 2. VALIDACIÓN DE COHERENCIA CLÍNICA (adaptada al dominio)
        try:
            if detected_domain == "covid" or is_covid_dataset:
                clinical_score = self._validate_covid_clinical_coherence(data, base_results)
            elif detected_domain == "cardiology":
                clinical_score = self._validate_cardiology_clinical_coherence(data, base_results)
            else:
                clinical_score = self._validate_general_clinical_coherence(data, base_results)
            base_results['clinical_coherence'] = clinical_score
        except Exception as e:
            base_results['clinical_coherence'] = 0.5
            base_results['issues'].append(f"Error en validación clínica: {str(e)}")

        # 3. VALIDACIÓN FARMACOLÓGICA (adaptada al dominio)
        try:
            pharma_score = self._validate_pharmacological_coherence(data, base_results, detected_domain)
            base_results['pharmacological_validation'] = pharma_score
        except Exception as e:
            base_results['pharmacological_validation'] = 0.5
            base_results['issues'].append(f"Error en validación farmacológica: {str(e)}")

        # 4. SCORE GENERAL
        base_results['overall_score'] = np.mean([
            base_results['data_quality'], 
            base_results['clinical_coherence'], 
            base_results['pharmacological_validation']
        ])

        # 5. RECOMENDACIONES GENERALES ADAPTADAS AL DOMINIO
        domain_name = {
            "covid": "COVID-19",
            "cardiology": "cardiología", 
            "generic": "médico general"
        }.get(detected_domain, detected_domain)
        
        if base_results['overall_score'] >= 0.9:
            base_results['recommendations'].append(f"Los datos de {domain_name} presentan excelente calidad para investigación y entrenamiento de modelos.")
        elif base_results['overall_score'] >= 0.8:
            base_results['recommendations'].append(f"Los datos de {domain_name} son de buena calidad, adecuados para la mayoría de aplicaciones.")
        elif base_results['overall_score'] >= 0.7:
            base_results['recommendations'].append(f"Los datos de {domain_name} son aceptables pero requieren revisión antes de uso crítico.")
        else:
            base_results['recommendations'].append(f"Se recomienda revisar y mejorar la calidad de los datos de {domain_name} antes del uso.")

        # Agregar información del dominio detectado
        base_results['detected_domain'] = detected_domain

        return base_results

    def _validate_covid_clinical_coherence(self, data: pd.DataFrame, results: Dict[str, Any]) -> float:
        """Validación específica para datasets COVID-19"""
        coherence_scores = []
        
        # Validar temperatura corporal
        if 'temperature' in data.columns or 'TEMP_ING/INPAT' in data.columns:
            temp_col = 'temperature' if 'temperature' in data.columns else 'TEMP_ING/INPAT'
            temp_valid, temp_invalid_count = self._safe_between_validation(data, temp_col, 35.0, 42.0)
            coherence_scores.append(temp_valid)
            if temp_valid < 0.95:
                results['clinical_inconsistencies'] += temp_invalid_count
                results['issues'].append(f"{temp_invalid_count} registros con temperatura fuera del rango fisiológico (35-42°C).")

        # Validar saturación de oxígeno
        if 'oxygen_saturation' in data.columns or 'SAT_02_ING/INPAT' in data.columns:
            sat_col = 'oxygen_saturation' if 'oxygen_saturation' in data.columns else 'SAT_02_ING/INPAT'
            sat_valid, sat_invalid_count = self._safe_between_validation(data, sat_col, 70, 100)
            coherence_scores.append(sat_valid)
            if sat_valid < 0.9:
                results['clinical_inconsistencies'] += sat_invalid_count
                results['issues'].append(f"{sat_invalid_count} registros con saturación O2 fuera del rango clínico (70-100%).")

        # Validar edad
        if 'age' in data.columns or 'EDAD/AGE' in data.columns:
            age_col = 'age' if 'age' in data.columns else 'EDAD/AGE'
            age_valid, age_invalid_count = self._safe_between_validation(data, age_col, 0, 120)
            coherence_scores.append(age_valid)
            if age_valid < 0.98:
                results['clinical_inconsistencies'] += age_invalid_count
                results['issues'].append(f"{age_invalid_count} registros con edad fuera del rango válido (0-120 años).")

        return np.mean(coherence_scores) if coherence_scores else 0.5

    def _validate_general_clinical_coherence(self, data: pd.DataFrame, results: Dict[str, Any]) -> float:
        """Validación clínica general para otros tipos de datasets"""
        coherence_scores = []
        
        # Validar edad
        age_columns = ['age', 'EDAD/AGE', 'Age', 'AGE']
        age_col = None
        for col in age_columns:
            if col in data.columns:
                age_col = col
                break
        
        if age_col:
            age_valid, age_invalid_count = self._safe_between_validation(data, age_col, 0, 120)
            coherence_scores.append(age_valid)
            if age_valid < 0.98:
                results['clinical_inconsistencies'] += age_invalid_count
                results['issues'].append(f"{age_invalid_count} registros con edad fuera del rango válido (0-120 años).")

        # Validar género
        sex_columns = ['sex', 'SEXO/SEX', 'gender', 'Gender']
        sex_col = None
        for col in sex_columns:
            if col in data.columns:
                sex_col = col
                break
        
        if sex_col:
            valid_genders = ['M', 'F', 'MALE', 'FEMALE', 'Male', 'Female', '1', '0']
            gender_valid = data[sex_col].isin(valid_genders).mean()
            coherence_scores.append(gender_valid)
            if gender_valid < 0.95:
                invalid_count = len(data[~data[sex_col].isin(valid_genders)])
                results['clinical_inconsistencies'] += invalid_count
                results['issues'].append(f"{invalid_count} registros con género no válido.")

        return np.mean(coherence_scores) if coherence_scores else 0.7

    def _validate_pharmacological_coherence(self, data: pd.DataFrame, results: Dict[str, Any], domain: str) -> float:
        """Validación de coherencia farmacológica usando clinical_rules adaptada por dominio"""
        
        # Buscar columnas de medicamentos según el dominio
        if domain == "covid":
            drug_columns = [col for col in data.columns if 'DRUG' in col.upper() or 'FARMACO' in col.upper()]
        elif domain == "cardiology":
            drug_columns = [col for col in data.columns if 'MEDICAMENTO' in col.upper() or 'DRUG' in col.upper() or 'MEDICATION' in col.upper()]
        else:
            # Dominio genérico
            drug_columns = [col for col in data.columns if any(term in col.upper() for term in ['DRUG', 'FARMACO', 'MEDICATION', 'MEDICAMENTO', 'TREATMENT', 'TRATAMIENTO'])]
        
        if not drug_columns:
            # No hay columnas de medicamentos, validar con score neutro más alto para dominios específicos
            base_score = 0.85 if domain in ["covid", "cardiology"] else 0.8
            return base_score
        
        pharma_scores = []
        pharma_issues = 0
        
        try:
            # Validación específica por dominio
            if domain == "cardiology":
                # Validación específica para medicamentos de cardiología
                cardio_medications = {
                    'METOPROLOL', 'ATENOLOL', 'PROPRANOLOL', 'ENALAPRIL', 'LOSARTAN', 
                    'AMLODIPINO', 'NIFEDIPINO', 'ATORVASTATINA', 'SIMVASTATINA',
                    'WARFARINA', 'ASPIRINA', 'CLOPIDOGREL', 'DIGOXINA', 'FUROSEMIDA'
                }
                
                sample_size = min(50, len(data))
                sample_data = data.sample(n=sample_size, random_state=42)
                valid_records = 0
                
                for _, row in sample_data.iterrows():
                    medication = str(row.get(drug_columns[0], '')).upper()
                    if any(med in medication for med in cardio_medications) or medication == '':
                        valid_records += 1
                    else:
                        pharma_issues += 1
                
                pharma_score = valid_records / sample_size if sample_size > 0 else 0.85
                pharma_scores.append(pharma_score)
                
            elif domain == "covid":
                # Usar validate_patient_case para COVID (método original)
                sample_size = min(100, len(data))
                sample_data = data.sample(n=sample_size, random_state=42)
                
                valid_records = 0
                for _, row in sample_data.iterrows():
                    try:
                        patient_dict = row.to_dict()
                        from src.validation.clinical_rules import validate_patient_case
                        validation_result = validate_patient_case(patient_dict)
                        
                        if validation_result and not validation_result.get('errors'):
                            valid_records += 1
                    except Exception:
                        pharma_issues += 1
                        continue
                
                pharma_score = valid_records / sample_size if sample_size > 0 else 0.8
                pharma_scores.append(pharma_score)
                
            else:
                # Validación genérica básica
                sample_size = min(50, len(data))
                sample_data = data.sample(n=sample_size, random_state=42)
                
                # Simplemente verificar que las columnas de medicamentos no estén vacías
                valid_records = 0
                for _, row in sample_data.iterrows():
                    has_medication = any(pd.notna(row.get(col)) and str(row.get(col)).strip() != '' 
                                       for col in drug_columns)
                    if has_medication:
                        valid_records += 1
                
                pharma_score = valid_records / sample_size if sample_size > 0 else 0.7
                pharma_scores.append(pharma_score)
            
            if pharma_scores and pharma_scores[0] < 0.7:
                results['pharmacological_issues'] = pharma_issues
                results['issues'].append(f"Se detectaron {pharma_issues} inconsistencias farmacológicas en el dominio '{domain}'.")
                
        except ImportError:
            # Si no se puede importar validate_patient_case, hacer validación básica
            pharma_scores.append(0.8)
        except Exception as e:
            pharma_scores.append(0.6)
            results['issues'].append(f"Error en validación farmacológica para dominio '{domain}': {str(e)}")

        return np.mean(pharma_scores) if pharma_scores else 0.8

    def _validate_row_schema(self, row: pd.Series, domain: str = "generic") -> bool:
        """Valida una única fila contra el esquema JSON de manera robusta usando el nuevo sistema genérico."""
        try:
            record = row.to_dict()
            # Convertir NaNs a None para validación JSON
            clean_record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
            
            # Usar el nuevo sistema de validación genérica
            from src.validation.json_schema import validate_medical_data
            results = validate_medical_data([clean_record], domain)
            return results['valid_count'] > 0
            
        except ImportError:
            # Si no se puede importar, hacer validación básica
            return self._basic_schema_validation(row)
        except Exception:
            return False

    def _basic_schema_validation(self, row: pd.Series) -> bool:
        """Validación básica de esquema cuando no está disponible json_schema"""
        try:
            # Verificar campos básicos
            has_id = any(col for col in row.index if 'ID' in col.upper() or 'id' in col)
            has_age = any(col for col in row.index if 'EDAD' in col.upper() or 'AGE' in col.upper())
            has_sex = any(col for col in row.index if 'SEXO' in col.upper() or 'SEX' in col.upper())
            
            return has_id and has_age and has_sex
        except Exception:
            return False

    def _validate_cardiology_clinical_coherence(self, data: pd.DataFrame, results: Dict[str, Any]) -> float:
        """Validación clínica específica para datos de cardiología"""
        coherence_scores = []
        
        # Validar presión arterial sistólica
        if 'PRESION_SISTOLICA' in data.columns:
            sys_valid, sys_invalid_count = self._safe_between_validation(data, 'PRESION_SISTOLICA', 70, 250)
            coherence_scores.append(sys_valid)
            if sys_valid < 0.95:
                results['clinical_inconsistencies'] += sys_invalid_count
                results['issues'].append(f"{sys_invalid_count} registros con presión sistólica fuera del rango clínico (70-250 mmHg).")

        # Validar presión arterial diastólica
        if 'PRESION_DIASTOLICA' in data.columns:
            dias_valid, dias_invalid_count = self._safe_between_validation(data, 'PRESION_DIASTOLICA', 40, 150)
            coherence_scores.append(dias_valid)
            if dias_valid < 0.95:
                results['clinical_inconsistencies'] += dias_invalid_count
                results['issues'].append(f"{dias_invalid_count} registros con presión diastólica fuera del rango clínico (40-150 mmHg).")

        # Validar frecuencia cardíaca
        if 'FRECUENCIA_CARDIACA' in data.columns:
            hr_valid, hr_invalid_count = self._safe_between_validation(data, 'FRECUENCIA_CARDIACA', 30, 200)
            coherence_scores.append(hr_valid)
            if hr_valid < 0.95:
                results['clinical_inconsistencies'] += hr_invalid_count
                results['issues'].append(f"{hr_invalid_count} registros con frecuencia cardíaca fuera del rango clínico (30-200 bpm).")

        # Validar colesterol total
        if 'COLESTEROL_TOTAL' in data.columns:
            chol_valid, chol_invalid_count = self._safe_between_validation(data, 'COLESTEROL_TOTAL', 100, 400)
            coherence_scores.append(chol_valid)
            if chol_valid < 0.9:
                results['clinical_inconsistencies'] += chol_invalid_count
                results['issues'].append(f"{chol_invalid_count} registros con colesterol total fuera del rango clínico (100-400 mg/dL).")

        # Validar HDL
        if 'HDL' in data.columns:
            hdl_valid, hdl_invalid_count = self._safe_between_validation(data, 'HDL', 20, 100)
            coherence_scores.append(hdl_valid)
            if hdl_valid < 0.9:
                results['clinical_inconsistencies'] += hdl_invalid_count
                results['issues'].append(f"{hdl_invalid_count} registros con HDL fuera del rango clínico (20-100 mg/dL).")

        # Validar LDL
        if 'LDL' in data.columns:
            ldl_valid, ldl_invalid_count = self._safe_between_validation(data, 'LDL', 50, 300)
            coherence_scores.append(ldl_valid)
            if ldl_valid < 0.9:
                results['clinical_inconsistencies'] += ldl_invalid_count
                results['issues'].append(f"{ldl_invalid_count} registros con LDL fuera del rango clínico (50-300 mg/dL).")

        # Validar edad
        if 'age' in data.columns or 'EDAD/AGE' in data.columns:
            age_col = 'age' if 'age' in data.columns else 'EDAD/AGE'
            age_valid, age_invalid_count = self._safe_between_validation(data, age_col, 0, 120)
            coherence_scores.append(age_valid)
            if age_valid < 0.98:
                results['clinical_inconsistencies'] += age_invalid_count
                results['issues'].append(f"{age_invalid_count} registros con edad fuera del rango válido (0-120 años).")

        return np.mean(coherence_scores) if coherence_scores else 0.5
