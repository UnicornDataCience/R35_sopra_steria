"""
Motor de reglas configurables para validación médica.
Parsea reglas desde YAML y las aplica a datasets.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class RulesEngine:
    """Motor de reglas configurables para validación."""
    
    def __init__(self, rules_file: str = None):
        """
        Inicializar motor de reglas.
        
        Args:
            rules_file: Path al archivo YAML de reglas
        """
        if rules_file is None:
            rules_file = Path(__file__).parent.parent / "config" / "clinical_rules.yaml"
        
        self.rules_file = Path(rules_file)
        self.rules: Dict[str, Any] = {}
        self.load_rules()
    
    def load_rules(self):
        """Cargar reglas desde archivo YAML."""
        try:
            with open(self.rules_file, 'r', encoding='utf-8') as f:
                self.rules = yaml.safe_load(f)
            logger.info(f"✅ Loaded clinical rules from {self.rules_file}")
        except Exception as e:
            logger.error(f"❌ Error loading rules from {self.rules_file}: {e}")
            self.rules = self._get_default_rules()
    
    def reload_rules(self):
        """Hot-reload de reglas (útil en desarrollo)."""
        logger.info("🔄 Reloading clinical rules...")
        self.load_rules()
    
    def _get_default_rules(self) -> Dict[str, Any]:
        """Reglas por defecto si falla la carga."""
        return {
            'generic': {
                'dataset_type': 'Generic',
                'numeric_rules': [
                    {
                        'name': 'Edad válida',
                        'fields': ['Age', 'Edad', 'age'],
                        'min': 0,
                        'max': 120,
                        'severity': 'critical',
                        'message': 'Edad fuera del rango válido (0-120 años)'
                    }
                ],
                'categorical_rules': [],
                'correlation_rules': []
            }
        }
    
    def get_rules_for_dataset(self, dataset_type: str) -> Dict[str, Any]:
        """
        Obtener reglas para un tipo de dataset específico.
        
        Args:
            dataset_type: Tipo de dataset ("covid19", "diabetes", etc.)
            
        Returns:
            Diccionario con reglas aplicables
        """
        # Normalizar nombre
        dataset_key = dataset_type.lower().replace('-', '').replace('_', '').replace(' ', '')
        
        # Mapeo de nombres comunes
        mapping = {
            'covid19': 'covid19',
            'covid': 'covid19',
            'diabetes': 'diabetes',
            'cardiology': 'cardiology',
            'cardiologia': 'cardiology'
        }
        
        dataset_key = mapping.get(dataset_key, 'generic')
        
        rules = self.rules.get(dataset_key, self.rules.get('generic', {}))
        logger.debug(f"Using rules for dataset type: {dataset_key}")
        return rules
    
    def validate_numeric_rules(self, 
                               df: pd.DataFrame, 
                               rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validar reglas numéricas.
        
        Args:
            df: DataFrame a validar
            rules: Lista de reglas numéricas
            
        Returns:
            Dict con score y lista de issues
        """
        issues = []
        valid_checks = 0
        total_checks = 0
        
        for rule in rules:
            # Buscar columna que coincida
            fields = rule['fields']
            found_col = None
            for field in fields:
                if field in df.columns:
                    found_col = field
                    break
            
            if found_col is None:
                continue  # Columna no existe, skip
            
            # Validar rango
            col_data = pd.to_numeric(df[found_col], errors='coerce')
            valid_mask = col_data.between(rule['min'], rule['max'])
            valid_ratio = valid_mask.mean()
            
            total_checks += 1
            if valid_ratio >= 0.95:  # 95% de valores válidos
                valid_checks += 1
            else:
                severity = rule.get('severity', 'warning')
                invalid_count = (~valid_mask).sum()
                issue_msg = f"[{severity.upper()}] {rule['name']}: {invalid_count} registros fuera de rango [{rule['min']}, {rule['max']}]"
                issues.append(issue_msg)
                logger.debug(issue_msg)
        
        score = (valid_checks / total_checks) if total_checks > 0 else 1.0
        return {'score': score, 'issues': issues, 'checks': total_checks}
    
    def validate_categorical_rules(self, 
                                   df: pd.DataFrame, 
                                   rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validar reglas categóricas.
        
        Args:
            df: DataFrame a validar
            rules: Lista de reglas categóricas
            
        Returns:
            Dict con score y lista de issues
        """
        issues = []
        valid_checks = 0
        total_checks = 0
        
        for rule in rules:
            # Buscar columna que coincida
            fields = rule['fields']
            found_col = None
            for field in fields:
                if field in df.columns:
                    found_col = field
                    break
            
            if found_col is None:
                continue
            
            # Validar valores permitidos
            allowed_values = rule['allowed_values']
            valid_mask = df[found_col].isin(allowed_values)
            valid_ratio = valid_mask.mean()
            
            total_checks += 1
            if valid_ratio >= 0.95:
                valid_checks += 1
            else:
                severity = rule.get('severity', 'warning')
                invalid_count = (~valid_mask).sum()
                issue_msg = f"[{severity.upper()}] {rule['name']}: {invalid_count} valores no permitidos (permitidos: {allowed_values})"
                issues.append(issue_msg)
                logger.debug(issue_msg)
        
        score = (valid_checks / total_checks) if total_checks > 0 else 1.0
        return {'score': score, 'issues': issues, 'checks': total_checks}
    
    def validate_correlation_rules(self, 
                                   df: pd.DataFrame, 
                                   rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validar reglas de correlación (lógica condicional).
        
        Args:
            df: DataFrame a validar
            rules: Lista de reglas de correlación
            
        Returns:
            Dict con lista de warnings
        """
        warnings = []
        
        # Implementación simplificada (puede mejorarse con parser de condiciones)
        for rule in rules:
            condition = rule.get('condition', '')
            severity = rule.get('severity', 'info')
            message = rule.get('message', '')
            
            # Por ahora solo logging de info
            warnings.append(f"[{severity.upper()}] Correlation rule: {message}")
        
        return {'warnings': warnings}
    
    def validate_dataframe(self, 
                          df: pd.DataFrame, 
                          dataset_type: str) -> Dict[str, Any]:
        """
        Validar DataFrame completo con reglas configurables.
        
        Args:
            df: DataFrame a validar
            dataset_type: Tipo de dataset
            
        Returns:
            Dict con scores, issues y detalles
        """
        rules = self.get_rules_for_dataset(dataset_type)
        
        # Validar reglas numéricas
        numeric_result = self.validate_numeric_rules(
            df, 
            rules.get('numeric_rules', [])
        )
        
        # Validar reglas categóricas
        categorical_result = self.validate_categorical_rules(
            df,
            rules.get('categorical_rules', [])
        )
        
        # Validar reglas de correlación
        correlation_result = self.validate_correlation_rules(
            df,
            rules.get('correlation_rules', [])
        )
        
        # Calcular score general
        scores = [numeric_result['score'], categorical_result['score']]
        overall_score = np.mean(scores) if scores else 1.0
        
        # Combinar issues
        all_issues = numeric_result['issues'] + categorical_result['issues'] + correlation_result['warnings']
        
        return {
            'overall_score': overall_score,
            'numeric_score': numeric_result['score'],
            'categorical_score': categorical_result['score'],
            'issues': all_issues,
            'total_checks': numeric_result['checks'] + categorical_result['checks']
        }


# Singleton global
_rules_engine: Optional[RulesEngine] = None

def get_rules_engine() -> RulesEngine:
    """Obtener instancia global del motor de reglas."""
    global _rules_engine
    if _rules_engine is None:
        _rules_engine = RulesEngine()
        logger.info("🔧 Rules engine initialized")
    return _rules_engine
