"""
Servicio para manejar la gestión de datasets y archivos
"""
import os
import pandas as pd
import uuid
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from api.models.schemas import DatasetInfo, APIResponse
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class DatasetService:
    """Servicio para gestión de datasets"""
    
    def __init__(self, upload_dir: str = "data/uploaded_datasets"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def upload_dataset(self, filename: str, file_content: bytes) -> DatasetInfo:
        """
        Procesa y guarda un dataset subido
        """
        try:
            # Generar ID único para el dataset
            dataset_id = str(uuid.uuid4())

             # Sanitizar nombre de archivo (evita separadores y caracteres inválidos en Windows)
            safe_name = Path(str(filename)).name  # evita rutas embebidas
            safe_name = re.sub(r'[\\/:*?"<>|]+', '_', safe_name).strip()
            
            # Guardar archivo temporalmente
            file_path = self.upload_dir / f"{dataset_id}_{filename}"
            
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Leer y analizar el dataset
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Formato de archivo no soportado: {filename}")
            
            # Análisis básico del dataset
            dataset_info = DatasetInfo(
                id=dataset_id,
                filename=filename,
                rows=len(df),
                columns=len(df.columns),
                column_names=list(df.columns),
                dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                missing_values={col: int(df[col].isnull().sum()) for col in df.columns},
                file_path=str(file_path),
                upload_time=datetime.now(),
                file_size=len(file_content)
            )
            
            logger.info("Dataset procesado: %s (%d filas, %d columnas)", 
                       filename, len(df), len(df.columns))
            
            return dataset_info
            
        except Exception as e:
            logger.error("Error procesando dataset: %s", e)
            raise Exception(f"Error procesando dataset: {str(e)}")
    
    async def get_dataset_info(self, dataset_id: str) -> Optional[DatasetInfo]:
        """
        Obtiene información de un dataset por ID
        """
        try:
            # Buscar el archivo por ID
            for file_path in self.upload_dir.glob(f"{dataset_id}_*"):
                filename = file_path.name.replace(f"{dataset_id}_", "")
                
                # Leer dataset
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    continue
                
                return DatasetInfo(
                    id=dataset_id,
                    filename=filename,
                    rows=len(df),
                    columns=len(df.columns),
                    column_names=list(df.columns),
                    dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                    missing_values={col: int(df[col].isnull().sum()) for col in df.columns},
                    file_path=str(file_path),
                    upload_time=datetime.fromtimestamp(file_path.stat().st_mtime),
                    file_size=file_path.stat().st_size
                )
            
            return None
            
        except Exception as e:
            logger.error("Error obteniendo info del dataset: %s", e)
            return None
    
    async def get_dataset_preview(self, dataset_id: str, rows: int = 10) -> Optional[Dict[str, Any]]:
        """
        Obtiene una vista previa del dataset
        """
        try:
            for file_path in self.upload_dir.glob(f"{dataset_id}_*"):
                filename = file_path.name.replace(f"{dataset_id}_", "")
                
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    continue
                
                return {
                    "preview": df.head(rows).to_dict('records'),
                    "summary": {
                        "total_rows": len(df),
                        "columns": len(df.columns),
                        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error("Error obteniendo preview del dataset: %s", e)
            return None
    
    async def load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Carga el dataframe completo de un dataset por ID
        """
        try:
            for file_path in self.upload_dir.glob(f"{dataset_id}_*"):
                filename = file_path.name.replace(f"{dataset_id}_", "")
                
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(file_path)
                    logger.info("✅ Dataset CSV cargado: %s (%sx%s)", dataset_id, df.shape[0], df.shape[1])
                    return df
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                    logger.info("✅ Dataset Excel cargado: %s (%sx%s)", dataset_id, df.shape[0], df.shape[1])
                    return df
            
            logger.warning("⚠️ Dataset no encontrado: %s", dataset_id)
            return None
            
        except Exception as e:
            logger.error("❌ Error cargando dataset %s: %s", dataset_id, e)
            return None

    async def delete_dataset(self, dataset_id: str) -> bool:
        """
        Elimina un dataset
        """
        try:
            deleted = False
            for file_path in self.upload_dir.glob(f"{dataset_id}_*"):
                file_path.unlink()
                deleted = True
                logger.info("Dataset eliminado: %s", file_path)
            
            return deleted
            
        except Exception as e:
            logger.error("Error eliminando dataset: %s", e)
            return False
    
    async def list_datasets(self) -> List[DatasetInfo]:
        """
        Lista todos los datasets disponibles
        """
        try:
            datasets = []
            processed_ids = set()
            
            for file_path in self.upload_dir.glob("*"):
                if file_path.is_file():
                    # Extraer ID del dataset del nombre del archivo
                    parts = file_path.stem.split("_", 1)
                    if len(parts) >= 2:
                        dataset_id = parts[0]
                        
                        if dataset_id not in processed_ids:
                            dataset_info = await self.get_dataset_info(dataset_id)
                            if dataset_info:
                                datasets.append(dataset_info)
                                processed_ids.add(dataset_id)
            
            return datasets
            
        except Exception as e:
            logger.error("Error listando datasets: %s", e)
            return []

    async def get_dataset_statistics(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Calcula estadísticas completas del dataset
        """
        try:
            for file_path in self.upload_dir.glob(f"{dataset_id}_*"):
                filename = file_path.name.replace(f"{dataset_id}_", "")
                
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    continue
                
                # Identificar columnas numéricas y categóricas
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Calcular estadísticas básicas
                total_nulls = df.isnull().sum().sum()
                
                # Detectar posibles columnas target (últimas columnas o con patrones comunes)
                target_candidates = []
                for col in df.columns:
                    if any(keyword in col.lower() for keyword in ['target', 'label', 'outcome', 'diagnosis', 'class', 'result']):
                        target_candidates.append(col)
                
                # Si no hay candidatos obvios, usar las últimas 3 columnas
                if not target_candidates:
                    target_candidates = df.columns[-min(3, len(df.columns)):].tolist()
                
                # Calcular edad media si existe columna de edad
                average_age = None
                age_col = None
                for col in numeric_cols:
                    if any(keyword in col.lower() for keyword in ['age', 'edad', 'anos', 'años']):
                        age_col = col
                        average_age = float(df[col].mean())
                        break
                
                statistics = {
                    "overview": {
                        "patients": len(df),
                        "total_rows": len(df),
                        "total_columns": len(df.columns),
                        "total_nulls": int(total_nulls),
                        "average_age": average_age,
                        "numeric_columns": len(numeric_cols),
                        "categorical_columns": len(categorical_cols)
                    },
                    "target": {
                        "candidates": target_candidates,
                        "selected": None  # El frontend puede establecer esto
                    },
                    "columns": {
                        "numeric": numeric_cols,
                        "categorical": categorical_cols,
                        "all": df.columns.tolist()
                    },
                    "missing_values": {
                        col: int(df[col].isnull().sum()) for col in df.columns
                    },
                    "summary_stats": {}
                }
                
                # Estadísticas descriptivas para columnas numéricas (primeras 5)
                for col in numeric_cols[:5]:
                    try:
                        statistics["summary_stats"][col] = {
                            "mean": float(df[col].mean()),
                            "std": float(df[col].std()),
                            "min": float(df[col].min()),
                            "max": float(df[col].max()),
                            "median": float(df[col].median()),
                            "q25": float(df[col].quantile(0.25)),
                            "q75": float(df[col].quantile(0.75))
                        }
                    except:
                        pass
                
                # Correlaciones (solo si hay suficientes columnas numéricas)
                if len(numeric_cols) >= 2:
                    try:
                        corr_matrix = df[numeric_cols[:10]].corr()
                        statistics["correlation"] = {
                            "labels": corr_matrix.columns.tolist(),
                            "matrix": corr_matrix.values.tolist()
                        }
                    except:
                        statistics["correlation"] = {"labels": [], "matrix": []}
                else:
                    statistics["correlation"] = {"labels": [], "matrix": []}
                
                # Histogramas para primeras 3 columnas numéricas
                statistics["histograms"] = []
                for col in numeric_cols[:3]:
                    try:
                        hist, bins = pd.cut(df[col].dropna(), bins=10, retbins=True, duplicates='drop')
                        counts = hist.value_counts().sort_index().tolist()
                        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
                        statistics["histograms"].append({
                            "column": col,
                            "bins": bin_labels[:len(counts)],
                            "counts": counts
                        })
                    except:
                        pass
                
                # Boxplots (resumen de cuartiles)
                statistics["boxplots"] = []
                for col in numeric_cols[:3]:
                    try:
                        statistics["boxplots"].append({
                            "column": col,
                            "min": float(df[col].min()),
                            "q1": float(df[col].quantile(0.25)),
                            "median": float(df[col].median()),
                            "q3": float(df[col].quantile(0.75)),
                            "max": float(df[col].max())
                        })
                    except:
                        pass
                
                # 🆕 Información detallada de columnas numéricas para gráficas
                statistics["numeric_columns"] = {}
                for col in numeric_cols[:10]:  # Top 10 columnas numéricas
                    try:
                        col_data = df[col].dropna()
                        statistics["numeric_columns"][col] = {
                            "mean": float(col_data.mean()),
                            "std": float(col_data.std()),
                            "min": float(col_data.min()),
                            "max": float(col_data.max()),
                            "unique_count": int(col_data.nunique()),
                            "distribution": None  # Se llenará con histograma si es necesario
                        }
                        
                        # Calcular histograma/distribución
                        try:
                            hist, bins = pd.cut(col_data, bins=min(10, col_data.nunique()), retbins=True, duplicates='drop')
                            counts = hist.value_counts().sort_index()
                            bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
                            statistics["numeric_columns"][col]["distribution"] = {
                                "bins": bin_labels[:len(counts)],
                                "counts": counts.tolist()
                            }
                        except:
                            pass
                    except:
                        pass
                
                # 🆕 Información detallada de columnas categóricas para gráficas
                statistics["categorical_columns"] = {}
                for col in categorical_cols[:10]:  # Top 10 columnas categóricas
                    try:
                        col_data = df[col].dropna()
                        value_counts = col_data.value_counts()
                        statistics["categorical_columns"][col] = {
                            "unique_count": int(col_data.nunique()),
                            "most_common": value_counts.index[0] if len(value_counts) > 0 else None,
                            "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                            "value_counts": {str(k): int(v) for k, v in value_counts.head(20).items()}  # Top 20 valores
                        }
                    except:
                        pass
                
                return statistics
            
            return None
            
        except Exception as e:
            logger.error("Error calculando estadísticas del dataset %s: %s", dataset_id, e)
            return None

    async def get_columns_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información detallada de las columnas del dataset
        """
        try:
            for file_path in self.upload_dir.glob(f"{dataset_id}_*"):
                filename = file_path.name.replace(f"{dataset_id}_", "")
                
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    continue
                
                columns_info = {
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
                    "unique_counts": {col: int(df[col].nunique()) for col in df.columns},
                    "total_rows": len(df)
                }
                
                return columns_info
            
            return None
            
        except Exception as e:
            logger.error("Error obteniendo info de columnas del dataset %s: %s", dataset_id, e)
            return None
    
    def get_dataframe(self, dataset_id: str) -> pd.DataFrame:
        """
        Carga y devuelve el dataframe de un dataset por ID
        IMPORTANTE: Este método es SÍNCRONO (no async) para uso en orquestador
        """
        try:
            # Buscar el archivo por ID
            for file_path in self.upload_dir.glob(f"{dataset_id}_*"):
                filename = file_path.name.replace(f"{dataset_id}_", "")
                
                logger.info("📂 Cargando dataset desde: %s", file_path)
                
                # Leer dataset según formato
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    continue
                
                logger.info("✅ Dataset cargado: %sx%s", df.shape[0], df.shape[1])
                return df
            
            # Si no se encontró el archivo
            raise FileNotFoundError(f"Dataset {dataset_id} no encontrado")
            
        except Exception as e:
            logger.error("❌ Error cargando dataframe: %s", e)
            raise Exception(f"Error cargando dataset: {str(e)}")
