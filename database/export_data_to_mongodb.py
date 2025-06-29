# Requisitos Previos e Instalación
# Programas necesarios:
# - MongoDB Server (Community Edition)
# - Configurar puerto local 27017
from pymongo import MongoClient
from datetime import datetime, timedelta
import json
import time
from datetime import datetime, timezone, timedelta  # ✅ Importar timezone
import os

script_dir = os.getcwd()
JSON_PATH = os.path.abspath(os.path.join(script_dir, 'data', 'synthetic', 'datos_sinteticos_sdv.json'))
JSON_PATH_2 = os.path.abspath(os.path.join(script_dir, 'data', 'synthetic', 'datos_sinteticos_tvae.json'))

class MongoDBTTLManager:
    def __init__(self, connection_string="mongodb://localhost:27017/", 
                database_name="medical_data", collection_name="patient_records"):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
    def setup_custom_ttl_index(self, field_name="created_at", expire_seconds=3600):
        try:
            self.collection.create_index(field_name, expireAfterSeconds=expire_seconds)
            print(f"✅ Índice TTL personalizado creado en campo '{field_name}': {expire_seconds} segundos")
        except Exception as e:
            print(f"❌ Error creando índice TTL personalizado: {e}")
    
    def insert_patient_data(self, patient_data, custom_ttl_seconds=None):
        try:
            document = patient_data.copy()
            # ✅ CAMBIO: Usar timezone.utc en lugar de datetime.UTC
            document["created_at"] = datetime.now(timezone.utc)
            document["inserted_timestamp"] = datetime.now(timezone.utc)
            
            if custom_ttl_seconds:
                document["ttl"] = custom_ttl_seconds
            
            result = self.collection.insert_one(document)
            print(f"✅ Documento insertado con ID: {result.inserted_id}")
            return result.inserted_id
            
        except Exception as e:
            print(f"❌ Error insertando documento: {e}")
            return None
    
    def insert_multiple_patients(self, patients_list):
        try:
            documents = []
            for patient in patients_list:
                doc = patient.copy()
                # ✅ CAMBIO: Usar timezone.utc en lugar de datetime.UTC
                doc["created_at"] = datetime.now(timezone.utc)
                doc["inserted_timestamp"] = datetime.now(timezone.utc)
                documents.append(doc)
            
            result = self.collection.insert_many(documents)
            print(f"✅ {len(result.inserted_ids)} documentos insertados")
            return result.inserted_ids
            
        except Exception as e:
            print(f"❌ Error insertando múltiples documentos: {e}")
            return []

import pandas as pd

def load_and_insert_with_pandas(file_path, mongo_manager):
    """
    Carga datos usando pandas.read_json con soporte para JSON Lines
    """
    try:
        # Usar pandas para leer JSON Lines
        df = pd.read_json(file_path, lines=True)
        print(f"✅ DataFrame cargado: {len(df)} registros")
        
        # Convertir DataFrame a lista de diccionarios
        data = df.to_dict('records')
        
        # Insertar en MongoDB
        mongo_manager.insert_multiple_patients(data)
        
    except Exception as e:
        print(f"❌ Error cargando con pandas: {e}")


load_and_insert_with_pandas(JSON_PATH, MongoDBTTLManager())
load_and_insert_with_pandas(JSON_PATH_2, MongoDBTTLManager())