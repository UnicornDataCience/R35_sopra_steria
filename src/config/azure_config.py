import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from typing import Optional

load_dotenv()

class AzureOpenAIConfig:
    """Configuración centralizada para Azure OpenAI"""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4-TFM")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self.model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4")
        
        self._validate_config()
    
    def _validate_config(self):
        """Valida que la configuración esté completa"""
        required_vars = {
            "AZURE_OPENAI_ENDPOINT": self.endpoint,
            "AZURE_OPENAI_API_KEY": self.api_key,
            "AZURE_OPENAI_DEPLOYMENT": self.deployment
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Variables de entorno faltantes: {', '.join(missing_vars)}")
    
    def create_llm(self, temperature: float = 0.1, max_tokens: int = 2000) -> AzureChatOpenAI:
        """Crea instancia de Azure OpenAI LLM"""
        return AzureChatOpenAI(
            azure_deployment=self.deployment,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            temperature=temperature,
            max_tokens=max_tokens,
            model=self.model
        )
    
    def test_connection(self) -> bool:
        """Prueba la conexión a Azure OpenAI"""
        try:
            llm = self.create_llm()
            response = llm.invoke("Test connection")
            return True
        except Exception as e:
            print(f"Error de conexión: {e}")
            return False
    
    @property
    def status_info(self) -> dict:
        """Información del estado de la configuración"""
        return {
            "endpoint": f"{self.endpoint[:50]}..." if len(self.endpoint) > 50 else self.endpoint,
            "deployment": self.deployment,
            "model": self.model,
            "api_version": self.api_version,
            "api_key_configured": bool(self.api_key and len(self.api_key) > 10)
        }

# Instancia global
azure_config = AzureOpenAIConfig()