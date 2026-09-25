"""
Configuración unificada de LLMs - Soporte para Azure OpenAI, Ollama y Grok
Permite cambiar fácilmente entre proveedores manteniendo compatibilidad.
"""

import os
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import logging

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)

# FORZAR GROQ - Sobrescribir variable del sistema
if os.getenv('FORCE_GROQ', 'false').lower() == 'true':
    os.environ['LLM_PROVIDER'] = 'groq'
    logger.debug("[DEBUG] Variable LLM_PROVIDER forzada a 'groq'")

class BaseLLMProvider(ABC):
    """Clase base para todos los proveedores de LLM"""
    
    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.llm = None
    
    @abstractmethod
    def create_llm(self, temperature: float = 0.1, max_tokens: int = 2000, **kwargs):
        """Crea una instancia del LLM"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Prueba la conexión al proveedor"""
        pass
    
    @property
    def status_info(self) -> Dict[str, Any]:
        """Información del estado del proveedor"""
        return {
            "provider": self.name,
            "available": self.available,
            "model": getattr(self, 'model', 'Unknown')
        }

class AzureOpenAIProvider(BaseLLMProvider):
    """Proveedor para Azure OpenAI (mantiene compatibilidad)"""
    
    def __init__(self):
        super().__init__("Azure OpenAI")
        try:
            from langchain_openai import AzureChatOpenAI
            self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4-TFM")
            self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            self.model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4")
            
            if all([self.endpoint, self.api_key, self.deployment]):
                self.available = True
            
        except ImportError:
            logger.warning("Azure OpenAI no disponible - langchain_openai no instalado")
    
    def create_llm(self, temperature: float = 0.1, max_tokens: int = 2000, **kwargs):
        if not self.available:
            raise RuntimeError("Azure OpenAI no está disponible")
        
        from langchain_openai import AzureChatOpenAI
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
        if not self.available:
            return False
        try:
            llm = self.create_llm()
            response = llm.invoke("Test connection")
            return True
        except Exception as e:
            error_msg = str(e)
            if "DeploymentNotFound" in error_msg:
                logger.error("Azure: Deployment '%s' no encontrado", self.deployment)
            else:
                logger.error("Azure: Error de conexión - %s", e)
            return False

class OllamaProvider(BaseLLMProvider):
    """Proveedor para Ollama local"""
    
    def __init__(self):
        super().__init__("Ollama")
        try:
            from langchain_ollama import OllamaLLM
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model = os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
            self.available = True
        except ImportError:
            logger.warning("Ollama no disponible - langchain_ollama no instalado. Instalar con: pip install langchain-ollama")
    
    def create_llm(self, temperature: float = 0.1, max_tokens: int = 2000, **kwargs):
        if not self.available:
            raise RuntimeError("Ollama no está disponible")
        
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=temperature,
            num_predict=max_tokens
        )
    
    def test_connection(self) -> bool:
        if not self.available:
            return False
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [m['name'] for m in models]
                if any(self.model in model for model in available_models):
                    return True
                else:
                    logger.error("Ollama: Modelo '%s' no encontrado", self.model)
                    logger.error("Modelos disponibles: %s", available_models)
                    return False
            return False
        except Exception as e:
            logger.error("Ollama: Error de conexión - %s", e)
            logger.error("Verifica que Ollama esté ejecutándose en %s", self.base_url)
            return False

class GrokProvider(BaseLLMProvider):
    """Proveedor para Grok (X.AI)"""
    
    def __init__(self):
        super().__init__("Grok")
        try:
            self.api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
            self.base_url = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
            self.model = os.getenv("GROK_MODEL", "grok-beta")
            
            if self.api_key:
                self.available = True
        except Exception:
            pass
    
    def create_llm(self, temperature: float = 0.1, max_tokens: int = 2000, **kwargs):
        if not self.available:
            raise RuntimeError("Grok no está disponible")
        
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except ImportError:
            # Fallback usando requests directo  
            return GroqDirectLLM(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens
            )
    
    def test_connection(self) -> bool:
        # Evitar pruebas de red por defecto en import
        if not self.available:
            return False
        try:
            llm = self.create_llm()
            _ = llm.invoke("Test connection")
            return True
        except Exception as e:
            logger.error("Grok: Error de conexión - %s", e)
            return False

class GroqProvider(BaseLLMProvider):
    """Proveedor para Groq (diferente de Grok/X.AI)"""
    
    def __init__(self):
        super().__init__("Groq")
        try:
            self.api_key = os.getenv("GROQ_API_KEY")
            self.base_url = "https://api.groq.com/openai/v1"
            self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            self.temperature_default = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
            self.max_tokens_default = int(os.getenv("GROQ_MAX_TOKENS", "1500"))
            self.chunk_size_tokens = int(os.getenv("GROQ_CHUNK_SIZE", "3000"))
            
            if self.api_key:
                self.available = True
                logger.info("Groq configurado con modelo: %s", self.model)
        except Exception as e:
            logger.error("Error configurando Groq: %s", e)
    
    def create_llm(self, temperature: float = None, max_tokens: int = None, **kwargs):
        if not self.available:
            raise RuntimeError("Groq no está disponible")
        
        # Usar defaults optimizados si no vienen parámetros
        temperature = self.temperature_default if temperature is None else temperature
        max_tokens = self.max_tokens_default if max_tokens is None else max_tokens
        
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=self.model,
                groq_api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except ImportError:
            try:
                # Fallback usando langchain_openai con base_url personalizada
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=self.model,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except ImportError:
                # Fallback usando requests directo
                return GroqDirectLLM(
                    api_key=self.api_key,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
    
    # --- Utilidades integradas ---
    def estimate_tokens(self, text: str) -> int:
        """Estimación simple: ~1 token por 4 caracteres."""
        return max(1, len(text) // 4)
    
    def chunk_text(self, text: str, max_chunk_tokens: Optional[int] = None) -> list:
        """Divide el texto en chunks para evitar límites de tokens por request."""
        size = max_chunk_tokens or self.chunk_size_tokens
        max_chars = size * 4  # Aproximación
        chunks = []
        current = ""
        for paragraph in text.split("\n\n"):
            # +2 por los dos saltos agregados
            if len(current) + len(paragraph) + 2 <= max_chars:
                current += paragraph + "\n\n"
            else:
                if current:
                    chunks.append(current.strip())
                current = paragraph + "\n\n"
        if current:
            chunks.append(current.strip())
        return chunks
    
    def safe_invoke(self, llm, prompt: str, max_retries: int = 3) -> str:
        """Invoca el LLM manejando rate limits y reduciendo tamaño si es necesario."""
        for attempt in range(max_retries):
            try:
                return llm.invoke(prompt)
            except Exception as e:
                msg = str(e).lower()
                if ("rate_limit" in msg) or ("tpm" in msg) or ("413" in msg):
                    logger.warning("Rate limit detectado (intento %d/%d)", attempt + 1, max_retries)
                    if attempt < max_retries - 1:
                        # Reducir tamaño del prompt o esperar
                        if len(prompt) > 2000:
                            prompt = prompt[:2000] + "..."
                            logger.info("Reduciendo tamaño del prompt...")
                        else:
                            import time
                            logger.info("Esperando 5s antes de reintentar...")
                            time.sleep(5)
                        continue
                # Si no es rate limit u otros intentos agotados, propagar
                raise
        return ""
    
    def test_connection(self) -> bool:
        if not self.available:
            return False
        try:
            # Usar pocos tokens en el test
            llm = self.create_llm(max_tokens=100)
            _ = llm.invoke("Hello")
            return True
        except Exception as e:
            logger.error("Groq: Error de conexión - %s", e)
            return False

class GeminiProvider(BaseLLMProvider):
    """Proveedor para Google Gemini"""
    
    def __init__(self):
        super().__init__("Gemini")
        try:
            self.api_key = os.getenv("GEMINI_API_KEY")
            # El modelo es configurable vía GEMINI_MODEL; ajústalo al que cubran
            # tus créditos (p. ej. gemini-3.8-flash, gemini-3.8-pro).
            self.model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            
            if self.api_key:
                self.available = True
                logger.info("Gemini configurado con modelo: %s", self.model)
        except Exception as e:
            logger.error("Error configurando Gemini: %s", e)
    
    def create_llm(self, temperature: float = 0.1, max_tokens: int = 2000, **kwargs):
        if not self.available:
            raise RuntimeError("Gemini no está disponible")
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
        except ImportError:
            logger.error("langchain_google_genai no instalado. Instalar con: pip install langchain-google-genai")
            raise RuntimeError("langchain_google_genai no disponible")
    
    def test_connection(self) -> bool:
        if not self.available:
            return False
        try:
            llm = self.create_llm(max_tokens=100)
            _ = llm.invoke("Hello")
            return True
        except Exception as e:
            logger.error("Gemini: Error de conexión - %s", e)
            return False

class GroqDirectLLM:
    """Implementación directa para Groq usando requests"""
    
    def __init__(self, api_key: str, model: str, temperature: float = 0.1, max_tokens: int = 2000):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def invoke(self, prompt: str) -> str:
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")

class UnifiedLLMConfig:
    """Configuración unificada que maneja múltiples proveedores"""
    
    def __init__(self):
        # Inicializar proveedores
        self.providers = {
            "azure": AzureOpenAIProvider(),
            "ollama": OllamaProvider(), 
            "grok": GrokProvider(),
            "groq": GroqProvider(),
            "gemini": GeminiProvider()
        }
        
        # Selección perezosa: preferir env o Groq por defecto, sin pruebas de red
        self.active_provider = self._determine_active_provider()
        logger.info("Proveedor LLM activo: %s", self.active_provider)
    
    def _determine_active_provider(self) -> str:
        """Determina proveedor sin test de conexión (lazy)."""
        preferred = (os.getenv("LLM_PROVIDER") or "").lower()
        if not preferred:
            preferred = "gemini"  # Proveedor por defecto: Google Gemini
        logger.debug("Preferencia LLM_PROVIDER: %s", preferred)
        
        if preferred in self.providers and self.providers[preferred].available:
            return preferred
        
        # Orden de prioridad si el preferido no está disponible
        for name in ["gemini", "groq", "azure", "grok", "ollama"]:
            provider = self.providers.get(name)
            if provider and provider.available:
                return name
        
        logger.warning("Ningún proveedor LLM disponible - modo simulado activado")
        return "mock"
    
    def create_llm(self, temperature: float = 0.1, max_tokens: int = 2000, **kwargs):
        """Crea una instancia del LLM usando el proveedor activo"""
        if self.active_provider == "mock":
            return MockLLM()
        
        provider = self.providers[self.active_provider]
        return provider.create_llm(temperature, max_tokens, **kwargs)
    
    def test_connection(self) -> bool:
        """Prueba la conexión del proveedor activo"""
        if self.active_provider == "mock":
            return False
        
        return self.providers[self.active_provider].test_connection()
    
    def switch_provider(self, provider_name: str) -> bool:
        """Cambia el proveedor activo"""
        if provider_name not in self.providers:
            logger.error("Proveedor '%s' no válido", provider_name)
            return False
        
        provider = self.providers[provider_name]
        if not provider.available:
            logger.error("Proveedor '%s' no disponible", provider_name)
            return False
        
        if provider.test_connection():
            self.active_provider = provider_name
            logger.info("Cambiado a proveedor: %s", provider_name)
            return True
        else:
            logger.error("No se pudo conectar a '%s'", provider_name)
            return False
    
    @property
    def status_info(self) -> Dict[str, Any]:
        """Información del estado actual"""
        active_provider_info = {}
        if self.active_provider != "mock":
            active_provider_info = self.providers[self.active_provider].status_info
        
        return {
            "active_provider": self.active_provider,
            "available_providers": [name for name, p in self.providers.items() if p.available],
            "provider_details": active_provider_info
        }

class MockLLM:
    """LLM simulado para desarrollo sin conexión"""
    
    def invoke(self, prompt: str) -> str:
        return f'🤖 **Respuesta Simulada**\n\nHe recibido tu consulta: *\"{prompt[:100]}...\"*\n\n📋 **Procesamiento completado en modo simulado**\n\n*Configura un proveedor LLM (Azure, Ollama o Grok) para obtener respuestas reales.*'

# Instancia global unificada
unified_llm_config = UnifiedLLMConfig()