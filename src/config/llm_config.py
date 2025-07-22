"""
Configuración unificada de LLMs - Soporte para Azure OpenAI, Ollama y Grok
Permite cambiar fácilmente entre proveedores manteniendo compatibilidad.
"""

import os
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv
from abc import ABC, abstractmethod

# Cargar variables de entorno
load_dotenv()

# FORZAR GROQ - Sobrescribir variable del sistema
if os.getenv('FORCE_GROQ', 'false').lower() == 'true':
    os.environ['LLM_PROVIDER'] = 'groq'
    print("🚀 [DEBUG] Variable LLM_PROVIDER forzada a 'groq'")

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
            print("⚠️ Azure OpenAI no disponible - langchain_openai no instalado")
    
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
                print(f"❌ Azure: Deployment '{self.deployment}' no encontrado")
            else:
                print(f"❌ Azure: Error de conexión - {e}")
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
            print("⚠️ Ollama no disponible - langchain_ollama no instalado")
            print("   Instalar con: pip install langchain-ollama")
    
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
                    print(f"❌ Ollama: Modelo '{self.model}' no encontrado")
                    print(f"   Modelos disponibles: {available_models}")
                    return False
            return False
        except Exception as e:
            print(f"❌ Ollama: Error de conexión - {e}")
            print(f"   Verifica que Ollama esté ejecutándose en {self.base_url}")
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
            return GrokDirectLLM(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens
            )
    
    def test_connection(self) -> bool:
        if not self.available:
            return False
        try:
            llm = self.create_llm()
            response = llm.invoke("Test connection")
            return True
        except Exception as e:
            print(f"❌ Grok: Error de conexión - {e}")
            return False

class GroqProvider(BaseLLMProvider):
    """Proveedor para Groq (diferente de Grok/X.AI)"""
    
    def __init__(self):
        super().__init__("Groq")
        try:
            self.api_key = os.getenv("GROQ_API_KEY")
            self.base_url = "https://api.groq.com/openai/v1"
            self.model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
            
            if self.api_key:
                self.available = True
                print(f"✅ Groq configurado con modelo: {self.model}")
        except Exception as e:
            print(f"❌ Error configurando Groq: {e}")
    
    def create_llm(self, temperature: float = 0.1, max_tokens: int = 2000, **kwargs):
        if not self.available:
            raise RuntimeError("Groq no está disponible")
        
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
    
    def test_connection(self) -> bool:
        if not self.available:
            return False
        try:
            llm = self.create_llm()
            response = llm.invoke("Hello")
            return True
        except Exception as e:
            print(f"❌ Groq: Error de conexión - {e}")
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
            "groq": GroqProvider()
        }
        
        # Determinar proveedor activo
        self.active_provider = self._determine_active_provider()
        print(f"🔧 Proveedor LLM activo: {self.active_provider}")
    
    def _determine_active_provider(self) -> str:
        """Determina qué proveedor usar basado en disponibilidad y configuración"""
        
        # 1. Verificar si hay preferencia explícita
        preferred = os.getenv("LLM_PROVIDER", "").lower()
        print(f"🔍 Preferencia LLM_PROVIDER: {preferred}")
        
        if preferred in self.providers and self.providers[preferred].available:
            print(f"🧪 Probando conexión para {preferred}...")
            if self.providers[preferred].test_connection():
                print(f"✅ {preferred} funciona - seleccionado")
                return preferred
            else:
                print(f"❌ {preferred} falló test de conexión")
        
        # 2. Buscar el primer proveedor disponible y funcional
        priority_order = ["groq", "azure", "grok", "ollama"]
        print(f"🔄 Probando proveedores en orden: {priority_order}")
        
        for provider_name in priority_order:
            provider = self.providers[provider_name]
            print(f"   🧪 {provider_name}: disponible={provider.available}")
            if provider.available and provider.test_connection():
                print(f"   ✅ {provider_name} seleccionado")
                return provider_name
            else:
                print(f"   ❌ {provider_name} falló")
        
        # 3. Fallback a modo simulado
        print("⚠️ Ningún proveedor LLM disponible - modo simulado activado")
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
            print(f"❌ Proveedor '{provider_name}' no válido")
            return False
        
        provider = self.providers[provider_name]
        if not provider.available:
            print(f"❌ Proveedor '{provider_name}' no disponible")
            return False
        
        if provider.test_connection():
            self.active_provider = provider_name
            print(f"✅ Cambiado a proveedor: {provider_name}")
            return True
        else:
            print(f"❌ No se pudo conectar a '{provider_name}'")
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
        return f"🤖 **Respuesta Simulada**\n\nHe recibido tu consulta: *\"{prompt[:100]}...\"*\n\n📋 **Procesamiento completado en modo simulado**\n\n*Configura un proveedor LLM (Azure, Ollama o Grok) para obtener respuestas reales.*"

# Instancia global unificada
unified_llm_config = UnifiedLLMConfig()
