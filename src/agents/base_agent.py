"""
Agente Base - Clase padre para todos los agentes especializados del sistema.

Este módulo define la estructura base y funcionalidades comunes que todos
los agentes médicos especializados deben implementar.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Importaciones de LangChain
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory

# Importar configuración unificada de LLMs
try:
    from src.config.llm_config import unified_llm_config
    LLM_CONFIG_AVAILABLE = True
except ImportError:
    try:
        from config.llm_config import unified_llm_config
        LLM_CONFIG_AVAILABLE = True
    except ImportError:
        LLM_CONFIG_AVAILABLE = False
        print("⚠️ Configuración unificada de LLM no disponible - usando modo simulado")

# Cargar las variables de entorno desde el archivo .env
# Esto asegura que estén disponibles tan pronto como se importe este módulo.
load_dotenv()

class BaseAgentConfig(BaseModel):
    """Configuración base para todos los agentes."""
    name: str
    description: str
    system_prompt: str = ""
    max_tokens: int = 1500
    temperature: float = 0.2
    model: str = "gpt-4"
    
    class Config:
        arbitrary_types_allowed = True

class BaseLLMAgent(ABC):
    """
    Clase base abstracta para todos los agentes LLM especializados.
    Define la interfaz común y funcionalidades compartidas.
    """
    
    def __init__(self, config: BaseAgentConfig, tools: List[BaseTool] = None):
        self.config = config
        self.tools = tools or []
        self.name = config.name
        self.description = config.description
        
        # Usar configuración unificada de LLMs
        if LLM_CONFIG_AVAILABLE:
            try:
                self.llm = unified_llm_config.create_llm(
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
                self._llm_available = True
                provider = unified_llm_config.active_provider
                print(f"✅ Agente '{self.name}' inicializado con {provider}")
            except Exception as e:
                print(f"⚠️ Error al conectar LLM para agente '{self.name}': {e}")
                self._llm_available = False
                self.llm = None
        else:
            self._llm_available = False
            self.llm = None
            print(f"⚠️ Agente '{self.name}' en modo simulado - LLM no disponible")
        
        # Configurar prompts y agentes solo si tenemos LLM disponible
        if self.llm and self.config.system_prompt:
            if self.tools:
                # Prompt completo para agentes con herramientas
                self.prompt = ChatPromptTemplate.from_messages([
                    ("system", self.config.system_prompt),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
                ])
                self.agent = create_openai_functions_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
                self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
            else:
                # Prompt simple para agentes sin herramientas
                self.prompt = ChatPromptTemplate.from_messages([
                    ("system", self.config.system_prompt),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "{input}"),
                ])
                # Si no hay herramientas, creamos una cadena simple de LLM
                self.agent_executor = self.prompt | self.llm
        else:
            # Modo simulado - sin agente real
            self.prompt = None
            self.agent = None
            self.agent_executor = None
        
        # Memoria para conversación
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        print(f"✅ Agente '{self.name}' inicializado correctamente.")

    def _extract_content(self, llm_response):
        """Extrae el contenido de la respuesta del LLM, maneja tanto strings como objetos"""
        if hasattr(llm_response, 'content'):
            return llm_response.content
        elif isinstance(llm_response, str):
            return llm_response
        elif isinstance(llm_response, dict) and 'content' in llm_response:
            return llm_response['content']
        else:
            return str(llm_response)

    @abstractmethod
    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Método abstracto principal para procesar la entrada del usuario.
        Este método DEBE ser implementado por todas las clases hijas.
        """
        pass

    async def analyze_dataset(self, dataframe, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Método para análisis de datasets. NO es abstracto.
        Las clases hijas pueden sobreescribirlo si tienen esta capacidad.
        """
        return {
            "message": f"El agente '{self.name}' no tiene la capacidad de analizar datasets directamente.",
            "agent": self.name,
            "error": True
        }

    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del agente."""
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config.model_dump()
        }
    
    def _generate_mock_response(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Genera una respuesta simulada cuando LLM no está disponible.
        """
        provider = unified_llm_config.active_provider if LLM_CONFIG_AVAILABLE else "none"
        return {
            "success": True,
            "result": f"🤖 **{self.name} (Modo Simulado)**\n\nHe recibido tu solicitud: *\"{input_text[:100]}...\"*\n\n📋 **Procesamiento simulado:**\n• Análisis completado\n• Datos procesados correctamente\n• Resultados generados\n\n*Nota: Proveedor activo: {provider}. Para funcionalidad completa, configura un LLM correctamente.*",
            "metadata": {
                "agent": self.name,
                "mode": "simulated",
                "llm_available": False,
                "provider": provider,
                "input_length": len(input_text) if input_text else 0
            }
        }

    def is_llm_available(self) -> bool:
        """Verifica si LLM está disponible para este agente."""
        return getattr(self, '_llm_available', False)

    async def safe_process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Procesa la entrada de forma segura, usando modo simulado si LLM no está disponible.
        """
        try:
            if self.is_llm_available():
                return await self.process(input_text, context)
            else:
                return self._generate_mock_response(input_text, context)
        except Exception as e:
            print(f"⚠️ Error en agente '{self.name}': {e}")
            return self._generate_mock_response(input_text, context)
    
    def _extract_response_text(self, llm_response) -> str:
        """
        Extrae el texto de una respuesta LLM manejando diferentes formatos.
        
        Args:
            llm_response: Respuesta del LLM (puede ser objeto con .content o string directo)
            
        Returns:
            str: Texto extraído de la respuesta
        """
        if hasattr(llm_response, 'content'):
            # Respuesta de AgentExecutor con herramientas
            return llm_response.content
        elif isinstance(llm_response, str):
            # Respuesta directa de LLM simple
            return llm_response
        else:
            # Otros tipos de respuesta
            return str(llm_response)
