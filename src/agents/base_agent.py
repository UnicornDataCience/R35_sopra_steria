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
        
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        if not all([azure_endpoint, api_key, deployment, api_version]):
            raise ValueError("Configuración de Azure OpenAI incompleta en el entorno.")
        
        try:
            self.llm = AzureChatOpenAI(
                azure_deployment=deployment,
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            if self.config.system_prompt:
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
            
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            print(f"✅ Agente '{self.name}' inicializado correctamente.")

        except Exception as e:
            raise RuntimeError(f"Error inicializando el LLM para el agente '{self.config.name}': {e}")

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
