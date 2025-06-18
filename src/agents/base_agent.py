from typing import Any, Dict, List, Optional, Union
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_openai import AzureChatOpenAI  # Cambiar import
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class BaseAgentConfig(BaseModel):
    """Configuración base para agentes"""
    name: str
    description: str
    system_prompt: str
    temperature: float = 0.1
    model: str = "gpt-4"
    max_tokens: int = 2000

class BaseLLMAgent:
    """Agente base que usa LLM con Azure OpenAI"""
    
    def __init__(self, config: BaseAgentConfig, tools: List[BaseTool] = None):
        self.config = config
        self.tools = tools or []
        
        # Verificar configuración de Azure
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        if not all([azure_endpoint, azure_api_key, azure_deployment]):
            raise ValueError("Configuración de Azure OpenAI incompleta")
        
        try:
            # Inicializar Azure OpenAI
            self.llm = AzureChatOpenAI(
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                model=config.model
            )
            
            # Crear prompt template
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", config.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Crear agente
            if self.tools:
                self.agent = create_openai_functions_agent(
                    llm=self.llm,
                    tools=self.tools,
                    prompt=self.prompt
                )
                self.agent_executor = AgentExecutor(
                    agent=self.agent,
                    tools=self.tools,
                    verbose=True,
                    return_intermediate_steps=True,
                    handle_parsing_errors=True
                )
            else:
                self.agent_executor = None
            
            # Memoria para conversación
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
        except Exception as e:
            raise RuntimeError(f"Error inicializando agente {config.name}: {e}")
    
    async def process(self, input_text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Procesa input del usuario y retorna respuesta"""
        try:
            if self.agent_executor:
                # Usar agente con tools
                result = await self.agent_executor.ainvoke({
                    "input": input_text,
                    "chat_history": self.memory.chat_memory.messages,
                    "context": context or {}
                })
                
                response = {
                    "message": result["output"],
                    "intermediate_steps": result.get("intermediate_steps", []),
                    "agent": self.config.name
                }
            else:
                # LLM directo sin tools
                messages = [
                    ("system", self.config.system_prompt),
                    ("human", f"Context: {context}\n\nUser: {input_text}")
                ]
                
                result = await self.llm.ainvoke(messages)
                response = {
                    "message": result.content,
                    "agent": self.config.name
                }
            
            # Guardar en memoria
            self.memory.save_context(
                {"input": input_text},
                {"output": response["message"]}
            )
            
            return response
            
        except Exception as e:
            return {
                "message": f"Error en {self.config.name}: {str(e)}",
                "error": True,
                "agent": self.config.name
            }
    
    def clear_memory(self):
        """Limpia la memoria de conversación"""
        self.memory.clear()