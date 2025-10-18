from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class AgentResponse(BaseModel):
    success: bool = True
    message: str
    agent: str
    payload: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None

class CoordinatorDecision(BaseModel):
    intention: str  # "conversacion" | "comando"
    agent: str      # "analyzer" | "generator" | "validator" | "simulator" | "evaluator" | "coordinator"
    is_medical_query: bool
    parameters: Dict[str, Any] = Field(default_factory=dict)
    message: str

class Context(BaseModel):
    dataset_uploaded: bool = False
    filename: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    universal_analysis: Dict[str, Any] = Field(default_factory=dict)
    selected_columns: Optional[List[str]] = None
    dataframe: Optional[Any] = None
    synthetic_data: Optional[Any] = None
