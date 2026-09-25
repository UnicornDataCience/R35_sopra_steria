"""
Servicio para orquestación de agentes médicos (wrapper sobre LangGraph)
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import asyncio
import inspect

from api.models.schemas import ChatResponse
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

class OrchestratorService:
    """Servicio fachada para el orquestador de LangGraph.
    Expone una interfaz estable para los routers y servicios de la API.
    """

    def __init__(self):
        self.orchestrator = None
        self.agents: Dict[str, Any] = {}
        self._initialize_orchestrator()

    def _initialize_orchestrator(self) -> None:
        """Inicializa el orquestador LangGraph y agentes disponibles."""
        try:
            from src.orchestration.langgraph_orchestrator import MedicalAgentsOrchestrator
            from src.agents.coordinator_agent import CoordinatorAgent
            from src.agents.analyzer_agent import ClinicalAnalyzerAgent
            from src.agents.generator_agent import SyntheticGeneratorAgent
            try:
                from src.agents.validator_agent import MedicalValidatorAgent  # type: ignore
            except Exception:
                MedicalValidatorAgent = None  # type: ignore
            try:
                from src.agents.simulator_agent import PatientSimulatorAgent  # type: ignore
            except Exception:
                PatientSimulatorAgent = None  # type: ignore
            try:
                from src.agents.evaluator_agent import UtilityEvaluatorAgent  # type: ignore
            except Exception:
                UtilityEvaluatorAgent = None  # type: ignore

            self.agents = {
                "coordinator": CoordinatorAgent(),
                "analyzer": ClinicalAnalyzerAgent(),
                "generator": SyntheticGeneratorAgent(),
            }
            if MedicalValidatorAgent:
                self.agents["validator"] = MedicalValidatorAgent()
            if PatientSimulatorAgent:
                self.agents["simulator"] = PatientSimulatorAgent()
            if UtilityEvaluatorAgent:
                self.agents["evaluator"] = UtilityEvaluatorAgent()

            self.orchestrator = MedicalAgentsOrchestrator(self.agents)
            logger.info("✅ LangGraph MedicalAgentsOrchestrator inicializado con %d agentes", len(self.agents))
        except Exception as e:
            logger.error("Error inicializando orquestador LangGraph: %s", e)
            self._initialize_fallback()

    def _initialize_fallback(self) -> None:
        """Fallback simple si LangGraph no está disponible."""
        agents = list(self.agents.keys()) or ["coordinator", "analyzer", "generator"]

        class MockOrchestrator:
            def __init__(self, agents_list: List[str]):
                self._agents_list = agents_list

            async def process_user_input(self, user_input: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
                ctx = context or {}
                preferred = ctx.get("preferred_agent") or "coordinator"
                return {
                    "success": True,
                    "agent": preferred,
                    "message": f"[mock] {preferred} procesó: {user_input[:200]}",
                    "metadata": {"mode": "mock", "context": ctx},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            def get_available_agents(self) -> List[str]:
                return list(self._agents_list)

            def get_available_transitions(self) -> Dict[str, List[str]]:
                return {
                    "coordinator": ["analyzer", "generator", "validator"],
                    "analyzer": ["generator", "validator"],
                    "generator": ["validator"],
                    "validator": [],
                }

            def get_status(self) -> Dict[str, Any]:
                return {
                    "engine": "mock",
                    "agents": self.get_available_agents(),
                    "initialized": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            def get_workflow_statistics(self) -> Dict[str, Any]:
                trans = self.get_available_transitions()
                return {
                    "nodes": len(self.get_available_agents()),
                    "edges": sum(len(v) for v in trans.values()),
                    "last_update": datetime.now(timezone.utc).isoformat(),
                }

        self.orchestrator = MockOrchestrator(agents)
        logger.warning("⚠️ Usando orquestador mock")

    def _build_chat_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normaliza cualquier payload a un dict estable para los routers."""
        base_message = (
            payload.get("message")
            or payload.get("result")
            or payload.get("response")
            or ""
        )
        base_agent = payload.get("agent") or payload.get("agent_used") or "coordinator"
        base_success = bool(payload.get("success", True))
        base_metadata = payload.get("metadata") or {}
        base_ts = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
        return {
            "message": base_message,
            "agent": base_agent,
            "success": base_success,
            "metadata": base_metadata,
            "timestamp": base_ts,
        }

    async def _maybe_await(self, func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        preferred_agent: Optional[str] = None,
    ):
        ctx = dict(context or {})
        if preferred_agent:
            ctx["preferred_agent"] = preferred_agent

        if hasattr(self.orchestrator, "process_user_input"):
            raw = await self._maybe_await(self.orchestrator.process_user_input, message, ctx)
            return self._build_chat_response(raw)

        agent_name = preferred_agent or "coordinator"
        agent = self.agents.get(agent_name)
        if agent and hasattr(agent, "safe_process"):
            raw = await agent.safe_process(message, ctx)  # type: ignore
            raw["agent"] = agent_name
            raw["timestamp"] = datetime.now(timezone.utc).isoformat()
            return self._build_chat_response(raw)

        return self._build_chat_response(
            {
                "success": True,
                "agent": agent_name,
                "message": f"[fallback] {agent_name} procesó: {message[:200]}",
                "metadata": {"mode": "fallback", "context": ctx},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def process_clinical_history(self, context: Optional[Dict[str, Any]] = None):
        """Ejecuta el pipeline determinista de historial de cohorte (si está disponible)."""
        if hasattr(self.orchestrator, "process_clinical_history"):
            return await self._maybe_await(self.orchestrator.process_clinical_history, dict(context or {}))
        return {
            "error": "El pipeline de historial de cohorte no está disponible (orquestador en modo mock).",
            "steps": {},
        }

    async def process_user_input(self, message: str, context: Optional[Dict[str, Any]] = None):
        ctx = dict(context or {})
        if hasattr(self.orchestrator, "process_user_input"):
            return await self._maybe_await(self.orchestrator.process_user_input, message, ctx)
        resp = await self.process_message(message, ctx)
        if hasattr(resp, "model_dump"):
            return resp.model_dump()  # type: ignore
        return resp

    def process_user_input_sync(self, message: str, context: Optional[Dict[str, Any]] = None):
        return asyncio.run(self.process_user_input(message, context))

    def get_status(self) -> Dict[str, Any]:
        if hasattr(self.orchestrator, "get_status"):
            try:
                return self.orchestrator.get_status()
            except Exception as e:
                logger.warning("get_status (orchestrator) falló: %s", e)
        return {
            "engine": type(self.orchestrator).__name__,
            "agents": self.get_available_agents(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_available_transitions(self) -> Dict[str, List[str]]:
        if hasattr(self.orchestrator, "get_available_transitions"):
            try:
                return self.orchestrator.get_available_transitions()
            except Exception as e:
                logger.warning("get_available_transitions falló: %s", e)
        agents = self.get_available_agents()
        return {a: [b for b in agents if b != a] for a in agents}

    def get_workflow_statistics(self) -> Dict[str, Any]:
        if hasattr(self.orchestrator, "get_workflow_statistics"):
            try:
                return self.orchestrator.get_workflow_statistics()
            except Exception as e:
                logger.warning("get_workflow_statistics falló: %s", e)
        trans = self.get_available_transitions()
        return {
            "nodes": len(self.get_available_agents()),
            "edges": sum(len(v) for v in trans.values()),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

    def get_available_agents(self) -> List[str]:
        if hasattr(self.orchestrator, "get_available_agents"):
            try:
                return list(self.orchestrator.get_available_agents())
            except Exception:
                pass
        return list(self.agents.keys()) or ["coordinator"]

    def get_agent_capabilities(self, agent_name: str) -> List[str]:
        agent = self.agents.get(agent_name)
        if hasattr(agent, "get_capabilities"):
            try:
                return list(agent.get_capabilities())  # type: ignore
            except Exception:
                pass
        defaults = {
            "coordinator": ["route_requests", "manage_workflow", "coordinate_agents"],
            "analyzer": ["analyze_dataset", "data_quality_check"],
            "generator": ["generate_synthetic_data"],
            "validator": ["validate_synthetic_data"],
            "simulator": ["simulate_patient_data"],
            "evaluator": ["evaluate_utility"],
        }
        return defaults.get(agent_name, [])

_orchestrator_singleton: OrchestratorService | None = None

def get_orchestrator() -> OrchestratorService:
    global _orchestrator_singleton
    if _orchestrator_singleton is None:
        _orchestrator_singleton = OrchestratorService()
    return _orchestrator_singleton