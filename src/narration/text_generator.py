import os
from typing import Optional
from src.utils.logging_config import get_logger
from transformers import pipeline

logger = get_logger(__name__)

_llm = None
_hf_pipe = None

def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    try:
        from src.config.llm_config import unified_llm_config
        _llm = unified_llm_config.create_llm()
        logger.info("Narration: usando LLM unificado (%s)", unified_llm_config.active_provider)
        return _llm
    except Exception as e:
        logger.warning("Narration: LLM unificado no disponible: %s", e)
        return None

def _get_hf_pipeline():
    global _hf_pipe
    if _hf_pipe is not None:
        return _hf_pipe
    model = os.getenv("HF_NARRATION_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    _hf_pipe = pipeline("text-generation", model=model)
    logger.info("Narration: usando Hugging Face pipeline (%s)", model)
    return _hf_pipe

def generate_clinical_note(prompt: str, max_new_tokens: int = 200) -> str:
    """Genera una nota clínica sintética.
    Prefiere el LLM unificado; si no, usa HF pipeline bajo demanda.
    Control por env: FORCE_HF_NARRATION=true fuerza Hugging Face.
    """
    force_hf = os.getenv("FORCE_HF_NARRATION", "false").lower() == "true"

    if not force_hf:
        llm = _get_llm()
        if llm is not None:
            try:
                content = llm.invoke(f"Genera una nota clínica concisa y estructurada:\n\n{prompt}")
                return content if isinstance(content, str) else getattr(content, 'content', str(content))
            except Exception as e:
                logger.warning("Narration: error usando LLM unificado, fallback a HF: %s", e)

    pipe = _get_hf_pipeline()
    out = pipe(prompt, max_new_tokens=max_new_tokens)[0]
    return out.get('generated_text') or str(out)