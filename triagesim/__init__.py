"""
triagesim

Simulated nurse <-> patient emergency department triage dialogues grounded in structured EHR
cases (e.g. MIMIC-IV-ED), with LLM agents served by vLLM or OpenRouter.
"""

from triagesim._version import __version__
from triagesim.cases import Case, load_cases
from triagesim.llm import VLLM, OpenRouter, load_backend
from triagesim.metrics import score, summarize
from triagesim.personas import NursePersona, PatientPersona, load_personas
from triagesim.simulation import simulate

__all__ = [
    "__version__",
    "Case",
    "load_cases",
    "VLLM",
    "OpenRouter",
    "load_backend",
    "NursePersona",
    "PatientPersona",
    "load_personas",
    "simulate",
    "score",
    "summarize",
]
