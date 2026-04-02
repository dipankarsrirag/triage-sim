"""
triagesim

A multi-speaker spoken dialogue simulation framework for emergency
department triage. Generates paired structured EHR → dialogue →
accented speech data for speech and language research.

Pipeline:
  1. Structured data (MIMIC-IV-ED / ATS / ETEK) → simulation
  2. Nurse ↔ patient dialogue via LLM agents
  3. Phrase-break annotation for TTS
  4. Speech synthesis via Qwen3-TTS (see scripts/)
"""

from triagesim._version import __version__
from triagesim.runner import TriageRunner, RunnerConfig

__all__ = [
    "__version__",
    "TriageRunner",
    "RunnerConfig",
]
