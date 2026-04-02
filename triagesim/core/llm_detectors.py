"""
LLM-backed detectors for extracting structured clinical evidence
from dialogue turns.

This module:
- Uses strict Pydantic schemas
- Never mutates environment state
- Never hard-codes clinical rules
- Uses a caller-provided LLM (persona-aware)
- Fails deterministically and safely
"""

from typing import Type, Union
from pydantic import BaseModel, ValidationError

from triagesim.agents.base_agent import BaseLLM
from triagesim.core.llm_schemas import (
    LLMPainDetection,
    LLMSymptomDetection,
    LLMRedFlagDetection,
    LLMDurationDetection,
    LLMChiefComplaint,
    LLMVitalsDetection,
)


class LLMEvidenceDetector:
    """
    Thin wrapper around an LLM for structured evidence extraction.

    IMPORTANT:
    - This detector NEVER creates its own LLM
    - The caller (e.g. NurseAgent) owns the LLM + persona
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_retries: int = 2,
    ):
        self.llm = llm
        self.max_retries = max_retries

    # ─────────────────────────────────────────
    # Core inference helper
    # ─────────────────────────────────────────

    def _infer(
        self,
        prompt: str,
        schema: Type[BaseModel],
    ) -> Union[BaseModel, None]:
        """
        Run the LLM with strict schema validation.

        Safety:
        - Retries on validation or runtime errors
        - Returns None on failure (never raises)
        """
        last_err: Exception | None = None

        for _ in range(self.max_retries + 1):
            try:
                # Temporarily override output type
                result = self.llm.generate(
                    prompt=prompt,
                    output_type=schema,
                )
                return result
            except (ValidationError, Exception) as e:
                last_err = e
                continue

        return None

    # ─────────────────────────────────────────
    # Detectors
    # ─────────────────────────────────────────

    def detect_pain(self, text: str) -> LLMPainDetection:
        prompt = f"""
Extract pain information from this utterance:

\"\"\"{text}\"\"\"

Return ONLY JSON matching LLMPainDetection:
- pain_severity: mild | moderate | severe | null
- pain_location: string | null
"""
        out = self._infer(prompt, LLMPainDetection)
        return out or LLMPainDetection()

    def detect_symptoms(self, text: str) -> LLMSymptomDetection:
        prompt = f"""
Extract associated symptoms from this utterance:

\"\"\"{text}\"\"\"

Return ONLY JSON matching LLMSymptomDetection:
- symptoms: list[str]
"""
        out = self._infer(prompt, LLMSymptomDetection)
        return out or LLMSymptomDetection()

    def detect_red_flags(self, text: str) -> LLMRedFlagDetection:
        prompt = f"""
Extract clinical red flags explicitly or implicitly suggested
by this utterance.

\"\"\"{text}\"\"\"

Return ONLY JSON matching LLMRedFlagDetection:
- red_flags: list[str]
"""
        out = self._infer(prompt, LLMRedFlagDetection)
        return out or LLMRedFlagDetection()

    def detect_duration(self, text: str) -> LLMDurationDetection:
        prompt = f"""
Extract temporal duration information from this utterance.

\"\"\"{text}\"\"\"

Return ONLY JSON matching LLMDurationDetection:
- duration: string | null
"""
        out = self._infer(prompt, LLMDurationDetection)
        return out or LLMDurationDetection()

    def detect_chief_complaint(self, text: str) -> LLMChiefComplaint:
        prompt = f"""
Extract the chief complaint phrase from this utterance.

\"\"\"{text}\"\"\"

Return ONLY JSON matching LLMChiefComplaint:
- chief_complaint: string | null
"""
        out = self._infer(prompt, LLMChiefComplaint)
        return out or LLMChiefComplaint()

    def detect_vitals(self, text: str) -> LLMVitalsDetection:
        prompt = f"""
You extract which vital sign a triage nurse is asking for.

Nurse utterance:
\"\"\"{text}\"\"\"

Return ONLY JSON matching LLMVitalsDetection:
- vitals: list[str]
  Allowed values:
  ["temperature", "heartrate", "resprate", "o2sat", "sbp"]
"""
        out = self._infer(prompt, LLMVitalsDetection)
        return out or LLMVitalsDetection(vitals=[])
