from typing import Optional, Literal, List
from pydantic import Field

from triagesim.core.base_schema import BaseAgentOutput


class PatientOutput(BaseAgentOutput):
    utterance: str


class NurseOutput(BaseAgentOutput):
    # ─────────────────────────────────────────
    # Intent-level action (NOT executable)
    # ─────────────────────────────────────────
    action: Literal[
        "utterance",
        "check_vital",
        "log_red_flag",
        "end",
    ] = Field(description="Next action chosen by the nurse")

    # ─────────────────────────────────────────
    # Optional payload
    # ─────────────────────────────────────────
    utterance: Optional[str] = Field(
        description="Nurse utterance or question; null if action is 'end' or 'log_red_flag'"
    )

    triage: int = Field(ge=1, le=5, description="Predicted triage level (ESI or ATS)")

    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in triage decision"
    )

    red_flags: List[str] = Field(
        default_factory=list,
        description="List of all identified red flags so far",
    )

    explanation: str = Field(
        description="Clinical reasoning grounded in the triage algorithm"
    )
