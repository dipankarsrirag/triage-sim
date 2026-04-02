"""
Executable environment actions.

These actions are produced by mapping LLM outputs (NurseOutput)
into schema-validated, environment-safe actions that can be:
- executed by TriageEnv
- serialized to Redis
- replayed and audited
- evaluated and annotated later

All actions are strict Pydantic models.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, ConfigDict


# ─────────────────────────────────────────
# Base Action
# ─────────────────────────────────────────


class BaseAction(BaseModel):
    """
    Base class for all executable actions.
    """

    type: str = Field(..., description="Action type identifier")

    model_config = ConfigDict(
        extra="forbid",  # no accidental keys
        validate_assignment=True,  # safety for mutations
    )


# ─────────────────────────────────────────
# Nurse Actions
# ─────────────────────────────────────────


class NurseUtteranceAction(BaseAction):
    """
    Nurse asks a question or provides reassurance.
    """

    type: Literal["utterance"] = "utterance"
    utterance: str = Field(..., min_length=1)
    triage: int = Field(..., ge=1, le=5)


class NurseCheckVitalAction(BaseAction):
    """
    Nurse requests ONE specific vital sign.
    """

    type: Literal["check_vital"] = "check_vital"
    vital: Literal[
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
    ]


class NurseLogRedFlagAction(BaseAction):
    """
    Nurse explicitly logs red flags already identified.
    """

    type: Literal["log_red_flag"] = "log_red_flag"
    red_flags: List[str] = Field(default_factory=list)


class NurseEndAction(BaseAction):
    """
    Nurse ends the triage interaction.
    """

    type: Literal["end"] = "end"
    triage: int = Field(..., ge=1, le=5)


# ─────────────────────────────────────────
# Union helper (optional but useful)
# ─────────────────────────────────────────

NurseAction = (
    NurseUtteranceAction
    | NurseCheckVitalAction
    | NurseLogRedFlagAction
    | NurseEndAction
)
