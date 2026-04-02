"""
Maps LLM-facing NurseOutput into executable environment actions.

This layer is CRITICAL for:
- safety
- auditability
- schema enforcement
- preventing LLM overreach

LLMs speak in NurseOutput.
The environment executes *actions*.
"""

from triagesim.core.output_schema import NurseOutput
from triagesim.core.actions import (
    NurseUtteranceAction,
    NurseCheckVitalAction,
    NurseLogRedFlagAction,
    NurseEndAction,
)

from typing import Optional


class ActionMappingError(RuntimeError):
    """Raised when NurseOutput cannot be safely mapped to an action."""


def map_nurse_output_to_action(output: NurseOutput):
    """
    Convert a NurseOutput into exactly ONE executable nurse action.

    Rules:
    - Exactly one action per turn
    - Action must be consistent with provided fields
    - Invalid combinations raise ActionMappingError
    """

    action = output.action
    utterance = output.utterance
    triage = output.triage
    red_flags = output.red_flags or []

    # ─────────────────────────────────────────
    # UTTERANCE
    # ─────────────────────────────────────────
    if action == "utterance":
        if not utterance:
            raise ActionMappingError(
                "Utterance action requires non-null utterance text."
            )

        return NurseUtteranceAction(
            utterance=utterance,
            triage=triage,
        )

    # ─────────────────────────────────────────
    # CHECK VITAL
    # ─────────────────────────────────────────
    if action == "check_vital":
        if not utterance:
            raise ActionMappingError(
                "check_vital action requires utterance specifying the vital."
            )

        # NOTE:
        # We do NOT infer which vital here.
        # The environment (via llm_detectors / vitals logic)
        # decides which vital to release.
        return NurseCheckVitalAction(
            vital=_extract_single_vital(utterance),
        )

    # ─────────────────────────────────────────
    # LOG RED FLAG
    # ─────────────────────────────────────────
    if action == "log_red_flag":
        if not red_flags:
            raise ActionMappingError(
                "log_red_flag action requires at least one red flag."
            )

        return NurseLogRedFlagAction(
            red_flags=red_flags,
        )

    # ─────────────────────────────────────────
    # END
    # ─────────────────────────────────────────
    if action == "end":
        return NurseEndAction(
            triage=triage,
        )

    # ─────────────────────────────────────────
    # UNKNOWN
    # ─────────────────────────────────────────
    raise ActionMappingError(f"Unsupported nurse action: {action}")


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

ALLOWED_VITALS = {
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
}


def _extract_single_vital(utterance: str) -> Optional[str]:
    """
    Extract exactly one allowed vital from a nurse utterance.

    If multiple vitals are mentioned, the FIRST match is used.
    If none are found, returns None.
    """

    text = utterance.lower()

    keyword_map = {
        "temperature": "temperature",
        "fever": "temperature",
        "heart rate": "heartrate",
        "pulse": "heartrate",
        "blood pressure": "sbp",
        "bp": "sbp",
        "respiratory rate": "resprate",
        "breathing": "resprate",
        "oxygen": "o2sat",
        "saturation": "o2sat",
        "o2": "o2sat",
    }

    for phrase, vital in keyword_map.items():
        if phrase in text and vital in ALLOWED_VITALS:
            return vital

    return None
