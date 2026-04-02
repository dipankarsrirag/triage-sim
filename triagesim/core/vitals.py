"""
Rule + LLM fall-back logic for inferring requested vitals.

Like belief.py, this provides:
1. Deterministic extraction of vitals from nurse text
2. LLM inference if no deterministic cues are found
"""

from typing import List, Set

from triagesim.core.llm_detectors import LLMEvidenceDetector

# Allowed vitals set (same as schema)
ALLOWED_VITALS: Set[str] = {"temperature", "heartrate", "resprate", "o2sat", "sbp"}

# Simple keyword mapping for deterministic detection
KEYWORD_TO_VITAL = {
    "temperature": "temperature",
    "temp": "temperature",
    "fever": "temperature",
    "heart rate": "heartrate",
    "pulse": "heartrate",
    "blood pressure": "sbp",
    "bp": "sbp",
    "respiratory rate": "resprate",
    "resp rate": "resprate",
    "breathing rate": "resprate",
    "oxygen saturation": "o2sat",
    "o2 sat": "o2sat",
    "o2": "o2sat",
}


def rule_extract_vitals(nurse_text: str) -> List[str]:
    """
    First pass: look for simple keyword matches in the nurse utterance.
    Returns a list of valid vitals requested.
    """
    text_lower = nurse_text.lower()
    hits = set()

    for phrase, key in KEYWORD_TO_VITAL.items():
        if phrase in text_lower and key in ALLOWED_VITALS:
            hits.add(key)

    # Only return ORDERED list of unique
    return sorted(list(hits))


def infer_vitals_with_llm(nurse_text: str, detector: LLMEvidenceDetector) -> List[str]:
    """
    Ask the LLM to infer the vitals if rule extraction found nothing.
    """
    llm_out = detector.detect_vitals(nurse_text)
    return llm_out.vitals if llm_out and isinstance(llm_out.vitals, list) else []


def extract_requested_vitals(
    nurse_text: str,
    detector: LLMEvidenceDetector = None,
) -> List[str]:
    """
    Main entrypoint: infer requested vitals from nurse text.
    Combines rule and LLM detection.
    """

    # 1) Try rule-based
    rule_hits = rule_extract_vitals(nurse_text)
    if rule_hits:
        return rule_hits

    # 2) If rule fails and we have a detector, use it
    if detector:
        llm_hits = infer_vitals_with_llm(nurse_text, detector)
        # Filter out anything not allowed
        return [v for v in llm_hits if v in ALLOWED_VITALS]

    # 3) Default fallback
    return []
