"""
Metrics for TriageSim simulations.

Design principles:
- Environment-native: consumes ONLY environment outputs
- No agent internals
- No gold belief state required
- Safe-by-default (metrics degrade gracefully)
- One file, many focused functions
"""

from typing import Dict, Any, List, Optional, Set


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────


def _extract_final_nurse_action(
    trace: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Return the final nurse action dict from the trace, or None.
    """
    for step in reversed(trace):
        if step.get("actor") == "nurse":
            return step
    return None


def _extract_final_triage(trace: List[Dict[str, Any]]) -> Optional[int]:
    """
    Extract final triage level from a NurseEndAction or last nurse utterance.
    """
    for step in reversed(trace):
        action = step.get("action", {})
        if action.get("type") == "end":
            return action.get("triage")
        if "triage" in action:
            return action["triage"]
    return None


# ─────────────────────────────────────────
# Decision / triage metrics
# ─────────────────────────────────────────


def triage_decision_metrics(
    trace: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate final triage decision.

    Returns:
        - correctness
        - absolute error
        - over/under-triage flags
    """
    gt = ground_truth.get("acuity")
    pred = _extract_final_triage(trace)

    if gt is None or pred is None:
        return {
            "ground_truth": gt,
            "final_prediction": pred,
            "correct": None,
            "absolute_error": None,
            "over_triage": None,
            "under_triage": None,
        }

    error = pred - gt

    return {
        "ground_truth": gt,
        "final_prediction": pred,
        "correct": pred == gt,
        "absolute_error": abs(error),
        "over_triage": error < 0,
        "under_triage": error > 0,
    }


def time_to_first_correct_triage(
    trace: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
) -> Optional[int]:
    """
    Turn index when correct triage was first stated.
    """
    gt = ground_truth.get("acuity")
    if gt is None:
        return None

    for step in trace:
        action = step.get("action", {})
        triage = action.get("triage")
        if triage == gt:
            return step.get("turn")

    return None


# ─────────────────────────────────────────
# Belief coverage metrics
# ─────────────────────────────────────────


def belief_coverage_metrics(
    belief: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Structural metrics of what information was elicited.
    """
    return {
        "num_associated_symptoms": len(belief.get("associated_symptoms", [])),
        "num_red_flags_inferred": len(belief.get("red_flags", [])),
        "has_chief_complaint": bool(belief.get("chief_complaint")),
        "has_pain_location": bool(belief.get("pain_location")),
        "has_pain_severity": bool(belief.get("pain_severity")),
        "has_duration": bool(belief.get("duration")),
        "vitals_known": belief.get("vitals_known", []),
        "num_vitals_known": len(belief.get("vitals_known", [])),
    }


# ─────────────────────────────────────────
# Red flag safety metrics
# ─────────────────────────────────────────


def red_flag_metrics(
    belief: Dict[str, Any],
    expert_red_flags: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Metrics for explicitly logged red flags.

    Works even without expert annotations.
    """
    logged = set(belief.get("red_flags_logged", []))

    metrics: Dict[str, Any] = {
        "num_red_flags_logged": len(logged),
        "logged_any": bool(logged),
        "logged_flags": sorted(logged),
    }

    if expert_red_flags is None:
        return metrics

    tp = logged & expert_red_flags
    fp = logged - expert_red_flags
    fn = expert_red_flags - logged

    metrics.update(
        {
            "expert_red_flags": sorted(expert_red_flags),
            "true_positives": sorted(tp),
            "false_positives": sorted(fp),
            "false_negatives": sorted(fn),
            "precision": len(tp) / len(logged) if logged else None,
            "recall": len(tp) / len(expert_red_flags) if expert_red_flags else None,
        }
    )

    return metrics


# ─────────────────────────────────────────
# Explanation metrics (hooks only)
# ─────────────────────────────────────────


def explanation_support_metrics(
    trace: List[Dict[str, Any]],
    belief: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Placeholder for explanation faithfulness metrics.

    Future ideas:
    - Citation overlap
    - Evidence hallucination checks
    """
    final = _extract_final_nurse_action(trace)
    explanation = None

    if final:
        explanation = final.get("action", {}).get("explanation")

    return {
        "has_explanation": explanation is not None,
        "explanation_length": len(explanation.split()) if explanation else 0,
    }


# ─────────────────────────────────────────
# Master orchestrator
# ─────────────────────────────────────────


def compute_all_metrics(
    *,
    trace: List[Dict[str, Any]],
    belief: Dict[str, Any],
    ground_truth: Dict[str, Any],
    expert_red_flags: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Compute all metrics for a single simulation run.
    """

    metrics: Dict[str, Any] = {}

    metrics["triage"] = triage_decision_metrics(trace, ground_truth)
    metrics["triage"]["first_correct_turn"] = time_to_first_correct_triage(
        trace, ground_truth
    )

    metrics["belief_coverage"] = belief_coverage_metrics(belief)

    metrics["red_flags"] = red_flag_metrics(belief, expert_red_flags)

    metrics["explanation"] = explanation_support_metrics(trace, belief)

    return metrics
