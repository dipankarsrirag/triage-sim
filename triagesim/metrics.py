"""Scoring of simulated episodes against the case's ground-truth acuity."""

from statistics import mean
from typing import Any, Iterable, Optional


def _grade(pred: Optional[int], gt: Optional[int]) -> dict[str, Any]:
    known = gt is not None and pred is not None
    return {
        "correct": pred == gt if known else None,
        "abs_error": abs(pred - gt) if known else None,
        "over_triage": pred < gt if known else None,
        "under_triage": pred > gt if known else None,
    }


def score(episode: dict) -> dict[str, Any]:
    """Per-episode metrics. Lower triage levels are more urgent, so predicting below the ground
    truth is over-triage and above it under-triage."""
    gt: Optional[int] = episode["case"].get("acuity")
    pred: Optional[int] = episode.get("final_triage")
    checks = [h["verification"] for h in episode["history"] if h.get("verification")]
    return {
        "ground_truth": gt,
        "final_triage": pred,
        **_grade(pred, gt),
        "first_correct_turn": next((t["turn"] for t in episode["trace"] if t["triage"] == gt), None),
        "end_reason": episode["end_reason"],
        "num_turns": episode["num_turns"],
        "num_vitals": sum(h.get("event") == "vital" for h in episode["history"]),
        "num_red_flags": len(episode["red_flags"]),
        "verified_utterances": len(checks),
        "regenerated_utterances": sum(bool(c["rejected"]) for c in checks),
        "flagged_utterances": sum(c["passed"] is False for c in checks),
    }


def summarize(episodes: Iterable[dict]) -> dict[str, Any]:
    """Aggregate metrics over the successful episodes."""
    scores = [score(e) for e in episodes if e["status"] == "ok"]
    graded = [s for s in scores if s["correct"] is not None]
    verified = sum(s["verified_utterances"] for s in scores)

    def rate(key: str, rows: list) -> Optional[float]:
        return mean(bool(r[key]) for r in rows) if rows else None

    def avg(key: str) -> Optional[float]:
        return mean(s[key] for s in scores) if scores else None

    return {
        "episodes": len(scores),
        "graded": len(graded),
        "accuracy": rate("correct", graded),
        "mean_abs_error": mean(s["abs_error"] for s in graded) if graded else None,
        "over_triage_rate": rate("over_triage", graded),
        "under_triage_rate": rate("under_triage", graded),
        "nurse_ended_rate": mean(s["end_reason"] == "nurse_end" for s in scores) if scores else None,
        "mean_turns": avg("num_turns"),
        "mean_vitals": avg("num_vitals"),
        "regenerated_rate": sum(s["regenerated_utterances"] for s in scores) / verified if verified else None,
        "flagged_rate": sum(s["flagged_utterances"] for s in scores) / verified if verified else None,
    }
