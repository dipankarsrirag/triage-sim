from triagesim.metrics import score, summarize


def episode(pred, gt=2, status="ok", trace_triage=(3, 2)):
    return {
        "status": status,
        "case": {"acuity": gt},
        "final_triage": pred,
        "end_reason": "nurse_end",
        "num_turns": 2,
        "history": [{"actor": "system", "event": "vital"}, {"actor": "nurse"}],
        "trace": [{"turn": i, "triage": t} for i, t in enumerate(trace_triage)],
        "red_flags": ["x"],
    }


def test_score_directions():
    assert score(episode(1))["over_triage"] and not score(episode(1))["under_triage"]
    assert score(episode(3))["under_triage"]
    s = score(episode(2))
    assert (s["correct"], s["abs_error"], s["first_correct_turn"], s["num_vitals"]) == (True, 0, 1, 1)


def test_score_without_ground_truth():
    s = score(episode(2, gt=None))
    assert s["correct"] is None and s["first_correct_turn"] is None


def test_summarize_skips_failed_episodes():
    summary = summarize([episode(2), episode(4), episode(1, status="error")])
    assert (summary["episodes"], summary["accuracy"], summary["mean_abs_error"]) == (2, 0.5, 1)
    assert summary["under_triage_rate"] == 0.5 and summary["over_triage_rate"] == 0


def test_summarize_empty():
    assert summarize([])["accuracy"] is None


def test_verification_counts():
    ep = episode(2)
    ep["history"] = [
        {"actor": "nurse", "verification": {"passed": True, "rejected": []}},
        {"actor": "patient", "verification": {"passed": True, "rejected": [{"utterance": "x"}]}},
        {"actor": "patient", "verification": {"passed": False, "rejected": [{"utterance": "y"}]}},
    ]
    s = score(ep)
    assert (s["verified_utterances"], s["regenerated_utterances"], s["flagged_utterances"]) == (3, 2, 1)
    assert abs(summarize([ep])["flagged_rate"] - 1 / 3) < 1e-9
