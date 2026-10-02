import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("select_episodes", Path(__file__).parent.parent / "scripts" / "select_episodes.py")
select = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select)


def episode(case_id, acuity, pred, nurse="a", flagged=False, ended=True, status="ok"):
    history = [{"actor": "patient", "verification": {"passed": not flagged}}]
    return {"episode_id": f"{case_id}-0", "status": status, "case": {"case_id": case_id, "acuity": acuity},
            "final_triage": pred, "end_reason": "nurse_end" if ended else "max_turns", "history": history,
            "models": {"nurse": f"vllm:/m/{nurse}", "patient": "vllm:/m/p"}}


def test_keeps_agreeing_episodes_across_runs(tmp_path, capsys):
    run1 = [episode("c1", 2, 2), episode("c2", 3, 2), episode("c3", 4, 4, flagged=True)]
    run2 = [episode("c2", 3, 3, nurse="b"), episode("c4", 5, 5, nurse="b", ended=False), episode("c5", 2, None, status="skipped")]
    paths = []
    for name, eps in (("run1.jsonl", run1), ("run2.jsonl", run2)):
        (tmp_path / name).write_text("".join(json.dumps(e) + "\n" for e in eps))
        paths.append(str(tmp_path / name))

    select.main([*paths, "--out", str(tmp_path / "pool.jsonl"), "--no-flags", "--nurse-ended"])
    pool = [json.loads(line) for line in (tmp_path / "pool.jsonl").read_text().splitlines()]
    assert [(e["case"]["case_id"], e["source"]) for e in pool] == [("c1", "run1.jsonl"), ("c2", "run2.jsonl")]
    report = capsys.readouterr().out
    assert "2 of 5 simulated episodes kept; 2 of 4 cases covered" in report
    assert "acuity 3: 1/1 cases covered" in report and "nurse b / patient p: 1/2 kept" in report

    select.main([*paths, "--out", str(tmp_path / "loose.jsonl")])  # no quality filters
    assert len((tmp_path / "loose.jsonl").read_text().splitlines()) == 4
