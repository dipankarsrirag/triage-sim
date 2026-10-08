import importlib.util
import json
from pathlib import Path

import pytest

from triagesim.cli import _parser

spec = importlib.util.spec_from_file_location("serve_args", Path(__file__).parent.parent / "scripts" / "serve_args.py")
serve_args = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve_args)


def test_one_server_per_model_and_roles_point_at_them(tmp_path):
    run = tmp_path / "run.args"
    run.write_text("\n".join([
        "--cases=c.jsonl", "--nurse-personas=n.jsonl", "--patient-personas=p.jsonl",
        "--nurse-model=vllm:/m/A", "--patient-model=vllm:/m/B", "--judge-model=vllm:/m/A",
        '--vllm-args={"max_model_len": 4096, "language_model_only": true}',
        '--model-args={"vllm:/m/B": {"language_model_only": false, "chat_template_kwargs": {"enable_thinking": false}}}',
    ]) + "\n")
    plan, overrides = tmp_path / "plan.txt", tmp_path / "serve.args"
    serve_args.main([str(run), "--port", "9000", "--plan", str(plan), "--overrides", str(overrides)])

    assert plan.read_text().splitlines() == [
        "9000 /m/A --served-model-name A --port 9000 --max-model-len 4096 --language-model-only",
        "9001 /m/B --served-model-name B --port 9001 --max-model-len 4096",
    ]
    merged = _parser().parse_args([f"@{run}", f"@{overrides}", "--out", "x"])
    assert (merged.nurse_model, merged.patient_model, merged.judge_model) == ("openrouter:A", "openrouter:B", "openrouter:A")
    assert merged.model_args["openrouter:A"]["base_url"] == "http://127.0.0.1:9000/v1"
    assert merged.model_args["openrouter:A"]["extra_body"] == {}
    assert merged.model_args["openrouter:B"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    json.loads(overrides.read_text().splitlines()[-1].split("=", 1)[1])


def test_mirror_pairs_share_one_server_per_model(tmp_path):
    common = ["--cases=c.jsonl", "--nurse-personas=n.jsonl", "--patient-personas=p.jsonl",
              '--vllm-args={"max_model_len": 4096}']
    ab, ba = tmp_path / "ab.args", tmp_path / "ba.args"
    ab.write_text("\n".join(common + ["--nurse-model=vllm:/m/A", "--patient-model=vllm:/m/B", "--judge-model=vllm:/m/A"]))
    ba.write_text("\n".join(common + ["--nurse-model=vllm:/m/B", "--patient-model=vllm:/m/A", "--judge-model=vllm:/m/B"]))
    plan, o_ab, o_ba = tmp_path / "plan.txt", tmp_path / "ab.serve", tmp_path / "ba.serve"
    serve_args.main([str(ab), str(ba), "--port", "9000", "--plan", str(plan), "--overrides", str(o_ab), str(o_ba)])

    assert [line.split()[:2] for line in plan.read_text().splitlines()] == [["9000", "/m/A"], ["9001", "/m/B"]]
    for run, overrides, nurse, patient in ((ab, o_ab, "A", "B"), (ba, o_ba, "B", "A")):
        merged = _parser().parse_args([f"@{run}", f"@{overrides}", "--out", "x"])
        assert (merged.nurse_model, merged.patient_model, merged.judge_model) == (
            f"openrouter:{nurse}", f"openrouter:{patient}", f"openrouter:{nurse}")
        assert merged.model_args["openrouter:A"]["base_url"] == "http://127.0.0.1:9000/v1"
        assert merged.model_args["openrouter:B"]["base_url"] == "http://127.0.0.1:9001/v1"


def test_runs_that_disagree_on_a_model_engine_are_rejected(tmp_path):
    common = ["--cases=c.jsonl", "--nurse-personas=n.jsonl", "--patient-personas=p.jsonl", "--model=vllm:/m/A"]
    a, b = tmp_path / "a.args", tmp_path / "b.args"
    a.write_text("\n".join(common + ['--vllm-args={"max_model_len": 4096}']))
    b.write_text("\n".join(common + ['--vllm-args={"max_model_len": 8192}']))
    with pytest.raises(SystemExit, match="different engine args"):
        serve_args.main([str(a), str(b), "--plan", str(tmp_path / "p"), "--overrides", str(tmp_path / "o1"), str(tmp_path / "o2")])
