import json
from pathlib import Path

import pytest
from conftest import FakeBackend

from triagesim import cli

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def backends(monkeypatch):
    made = {}

    def fake_load_backend(spec, **kwargs):
        made[spec] = FakeBackend(model=spec)
        made[spec].kwargs = kwargs
        return made[spec]

    monkeypatch.setattr(cli, "load_backend", fake_load_backend)
    return made


def argv(out, *extra):
    return [
        "--cases", str(EXAMPLES / "cases.jsonl"),
        "--nurse-personas", str(EXAMPLES / "personas" / "nurse.yaml"),
        "--patient-personas", str(EXAMPLES / "personas" / "patient.yaml"),
        "--out", str(out),
        *extra,
    ]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_run_then_resume(backends, tmp_path, capsys):
    out = tmp_path / "runs" / "run.jsonl"
    assert cli.main(argv(out, "--model", "vllm:/models/x", "--vllm-args", '{"max_model_len": 4096}', "--limit", "2")) == 0
    first = read(out)
    assert [e["episode_id"] for e in first] == [f"{c}-0" for c in ("syncope-01", "chest-pain-01")]
    assert backends["vllm:/models/x"].kwargs == {"max_model_len": 4096}
    assert json.loads(capsys.readouterr().out)["episodes"] == 2

    cli.main(argv(out, "--model", "vllm:/models/x"))  # all cases now; the first two are skipped
    ids = [e["episode_id"] for e in read(out)]
    assert len(ids) == len(set(ids)) == sum(1 for _ in (EXAMPLES / "cases.jsonl").open())


def test_separate_models_per_role(backends, tmp_path):
    out = tmp_path / "run.jsonl"
    cli.main(argv(out, "--nurse-model", "openrouter:a/b", "--patient-model", "vllm:/m", "--limit", "1",
                  "--openrouter-args", '{"max_workers": 4}'))
    [ep] = read(out)
    assert ep["models"] == {"nurse": "openrouter:a/b", "patient": "vllm:/m", "belief": "openrouter:a/b"}
    assert backends["openrouter:a/b"].kwargs == {"max_workers": 4}


def test_failures_go_to_errors_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_backend", lambda spec, **kw: FakeBackend(patient=lambda r: "nope"))
    out = tmp_path / "run.jsonl"
    assert cli.main(argv(out, "--model", "vllm:/m", "--limit", "1")) == 1
    assert out.read_text() == ""
    [err] = read(tmp_path / "run.errors.jsonl")
    assert err["status"] == "error"


def test_model_is_required(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(argv(tmp_path / "run.jsonl"))


def test_arguments_from_file(backends, tmp_path):
    args_file = tmp_path / "run.args"
    args_file.write_text('--model=vllm:/m\n--limit=1\n--sampling={"nurse": {"temperature": 0.2}}\n')
    out = tmp_path / "run.jsonl"
    cli.main(argv(out, f"@{args_file}"))
    assert len(read(out)) == 1
    nurse_calls = [r for b in backends["vllm:/m"].batches for r in b if "vital" in r.output_type.model_fields]
    assert nurse_calls[0].sampling["temperature"] == 0.2


def test_skipped_episodes_are_written_and_not_redone(backends, tmp_path, caplog):
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"case_id": "a", "chief_complaint": "Chest pain", "acuity": 2}\n'
                     '{"case_id": "b", "chief_complaint": "Rash"}\n')
    out = tmp_path / "run.jsonl"
    args = argv(out, "--model", "vllm:/m")
    args[args.index("--cases") + 1] = str(cases)
    cli.main(args)
    assert {e["episode_id"]: e["status"] for e in read(out)} == {"a-0": "ok", "b-0": "skipped"}
    assert "skipped 1: no ground-truth acuity" in caplog.text
    cli.main(args)
    assert len(read(out)) == 2  # nothing redone
