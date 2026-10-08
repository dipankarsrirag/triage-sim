import json
import signal
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


def test_resume_drops_a_partial_last_line(backends, tmp_path):
    out = tmp_path / "run.jsonl"
    cli.main(argv(out, "--model", "vllm:/m", "--limit", "1"))
    with out.open("a") as f:
        f.write('{"episode_id": "chest-pain-01-0", "sta')  # killed mid-write
    cli.main(argv(out, "--model", "vllm:/m", "--limit", "2"))
    assert [e["episode_id"] for e in read(out)] == ["syncope-01-0", "chest-pain-01-0"]


def test_separate_models_per_role(backends, tmp_path):
    out = tmp_path / "run.jsonl"
    cli.main(argv(out, "--nurse-model", "openrouter:a/b", "--patient-model", "vllm:/m", "--limit", "1",
                  "--openrouter-args", '{"max_workers": 4}'))
    [ep] = read(out)
    assert ep["models"] == {"nurse": "openrouter:a/b", "patient": "vllm:/m"}
    assert backends["openrouter:a/b"].kwargs == {"max_workers": 4}


def test_model_args_override_shared_args(backends, tmp_path):
    out = tmp_path / "run.jsonl"
    cli.main(argv(out, "--nurse-model", "vllm:/big", "--patient-model", "vllm:/small", "--limit", "1",
                  "--vllm-args", '{"max_model_len": 4096}',
                  "--model-args", '{"vllm:/big": {"gpu_memory_utilization": 0.7, "max_model_len": 8192}}'))
    assert backends["vllm:/big"].kwargs == {"max_model_len": 8192, "gpu_memory_utilization": 0.7}
    assert backends["vllm:/small"].kwargs == {"max_model_len": 4096, "gpu_memory_utilization": 0.45}


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


class Stopped(BaseException):
    """Stands in for os._exit in tests."""


@pytest.fixture
def no_exit(monkeypatch):
    def fake_exit(code):
        raise Stopped(code)

    monkeypatch.setattr(cli.os, "_exit", fake_exit)


def test_signal_stops_at_once_keeping_finished_episodes(monkeypatch, tmp_path, no_exit):
    class SignalledBackend(FakeBackend):
        def complete(self, requests):
            if len(self.batches) == 40:  # mid-run, while waiting for the model
                signal.raise_signal(signal.SIGTERM)
            return super().complete(requests)

    monkeypatch.setattr(cli, "load_backend", lambda spec, **kw: SignalledBackend())
    out = tmp_path / "run.jsonl"
    with pytest.raises(Stopped) as stopped:
        cli.main(argv(out, "--model", "vllm:/m", "--concurrency", "1"))
    assert stopped.value.args == (128 + signal.SIGTERM,)
    kept = [e["episode_id"] for e in read(out)]
    total = sum(1 for _ in (EXAMPLES / "cases.jsonl").open())
    assert 0 < len(kept) < total
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL  # the CLI's handler is gone again

    monkeypatch.setattr(cli, "load_backend", lambda spec, **kw: FakeBackend())
    assert cli.main(argv(out, "--model", "vllm:/m")) == 0  # the next run finishes the rest
    ids = [e["episode_id"] for e in read(out)]
    assert ids[: len(kept)] == kept and len(ids) == len(set(ids)) == total


def test_signal_while_writing_finishes_the_episode(backends, monkeypatch, tmp_path, no_exit):
    dumps = json.dumps

    def signalling_dumps(obj, *a, **kw):
        if isinstance(obj, dict) and "episode_id" in obj:
            signal.raise_signal(signal.SIGINT)
        return dumps(obj, *a, **kw)

    monkeypatch.setattr(cli.json, "dumps", signalling_dumps)
    out = tmp_path / "run.jsonl"
    with pytest.raises(Stopped) as stopped:
        cli.main(argv(out, "--model", "vllm:/m", "--limit", "2", "--concurrency", "1"))
    assert stopped.value.args == (128 + signal.SIGINT,)
    monkeypatch.setattr(cli.json, "dumps", dumps)
    assert [e["episode_id"] for e in read(out)] == ["syncope-01-0"]  # written in full, then stopped
