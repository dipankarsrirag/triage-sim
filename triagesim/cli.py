"""
Command line entry point: simulate episodes for a case file, appending them to a JSONL file.

Re-running with the same --out resumes: episodes already in the file are skipped. Failed episodes
go to <out>.errors.jsonl (with the dialogue up to the failure) and are retried on the next run.
SIGINT or SIGTERM stops the run at once (exit 128 + signal): finished episodes are kept, and the
ones in flight run again on the next run.
"""

import argparse
import json
import logging
import os
import signal
import time
from collections import Counter
from pathlib import Path

from triagesim.cases import load_cases
from triagesim.llm import load_backend
from triagesim.metrics import summarize
from triagesim.personas import NursePersona, PatientPersona, load_personas
from triagesim.simulation import Features, simulate

log = logging.getLogger("triagesim")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="triagesim",
        description="Simulate nurse-patient ED triage dialogues with vLLM or OpenRouter models.",
        epilog="Arguments can also be read from a file, one per line: triagesim @run.args",
        fromfile_prefix_chars="@",
    )
    p.add_argument("--cases", required=True, help="cases file (.jsonl, .json or .csv, e.g. MIMIC-IV-ED triage.csv)")
    p.add_argument("--nurse-personas", required=True, help="YAML list of nurse personas")
    p.add_argument("--patient-personas", required=True, help="YAML list of patient personas")
    p.add_argument("--out", required=True, help="output JSONL (appended to; existing episodes are skipped)")

    p.add_argument("--model", help="model for every role: vllm:<path or HF id> or openrouter:<model id>")
    p.add_argument("--nurse-model", help="nurse model (default: --model)")
    p.add_argument("--patient-model", help="patient model (default: --model)")
    p.add_argument("--judge-model", help="dialogue master: checks each case would have a triage conversation, "
                   "writes the patient's script, verifies every spoken line and keeps the nurse's triage "
                   "record (default: none)")
    p.add_argument("--max-regenerations", type=int, default=1,
                   help="regenerations of a rejected line before it is kept and flagged")
    p.add_argument("--vllm-args", type=json.loads, default={},
                   help='JSON kwargs for every vLLM engine, e.g. \'{"max_model_len": 8192}\'; several vLLM '
                        'models split gpu_memory_utilization evenly unless it is given')
    p.add_argument("--openrouter-args", type=json.loads, default={},
                   help='JSON kwargs for OpenRouter backends, e.g. \'{"max_workers": 32}\'')
    p.add_argument("--model-args", type=json.loads, default={},
                   help='JSON kwargs for single models, keyed by model spec and merged over --vllm-args or '
                        '--openrouter-args, e.g. \'{"vllm:/models/qwen": {"gpu_memory_utilization": 0.6, '
                        '"chat_template_kwargs": {"enable_thinking": false}}}\'')
    p.add_argument("--sampling", type=json.loads, default=None,
                   help='JSON per-role sampling overrides, e.g. \'{"nurse": {"temperature": 0.7}}\'')

    p.add_argument("--algorithm", choices=["esi", "ats"], help="triage scale for every case (default: the case's acuity_scale, else esi)")
    p.add_argument("--max-turns", type=int, default=12, help="maximum nurse questions per episode")
    p.add_argument("--min-questions", type=int, default=3, help="questions the nurse must ask before it may end")
    p.add_argument("--episodes-per-case", type=int, default=1)
    p.add_argument("--limit", type=int, help="only use the first N cases")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--appearance", action="store_true",
                   help="the dialogue master describes how the patient looks on arrival, for the nurse")
    p.add_argument("--belief-state", action="store_true",
                   help="show the nurse its last level, confidence, red flags and reasoning at each step")
    p.add_argument("--feedback", action="store_true",
                   help="the dialogue master tells the nurse how warm or cold its last level is (uses the ground truth)")
    p.add_argument("--concurrency", type=int, default=256, help="episodes in flight at once")
    return p


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.setLevel(logging.INFO)  # progress from triagesim, not per-request lines from httpx

    nurse_spec = args.nurse_model or args.model
    patient_spec = args.patient_model or args.model
    if not nurse_spec or not patient_spec:
        parser.error("give --model, or both --nurse-model and --patient-model")
    specs = {nurse_spec, patient_spec, args.judge_model} - {None}
    n_vllm = sum(spec.startswith("vllm:") for spec in specs)

    backends = {}

    def backend(spec: str | None):
        if spec is None:
            return None
        if spec not in backends:
            kwargs = {**(args.vllm_args if spec.startswith("vllm:") else args.openrouter_args),
                      **args.model_args.get(spec, {})}
            if spec.startswith("vllm:") and n_vllm > 1:
                kwargs.setdefault("gpu_memory_utilization", round(0.9 / n_vllm, 2))
            backends[spec] = load_backend(spec, **kwargs)
        return backends[spec]

    cases = load_cases(args.cases)[: args.limit]
    out = Path(args.out)
    errors = out.with_suffix(".errors.jsonl")
    if out.exists() and (data := out.read_bytes()) and not data.endswith(b"\n"):
        out.write_bytes(data[: data.rfind(b"\n") + 1])  # a run killed mid-write left a partial last line
    done = {e["episode_id"] for e in _read_jsonl(out)}
    todo = sum(f"{c.case_id}-{k}" not in done for c in cases for k in range(args.episodes_per_case))
    log.info("%d cases, %d episodes to run (%d already in %s)", len(cases), todo, len(done), out)

    episodes = simulate(
        cases,
        nurse_llm=backend(nurse_spec),
        patient_llm=backend(patient_spec),
        judge_llm=backend(args.judge_model),
        max_regenerations=args.max_regenerations,
        nurse_personas=load_personas(args.nurse_personas, NursePersona),
        patient_personas=load_personas(args.patient_personas, PatientPersona),
        algorithm=args.algorithm,
        max_turns=args.max_turns,
        min_questions=args.min_questions,
        episodes_per_case=args.episodes_per_case,
        seed=args.seed,
        concurrency=args.concurrency,
        sampling=args.sampling,
        skip=done,
        features=Features(appearance=args.appearance, belief_state=args.belief_state, feedback=args.feedback),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_failed = 0
    skipped: Counter = Counter()
    writing, stop_signal = False, None

    def stop(signum=None, frame=None):
        """SIGINT/SIGTERM (the sweep's timeout, qdel, walltime): exit at once instead of waiting for the
        model calls in flight. Finished episodes are on disk; unfinished ones run again on the next run.
        An episode being written is finished first."""
        nonlocal stop_signal
        stop_signal = signum or stop_signal
        if not writing:
            log.info("stopped by %s after %d episodes; episodes in flight run again on the next run",
                     signal.Signals(stop_signal).name, n_ok + n_failed)
            logging.shutdown()
            os._exit(128 + stop_signal)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    start = last_log = time.time()
    try:
        with out.open("a") as f:
            for ep in episodes:
                writing = True
                if ep["status"] in ("ok", "skipped"):  # skipped ones too, so a rerun does not redo them
                    f.write(json.dumps(ep) + "\n")
                    f.flush()
                    if ep["status"] == "ok":
                        n_ok += 1
                    else:
                        skipped[ep["skip_reason"]] += 1
                else:
                    with errors.open("a") as f_err:
                        f_err.write(json.dumps(ep) + "\n")
                    n_failed += 1
                    log.warning("episode %s failed: %s", ep["episode_id"], ep["error"])
                writing = False
                if stop_signal:
                    stop()
                if time.time() - last_log > 30:
                    last_log = time.time()
                    rate = (n_ok + n_failed) / (last_log - start)
                    log.info("%d/%d episodes done (%d failed), %.2f episodes/s", n_ok + n_failed, todo, n_failed, rate)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    log.info("finished %d episodes (%d failed, %d skipped) in %.0fs",
             n_ok + n_failed, n_failed, sum(skipped.values()), time.time() - start)
    for reason, n in skipped.most_common(10):
        log.info("skipped %d: %s", n, reason)
    for spec, b in backends.items():
        log.info("usage %s: %s", spec, dict(b.usage))
    print(json.dumps(summarize(_read_jsonl(out)), indent=2))
    return 1 if n_failed and not n_ok else 0
