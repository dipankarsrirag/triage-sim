"""
Turn run argument files that use in-process vLLM models into one `vllm serve` per model (each on
its own GPU) plus, per run, triagesim overrides that point every role at its server. Used by
jobs/sweep.pbs with SERVE=1, for GPUs too small to hold both models of a pair (e.g. 2x L40S). Runs
given together share the servers of the models they have in common (e.g. a pair and its mirror).

    python scripts/serve_args.py a.args [b.args ...] --port 21000 --plan plan.txt --overrides a.serve [b.serve ...]
    python -m triagesim @a.args @a.serve --out a.jsonl

plan.txt has one line per model, in role order (nurse first): "<port> <model> <vllm serve flags>".
Each model's chat_template_kwargs (e.g. enable_thinking) move into its requests' extra_body.
"""

import argparse
import json
import shlex
from pathlib import Path

from triagesim.cli import _parser


def serve_flags(engine_args: dict) -> list[str]:
    flags = []
    for key, value in engine_args.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            flags.append(flag)
        elif value not in (False, None):
            flags += [flag, json.dumps(value) if isinstance(value, (dict, list)) else str(value)]
    return flags


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("args_files", nargs="+")
    p.add_argument("--port", type=int, default=21000, help="port of the first server; the next ones count up")
    p.add_argument("--plan", required=True)
    p.add_argument("--overrides", nargs="+", required=True, help="one output file per args file")
    args = p.parse_args(argv)
    if len(args.overrides) != len(args.args_files):
        p.error("give one --overrides file per args file")

    runs = [_parser().parse_args([f"@{f}", "--out", "unused"]) for f in args.args_files]
    roles = [{"nurse": r.nurse_model or r.model, "patient": r.patient_model or r.model, "judge": r.judge_model}
             for r in runs]
    plan, model_args, served, engines = [], {}, {}, {}
    for run, run_roles in zip(runs, roles):
        for spec in dict.fromkeys(s for s in run_roles.values() if s):
            kind, _, model = spec.partition(":")
            if kind != "vllm":
                raise SystemExit(f"{spec}: serving needs vllm:<model> specs")
            engine = {**run.vllm_args, **run.model_args.get(spec, {})}
            if spec in engines:
                if engine != engines[spec]:
                    raise SystemExit(f"{spec}: the args files give it different engine args")
                continue
            engines[spec] = dict(engine)
            template = engine.pop("chat_template_kwargs", None)
            port, name = args.port + len(plan), Path(model).name
            plan.append(shlex.join([str(port), model, "--served-model-name", name, "--port", str(port), *serve_flags(engine)]))
            served[spec] = f"openrouter:{name}"
            model_args[served[spec]] = {
                "base_url": f"http://127.0.0.1:{port}/v1",
                "api_key": "EMPTY",
                "max_workers": 256,
                "extra_body": {"chat_template_kwargs": template} if template else {},
            }
    Path(args.plan).write_text("\n".join(plan) + "\n")
    for run_roles, out in zip(roles, args.overrides):
        overrides = [f"--{role}-model={served[spec]}" for role, spec in run_roles.items() if spec]
        mine = {served[spec]: model_args[served[spec]] for spec in dict.fromkeys(run_roles.values()) if spec}
        Path(out).write_text("\n".join(overrides + [f"--model-args={json.dumps(mine)}"]) + "\n")


if __name__ == "__main__":
    main()
