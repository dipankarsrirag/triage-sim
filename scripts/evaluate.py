"""
Triage evaluation on the seed datasets with one engine load per model.

Runs every dataset in --data-dir/cases (<name>.jsonl), each case with its own triage scale
(acuity_scale: ETEK is ATS, the rest ESI), writes <out-dir>/<name>.jsonl and prints a summary per
dataset. With --judge-model the dialogue master screens cases, writes scripts and checks lines.

    python scripts/evaluate.py --model vllm:/path/to/model --out-dir outputs/eval/gemma4
"""

import argparse
import json
from pathlib import Path

from triagesim import NursePersona, PatientPersona, load_backend, load_cases, load_personas, simulate, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0], fromfile_prefix_chars="@")
    p.add_argument("--model", help="model for both roles: vllm:<path> or openrouter:<id>")
    p.add_argument("--nurse-model")
    p.add_argument("--patient-model")
    p.add_argument("--judge-model", help="dialogue master (default: none)")
    p.add_argument("--vllm-args", type=json.loads, default={"max_model_len": 16384, "language_model_only": True})
    p.add_argument("--data-dir", default="data")
    p.add_argument("--datasets", nargs="+", default=["esi", "etek", "mimic"])
    p.add_argument("--limit", type=int, help="first N cases per dataset")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    specs = {"nurse": args.nurse_model or args.model, "patient": args.patient_model or args.model, "judge": args.judge_model}
    n_vllm = len({s for s in specs.values() if s and s.startswith("vllm:")})
    backends = {None: None}
    for spec in set(specs.values()) - {None}:
        kwargs = dict(args.vllm_args) if spec.startswith("vllm:") else {}
        if spec.startswith("vllm:") and n_vllm > 1:
            kwargs.setdefault("gpu_memory_utilization", round(0.9 / n_vllm, 2))
        backends[spec] = load_backend(spec, **kwargs)

    data = Path(args.data_dir)
    nurses = load_personas(data / "personas" / "nurse.jsonl", NursePersona)  # scripts/generate_personas.py
    patients = load_personas(data / "personas" / "patient.jsonl", PatientPersona)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.datasets:
        cases = load_cases(data / "cases" / f"{name}.jsonl")[: args.limit]
        episodes = list(
            simulate(
                cases,
                nurse_llm=backends[specs["nurse"]],
                patient_llm=backends[specs["patient"]],
                judge_llm=backends[specs["judge"]],
                nurse_personas=nurses,
                patient_personas=patients,
                extract_beliefs=False,
            )
        )
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")
        status = {s: sum(ep["status"] == s for ep in episodes) for s in ("ok", "skipped", "error")}
        print(f"== {name}: {len(episodes)} episodes {status}")
        print(json.dumps(summarize(episodes), indent=2), flush=True)


if __name__ == "__main__":
    main()
