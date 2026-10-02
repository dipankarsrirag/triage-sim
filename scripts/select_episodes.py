"""
Pool episodes from many simulation runs (model pairings, personas, seeds) and keep those whose nurse
reached the ground-truth triage level, optionally with quality filters. Reports how much of the case
set the pool covers, by acuity and by model pairing.

    python scripts/select_episodes.py runs/*.jsonl --out pool.jsonl --no-flags --nurse-ended
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def keep(ep: dict, no_flags: bool, nurse_ended: bool) -> bool:
    if ep.get("status") != "ok" or ep["final_triage"] is None or ep["final_triage"] != ep["case"].get("acuity"):
        return False
    if nurse_ended and ep["end_reason"] != "nurse_end":
        return False
    if no_flags and any((h.get("verification") or {}).get("passed") is False for h in ep["history"]):
        return False
    return True


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("runs", nargs="+", help="episode JSONL files from triagesim runs")
    p.add_argument("--out", required=True)
    p.add_argument("--no-flags", action="store_true", help="drop episodes with a line the dialogue master flagged")
    p.add_argument("--nurse-ended", action="store_true", help="drop episodes that hit max_turns")
    args = p.parse_args(argv)

    episodes = []
    for path in map(Path, args.runs):
        for line in path.read_text().splitlines():
            if line.strip():
                episodes.append({"source": path.name, **json.loads(line)})
    ok = [e for e in episodes if e.get("status") == "ok"]
    selected = [e for e in ok if keep(e, args.no_flags, args.nurse_ended)]
    Path(args.out).write_text("".join(json.dumps(e) + "\n" for e in selected))

    acuity = {e["case"]["case_id"]: e["case"]["acuity"] for e in ok}
    covered = {e["case"]["case_id"] for e in selected}
    print(f"{len(selected)} of {len(ok)} simulated episodes kept; {len(covered)} of {len(acuity)} cases covered")
    cases, hit = Counter(acuity.values()), Counter(acuity[c] for c in covered)
    for level in sorted(cases):
        print(f"  acuity {level}: {hit[level]}/{cases[level]} cases covered")
    pairs: dict[tuple, list[bool]] = defaultdict(list)
    for e in ok:
        pairs[(e["models"]["nurse"].split("/")[-1], e["models"]["patient"].split("/")[-1])].append(
            keep(e, args.no_flags, args.nurse_ended)
        )
    for (nurse, patient), kept in sorted(pairs.items()):
        print(f"  nurse {nurse} / patient {patient}: {sum(kept)}/{len(kept)} kept")


if __name__ == "__main__":
    main()
