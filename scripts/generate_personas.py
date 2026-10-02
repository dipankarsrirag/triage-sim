"""
Generate persona pools combinatorially: every combination of trait levels (triagesim.personas), except
those containing a pair of levels that contradict each other (personas.INCOMPATIBLE, reviewed in
docs/persona-pair-review.md). Nurses are generated for each gender; a patient's gender comes from the case.

    python scripts/generate_personas.py --out-dir data/personas
"""

import argparse
from pathlib import Path

from triagesim.personas import NURSE_TRAITS, PATIENT_TRAITS, NursePersona, PatientPersona, combinations, compatible


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for role, traits, model, extra in [
        ("patient", PATIENT_TRAITS, PatientPersona, [{}]),
        ("nurse", NURSE_TRAITS, NursePersona, [{"gender": "female"}, {"gender": "male"}]),
    ]:
        total = kept = 0
        with (out_dir / f"{role}.jsonl").open("w") as f:
            for combo in combinations(traits):
                total += 1
                if compatible(combo):
                    for fields in extra:
                        f.write(model(**combo, **fields).model_dump_json(exclude_none=True) + "\n")
                        kept += 1
        print(f"{role}: {total} combinations, {kept} personas written to {out_dir / f'{role}.jsonl'}")


if __name__ == "__main__":
    main()
