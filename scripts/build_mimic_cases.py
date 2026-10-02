"""
Build triagesim cases from the MIMIC-IV-ED tables.

Joins triage.csv (vitals, pain, acuity, chief complaint) with edstays.csv (gender, arrival, race,
disposition, and the patient's ED visits in the 12 months before this one), diagnosis.csv (ED ICD
titles, in sequence order) and medrecon.csv (home medications). Race and disposition are kept for
analysis only: triagesim never shows them to the models.

Stays are either listed in --stays (any CSV with a stay_id column; its specialisation column is kept
if present) or drawn from every eligible stay: acuity recorded, chief complaint present and not
redacted ("___"), all five vitals present and physiologically plausible, and pain a number from 0 to
10. --per-acuity samples up to N stays per acuity level; --sample draws N at random (the natural
acuity mix).

    python scripts/build_mimic_cases.py --mimic-dir /path/to/mimic-iv-ed --per-acuity 2000 \
        --out data/cases/mimic.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def read_for_stays(path: Path, stays: set, columns: list[str]) -> pd.DataFrame:
    chunks = pd.read_csv(path, usecols=columns, chunksize=500_000)
    return pd.concat(chunk[chunk["stay_id"].isin(stays)] for chunk in chunks)


VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp"]
PLAUSIBLE = {"temperature": (90, 108), "heartrate": (20, 250), "resprate": (4, 60), "o2sat": (50, 100), "sbp": (40, 300)}


def eligible(triage: pd.DataFrame) -> pd.Series:
    complaint = triage["chiefcomplaint"]
    ok = triage["acuity"].notna() & complaint.notna() & ~complaint.astype(str).str.contains("___", regex=False)
    ok &= pd.to_numeric(triage["pain"], errors="coerce").between(0, 10)
    for name, (low, high) in PLAUSIBLE.items():
        ok &= triage[name].between(low, high)  # also drops missing vitals
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--mimic-dir", required=True)
    p.add_argument("--stays", help="CSV with a stay_id column (default: every eligible stay)")
    p.add_argument("--per-acuity", type=int, help="sample up to N eligible stays per acuity level")
    p.add_argument("--sample", type=int, help="sample N eligible stays at random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    root = Path(args.mimic_dir)
    triage = pd.read_csv(root / "triage.csv", usecols=["subject_id", "stay_id", *VITALS, "pain", "acuity",
                                                       "chiefcomplaint"])
    if args.stays:
        selection = pd.read_csv(args.stays)
        triage = triage[triage["stay_id"].isin(set(selection["stay_id"]))]
    else:
        selection = pd.DataFrame()
        pool = triage[eligible(triage)]
        if args.per_acuity:
            triage = pool.sample(frac=1, random_state=args.seed).groupby("acuity").head(args.per_acuity)
        elif args.sample:
            triage = pool.sample(min(len(pool), args.sample), random_state=args.seed)
        else:
            triage = pool
        print(f"{len(pool):,} eligible stays; selected {len(triage):,}")
    stays = set(triage["stay_id"])
    edstays = read_for_stays(root / "edstays.csv", stays, ["stay_id", "gender", "race", "arrival_transport",
                                                           "disposition"])
    diagnoses = read_for_stays(root / "diagnosis.csv", stays, ["stay_id", "seq_num", "icd_title"])
    meds = read_for_stays(root / "medrecon.csv", stays, ["stay_id", "name"])

    dx = diagnoses.sort_values("seq_num").groupby("stay_id")["icd_title"].agg(lambda t: [s.capitalize() for s in t])
    rx = meds.groupby("stay_id")["name"].agg(lambda n: sorted({s.lower() for s in n.dropna()}))
    cases = triage.merge(edstays, on="stay_id", how="left")

    # earlier ED visits by the same patient in the past year (dates are shifted per patient in MIMIC,
    # but intervals within a patient are preserved)
    visits = pd.read_csv(root / "edstays.csv", usecols=["subject_id", "stay_id", "intime"], parse_dates=["intime"])
    visits = visits[visits["subject_id"].isin(set(cases["subject_id"]))]
    this = visits[visits["stay_id"].isin(stays)].rename(columns={"stay_id": "this_stay", "intime": "this_time"})
    pairs = this.merge(visits, on="subject_id")
    earlier = (pairs["intime"] < pairs["this_time"]) & (pairs["intime"] >= pairs["this_time"] - pd.Timedelta(days=365))
    prior = pairs[earlier].groupby("this_stay").size()
    cases["prior_ed_visits"] = cases["stay_id"].map(prior).fillna(0).astype(int)
    if "specialisation" in selection:
        cases = cases.merge(selection[["stay_id", "specialisation"]], on="stay_id", how="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in cases.to_dict("records"):
            sid = row.pop("stay_id")
            row = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}
            row.update(
                case_id=str(sid),
                chiefcomplaint=str(row["chiefcomplaint"]).lower(),
                diagnoses=dx.get(sid, []),
                medications=rx.get(sid, []),
                dataset="mimic",
                acuity_scale="esi",
            )
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(cases):,} cases to {out}; acuity {cases['acuity'].value_counts().sort_index().astype(int).to_dict()}")


if __name__ == "__main__":
    main()
