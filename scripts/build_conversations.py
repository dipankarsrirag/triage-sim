"""Build the MIMIC-IV-ED-Ext-TriageSim conversations with the nurse's vital-sign readings in place.

Reads utterances_<split>.csv, vitals_<split>.csv and, if present, dialogues_<split>.csv (for the ESI
label) from the folder this script is in (or --data) and writes one conversation per dialogue, in which
each vital-sign reading follows the utterance after which the nurse has it. Standard library only
(Python 3.8 or newer).

    python build_conversations.py                         # every split found -> conversations_<split>.jsonl
    python build_conversations.py --split dev --format text

JSON lines: {"dialogue_id", "esi", "turns": [{"turn", "speaker", "text"} or
{"speaker": "vital", "vital", "value", "text"}, ...]}; utterances keep their turn numbers.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

NAMES = {"temperature": ("Temperature", "°F"), "heartrate": ("Heart rate", "per minute"),
         "resprate": ("Respiratory rate", "per minute"), "o2sat": ("Oxygen saturation", "%"),
         "sbp": ("Systolic blood pressure", "mmHg")}


def reading(vital, value):
    name, unit = NAMES[vital]
    if value == "":
        return f"{name}: not recorded"
    number = float(value)
    shown = f"{number:.1f}" if vital == "temperature" else f"{number:g}"
    return f"{name}: {shown}{unit}" if unit == "%" else f"{name}: {shown} {unit}"


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build(data, split):
    utterances, vitals = defaultdict(list), defaultdict(list)
    found = rows(data / f"utterances_{split}.csv")
    key = "dialogue_id" if "dialogue_id" in found[0] else "uid"
    for u in found:
        utterances[u[key]].append(u)
    for v in rows(data / f"vitals_{split}.csv"):
        vitals[v[key], int(v["after_turn"])].append(v)
    labels = data / f"dialogues_{split}.csv"
    dialogues = rows(labels) if labels.exists() else [{key: k} for k in utterances]
    for d in dialogues:
        turns = []
        for u in sorted(utterances[d[key]], key=lambda u: int(u["turn"])):
            turns.append({"turn": int(u["turn"]), "speaker": u["speaker"], "text": u["text"]})
            for v in vitals[d[key], int(u["turn"])]:
                turns.append({"speaker": "vital", "vital": v["vital"],
                              "value": float(v["value"]) if v["value"] else None, "text": reading(v["vital"], v["value"])})
        yield {key: d[key], **({"esi": int(d["esi"])} if "esi" in d else {}), "turns": turns}


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--data", type=Path, default=Path(__file__).resolve().parent, help="folder with the CSV files")
    p.add_argument("--split", action="append", help="train, dev, ... (default: every split found)")
    p.add_argument("--format", choices=("jsonl", "text"), default="jsonl")
    p.add_argument("--out", type=Path, default=Path("."), help="output folder")
    args = p.parse_args()
    splits = args.split or sorted(f.stem[len("utterances_"):] for f in args.data.glob("utterances_*.csv"))
    args.out.mkdir(parents=True, exist_ok=True)
    for split in splits:
        path = args.out / f"conversations_{split}.{'jsonl' if args.format == 'jsonl' else 'txt'}"
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for c in build(args.data, split):
                n += 1
                if args.format == "jsonl":
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                else:
                    f.write(f"### {c.get('dialogue_id') or c['uid']}\n")
                    for t in c["turns"]:
                        f.write(f"[{t['text']}]\n" if t["speaker"] == "vital" else f"{t['speaker'].capitalize()}: {t['text']}\n")
                    f.write("\n")
        print(f"{split}: {n} conversations -> {path}")


if __name__ == "__main__":
    main()
