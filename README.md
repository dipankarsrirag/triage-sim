# TriageSim

**TriageSim** is an open-source Python framework for generating synthetic, multi-speaker spoken dialogues for emergency department (ED) triage. It produces paired structured EHR → dialogue data grounded in real clinical vignettes, enabling controlled evaluation of speech and language systems in healthcare.

---

## Features

- **Nurse ↔ patient dialogue simulation** with LLM agents served by **vLLM** (local, in-process) or **OpenRouter** (hosted models: Claude, Gemini, GPT, ...)
- **Batched generation**: many episodes run side by side and every round of LLM calls goes to vLLM as one batch
- **Schema-constrained outputs**: agents answer in JSON that is enforced during decoding, including which actions and vitals the nurse may pick at each step
- **Two triage algorithms**: ESI (Emergency Severity Index) and ATS (Australasian Triage Scale)
- **Research-grounded personas**: combinatorial patient and nurse personas built from ED-research traits with behavioural anchors (health literacy, symptom reporting, disclosure, questioning style, ...)
- **Structured run artifacts**: dialogue, vitals checked, red flags logged, per-step triage reasoning and per-turn belief extraction
- **Reproducible and resumable** runs to JSONL, straight from MIMIC-IV-ED `triage.csv` or your own vignettes
- **Optional audio rendering**: XTTS-v2 voice cloning for multi-speaker TTS synthesis

---

## Installation

```bash
pip install triagesim            # OpenRouter backend
pip install "triagesim[vllm]"    # + local models with vLLM (needs a CUDA GPU)
pip install "triagesim[audio]"   # + XTTS-v2 speech rendering
```

Or the latest development version from GitHub:
```bash
pip install "git+https://github.com/dipankarsrirag/triage-sim.git"
```

For OpenRouter, set your [API key](https://openrouter.ai/) in the environment or in a `.env` file:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

---

## Quick Start (command line)

Models are given as `vllm:<local path or Hugging Face id>` or `openrouter:<model id>`:

```bash
triagesim --model vllm:google/gemma-4-E4B-it \
    --cases examples/cases.jsonl \
    --nurse-personas examples/personas/nurse.yaml \
    --patient-personas examples/personas/patient.yaml \
    --out outputs/gemma.jsonl \
    --vllm-args '{"max_model_len": 16384}'
```

```bash
triagesim --model openrouter:anthropic/claude-sonnet-4.5 --algorithm ats --episodes-per-case 3 \
    --cases examples/cases.jsonl \
    --nurse-personas examples/personas/nurse.yaml \
    --patient-personas examples/personas/patient.yaml \
    --out outputs/claude.jsonl
```

Each role can use its own model (`--nurse-model`, `--patient-model`, `--belief-model`, `--judge-model`),
each either vLLM or OpenRouter, e.g. a hosted nurse against a local patient simulator. Different vLLM
models share one GPU and split its memory evenly unless `--vllm-args` sets `gpu_memory_utilization`.

`--judge-model` turns on the dialogue master, which screens cases, writes the patient's script and
checks every spoken line (`--max-regenerations`, default 1).

Episodes are appended to `--out` as they finish; re-running the same command skips episodes already
there, so an interrupted run just resumes. Episodes whose LLM calls keep failing go to
`<out>.errors.jsonl` and are retried on the next run. At the end the command prints summary metrics.
`triagesim --help` lists the other options (`--max-turns`, `--sampling`, `--concurrency`, `--seed`, ...);
arguments can also come from a file, one per line: `triagesim @run.args`.

---

## Python API

```python
from triagesim import (
    VLLM, OpenRouter, NursePersona, PatientPersona,
    load_cases, load_personas, simulate, summarize,
)

llm = VLLM("google/gemma-4-E4B-it", max_model_len=16384)   # or OpenRouter("anthropic/claude-sonnet-4.5")

episodes = list(simulate(
    load_cases("examples/cases.jsonl"),
    nurse_llm=llm,
    patient_llm=llm,
    nurse_personas=load_personas("examples/personas/nurse.yaml", NursePersona),
    patient_personas=load_personas("examples/personas/patient.yaml", PatientPersona),
    max_turns=12,
    judge_llm=llm,               # optional dialogue master
    sampling={"nurse": {"temperature": 0.7}},
))
print(summarize(episodes))
```

`simulate` yields one dict per episode as it finishes, so large runs can be streamed to disk.
Keyword arguments to `VLLM` go to `vllm.LLM` (plus `chat_template_kwargs`, e.g.
`{"enable_thinking": False}`); sampling parameters per role (`nurse`, `patient`, `belief`, `judge`)
override the model's `generation_config` defaults.

---

## Cases

A case is the ground truth for one presentation. The nurse never sees it: it learns only what the
patient says, what is visible at the triage desk (gender, age group, arrival mode) and the vital signs
it asks for. The patient knows its complaint, pain, and how unwell it is (the acuity, in plain words).

```json
{"case_id": "syncope-01", "chief_complaint": "Syncope", "vitals": {"temperature": 99.1, "heartrate": 112, "resprate": 26, "o2sat": 91, "sbp": 98}, "acuity": 2, "acuity_scale": "esi", "pain": 7, "gender": "male", "arrival_transport": "ambulance"}
```

`load_cases` reads `.jsonl`, `.json` or `.csv`. Vitals may also be flat columns, so MIMIC-IV-ED
`triage.csv` loads as is (`stay_id` becomes the case id, `chiefcomplaint` the chief complaint, and
blank values become "not recorded"). Optional fields: `gender` (patient personas are matched to it),
`acuity_scale` (`esi`/`ats`: the scale the nurse triages with unless `--algorithm` is given),
`diagnoses` and `medications` (e.g. MIMIC-IV-ED ICD titles and medrecon; they ground the patient's
script). Any other field is kept in the artifacts. `scripts/build_mimic_cases.py` builds cases from the
MIMIC-IV-ED tables.

### The dialogue master

With `--judge-model`, the dialogue master runs each episode around the two agents:

1. **In-situ check.** It reasons from the case whether the patient would have a triage conversation
   at the desk at all (not, for example, when they go straight to resuscitation). If not, the episode
   is recorded with status `"skipped"` and the master's reasoning. Cases without a ground-truth acuity
   are always skipped.
2. **Patient script.** It writes a standardized-patient script from the case and the patient persona:
   story, onset, quality, associated symptoms, pertinent negatives, history, medications,
   allergies. The patient answers from it.
3. **Line checks.** It verifies every spoken line (see below).

### Building an evaluation pool

Run many simulations, over many model pairings and personas, then pool them and keep the episodes
whose nurse reached the ground-truth level (with optional quality filters):

```bash
python scripts/select_episodes.py runs/*.jsonl --out pool.jsonl --no-flags --nurse-ended
```

It reports how many cases the pool covers, by acuity and by model pairing.

---

## Personas

Personas are combinations of traits grounded in emergency-department research. Each trait has a
definition, levels anchored in observable behaviour, and sources; the registry in
`triagesim/personas.py` is the single source of truth, and `docs/persona-traits-review.md` holds the
evidence and the design decisions.

| Patient trait | Levels |
|---|---|
| `age_group` | adult, older |
| `english_proficiency` | fluent, functional, limited |
| `health_literacy` | limited, marginal, adequate |
| `recall_reliability` | reliable, patchy, unreliable |
| `symptom_reporting` | minimising, faithful, amplifying |
| `distress` | calm, anxious, highly_distressed |
| `disclosure` | open, selective, guarded |
| `demeanour` | cooperative, impatient, hostile |
| `verbosity` | terse, typical, expansive |
| `focus` | on_target, occasional_tangents, frequent_tangents |
| `disfluency` | low, moderate, high |

| Nurse trait (interview style only) | Levels |
|---|---|
| `expertise` | novice, competent, expert |
| `algorithm_reliance` | protocol_driven, blended, intuition_led |
| `risk_attitude` | risk_averse, balanced, risk_accepting |
| `self_report_credence` | accepting, corroborating, discounting |
| `questioning_style` | open_facilitative, mixed, closed_directive |
| `emotional_responsiveness` | task_focused, acknowledging, empathic |
| `workload` | quiet, busy, overloaded |

The models see each trait with its behavioural anchor. A patient's gender comes from the case, and
its cognitive state and intoxication are set by the dialogue master from the case (they are triage
signs, not personality). Nurse traits shape which questions are asked and how, never the level
assigned. `ethnicity` and `instruction` (voice direction for TTS) may be stored on a persona but are
never shown to the models.

Pools are generated combinatorially: every combination of levels except the two pairs that
contradict each other by definition (`personas.INCOMPATIBLE`; the review of all trait pairs is in
`docs/persona-pair-review.md`). That gives 104,976 patient and 3,888 nurse personas:

```bash
python scripts/generate_personas.py --out-dir data/personas
```

`load_personas` reads these JSONL pools, or YAML lists such as `examples/personas/`.

---

## How an Episode Runs

Each turn the nurse acts until it speaks: it may first check **one** vital sign (`check_vital`, the
value comes from the case) and log red flags once (`log_red_flag`), then either asks the patient
something (`utterance`) or ends triage (`end`). Every nurse step also gives its rationale, then its
current triage estimate and confidence. The patient then replies once, and the nurse's model extracts
what the reply revealed (chief complaint, pain, duration, symptoms, red flags). The episode ends when
the nurse ends it (its triage level then is the final one) or after `max_turns` patient replies.

With a dialogue master, every nurse and patient line is checked before it enters the dialogue for
faithfulness (to the case and script, or to what the nurse has actually learned), information gain,
and persona adherence. A rejected line is regenerated with the critique; if it still fails it is kept
and flagged, with the rejected drafts, for post-hoc filtering.

Every patient line also records what information passed, from three sides, under `information`:
`disclosed` (the patient's own one-sentence account of what it revealed), `understood` (the nurse's
one-sentence reading of it, written in its next step before it acts) and, with a dialogue master,
`conveyed` (the master's reading of the utterance's words alone, without the case or the dialogue).

---

## Run Artifact

Each episode is a dict (one JSONL line):

| Key | Description |
|---|---|
| `episode_id` | `<case_id>-<k>` |
| `status`, `error` | `"ok"`, or `"error"` with the reason (dialogue kept up to the failure) |
| `case` | The ground-truth case |
| `algorithm`, `models` | Triage algorithm and the model used for each role |
| `nurse_persona`, `patient_persona` | Personas drawn for this episode |
| `patient_script` | The script drawn for this episode, if any |
| `end_reason` | `"nurse_end"` or `"max_turns"` |
| `final_triage`, `num_turns` | The nurse's triage level at its last step; number of patient replies |
| `history` | Dialogue and events: utterances (patient lines with `information`, and `verification` when a judge is used), vitals, triage end |
| `trace` | Every nurse step: action, vital, utterance, triage, confidence, red flags, explanation |
| `red_flags` | Red flags logged by the nurse (deduplicated) |
| `beliefs` | Per patient turn, the information extracted from the reply |

---

## Evaluation Metrics

```python
from triagesim import score, summarize

score(episode)       # correctness, absolute error, over/under-triage, first correct turn, flagged lines, ...
summarize(episodes)  # accuracy, mean absolute error, over/under-triage, flag rates, ...
```

---

## Running on a PBS Cluster

`jobs/simulate.pbs` runs a simulation on one GPU (vLLM start-up takes a few minutes, so batch many
episodes per job), `jobs/evaluate.pbs` measures triage accuracy on the seed datasets in `data/`
(`scripts/evaluate.py`), and `jobs/smoke_test.pbs` checks both backends end to end on a small model.
They are set up for UNSW Katana; adjust the module and scratch paths for your cluster.

---

## Citation

If you use TriageSim in your research, please cite:

```bibtex
@misc{srirag2026triagesimconversationalemergencytriage,
      title={TriageSim: A Conversational Emergency Triage Simulation Framework from Structured Electronic Health Records}, 
      author={Dipankar Srirag and Quoc Dung Nguyen and Aditya Joshi and Padmanesan Narasimhan and Salil Kanhere},
      year={2026},
      eprint={2603.10035},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.10035}, 
}
```

---

## License

MIT © Dipankar Srirag
