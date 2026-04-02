# TriageSim

**TriageSim** is an open-source Python framework for generating synthetic, multi-speaker spoken dialogues for emergency department (ED) triage. It produces paired structured EHR → dialogue data grounded in real clinical vignettes, enabling controlled evaluation of speech and language systems in healthcare.

---

## Features

- **Nurse ↔ patient dialogue simulation** via LLM agents (OpenRouter)
- **Two triage algorithms**: ESI (Emergency Severity Index) and ATS (Australasian Triage Scale)
- **Persona-conditioned generation**: diverse patient and nurse personas (ethnicity, experience, communication style, etc.)
- **Structured run artifacts**: belief state, red-flag trace, per-turn triage reasoning
- **Pluggable LLM backends**: any OpenRouter-compatible model (Claude, Gemini, GPT, etc.)
- **Optional audio rendering**: XTTS-v2 voice cloning for multi-speaker TTS synthesis

---

## Installation

```bash
pip install triagesim
```

With Redis state store:
```bash
pip install "triagesim[redis]"
```

With audio rendering (requires PyTorch):
```bash
pip install "triagesim[audio]"
```

Or install the latest development version directly from GitHub:
```bash
pip install "git+https://github.com/dipankarsrirag/triage-sim.git"
```

---

## Configuration

Set your [OpenRouter](https://openrouter.ai/) API key as an environment variable or in a `.env` file at your project root:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

```ini
# .env
OPENROUTER_API_KEY=sk-or-...
```

---

## Quick Start

```python
from triagesim import TriageRunner, RunnerConfig
from triagesim.agents import OpenRouterLLM, NurseAgent, PatientAgent
from triagesim.core import NurseOutput, PatientOutput
from triagesim.personas import (
    load_patient_personas,
    load_nurse_personas,
    sample_patient_personas,
    sample_nurse_personas,
)

# Load personas from YAML files
patients = load_patient_personas("path/to/patient.yaml")
nurses = load_nurse_personas("path/to/nurse.yaml")

patient_persona = sample_patient_personas(patients, k=1, seed=42)[0]
nurse_persona = sample_nurse_personas(nurses, k=1, seed=42)[0]

# Ground-truth clinical vignette (NOT visible to agents)
ground_truth = {
    "chiefcomplaint": "Syncope",
    "vitals": {
        "temperature": 99.1,
        "heartrate": 112,
        "resprate": 26,
        "o2sat": 91,
        "sbp": 98,
    },
    "acuity": 2,
    "pain": 7,
}

# Instantiate LLM backends — one per agent, each typed to its output schema
model = "anthropic/claude-sonnet-4-5"
patient_llm = OpenRouterLLM(model_name=model, output_type=PatientOutput)
nurse_llm = OpenRouterLLM(model_name=model, output_type=NurseOutput)

# Instantiate agents
patient = PatientAgent(llm=patient_llm, persona=patient_persona)
nurse = NurseAgent(llm=nurse_llm, persona=nurse_persona, algorithm="esi")  # or "ats"

# Run
config = RunnerConfig(max_turns=20, store_backend="memory", seed=42)
runner = TriageRunner(
    nurse_agent=nurse,
    patient_agent=patient,
    ground_truth=ground_truth,
    config=config,
)
artifact = runner.run()
```

---

## Persona Format

Personas are loaded from YAML files. Each file is a list of persona dicts.

**Patient persona fields:**

```yaml
- age_group: adult
  gender: female
  ethnicity: Australian
  socioeconomic_status: middle
  language_proficiency: high
  recall_accuracy: high
  cognitive_state: clear
  trust_in_healthcare: high
  pain_expression: moderate
  reactivity_to_clinician_emotion: low
  emotion_regulation: stable
  disfluency_rate: low
  topic_drift: low
  response_length: medium
```

**Nurse persona fields:**

```yaml
- gender: female
  ethnicity: Australian
  experience_level: senior
  risk_tolerance: low
  guideline_adherence: high
  communication_style: direct
  verbosity: medium
  emotional_expression: neutral
```

---

## Run Artifact

`runner.run()` returns a dict:

| Key | Description |
|---|---|
| `run_id` | Unique run identifier |
| `ground_truth` | The input clinical vignette |
| `history` | Full dialogue turn list (utterances + vitals + events) |
| `trace` | Per-turn nurse cognition trace (actions + reasoning) |
| `belief` | Final inferred belief state |
| `red_flags` | Red flags logged during triage |
| `state` | Final simulation state |

---

## Evaluation Metrics

```python
from triagesim.utils import compute_all_metrics

metrics = compute_all_metrics(
    trace=artifact["trace"],
    belief=artifact["belief"],
    ground_truth=ground_truth,
)
# Returns: triage accuracy, belief coverage, red-flag P/R/F1, explanation stats
```

---

## Using Redis State Store

```python
config = RunnerConfig(
    max_turns=20,
    store_backend="redis",
    redis_db=0,
    seed=42,
)
```

Requires a running Redis instance and `pip install "triagesim[redis]"`.

---

## Custom LLM Backends

Subclass `BaseLLM` to use any model provider:

```python
from triagesim.agents import BaseLLM

class MyLLM(BaseLLM):
    def generate(self, prompt, **kwargs):
        # call your model here
        ...
```

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
