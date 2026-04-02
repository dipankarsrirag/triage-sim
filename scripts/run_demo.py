from pprint import pprint
import json
from pathlib import Path

from triagesim.agents.base_agent import OpenRouterLLM
from triagesim.agents.patient_agent import PatientAgent
from triagesim.agents.nurse_agent import NurseAgent

from triagesim.runner import TriageRunner, RunnerConfig
from triagesim.core.output_schema import NurseOutput, PatientOutput

from triagesim.personas.loader import (
    load_patient_personas,
    load_nurse_personas,
    sample_patient_personas,
    sample_nurse_personas,
)

# ─────────────────────────────────────────
# Load persona YAMLs
# ─────────────────────────────────────────

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEMO_FILE = OUTPUT_DIR / "demo.json"

patients = load_patient_personas("./data/personas/gemini-3-pro-preview/Australian/patient.yaml")
nurses = load_nurse_personas("./data/personas/gemini-3-pro-preview/Australian/nurse.yaml")

patient_persona = sample_patient_personas(patients, k=1, seed=4)[0]
nurse_persona = sample_nurse_personas(nurses, k=1, seed=4)[0]

print("Selected Patient Persona:")
pprint(patient_persona.model_dump())

print("\nSelected Nurse Persona:")
pprint(nurse_persona.model_dump())

# ─────────────────────────────────────────
# Ground truth vignette
# (NOT visible to agents)
# ─────────────────────────────────────────
model_name = "anthropic/claude-sonnet-4.5"

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

# ─────────────────────────────────────────
# LLM backends
# ─────────────────────────────────────────

patient_llm = OpenRouterLLM(
    model_name=model_name,
    output_type=PatientOutput,
)

nurse_llm = OpenRouterLLM(
    model_name=model_name,
    output_type=NurseOutput,
)

# ─────────────────────────────────────────
# Agents
# ─────────────────────────────────────────

patient = PatientAgent(
    llm=patient_llm,
    persona=patient_persona,
)

nurse = NurseAgent(
    llm=nurse_llm,
    persona=nurse_persona,
    algorithm="esi",  # or "ats"
)

# ─────────────────────────────────────────
# Runner config
# ─────────────────────────────────────────

config = RunnerConfig(
    max_turns=12,
    store_backend="memory",  # or "redis"
    seed=42,
)

# ─────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────

runner = TriageRunner(
    nurse_agent=nurse,
    patient_agent=patient,
    ground_truth=ground_truth,
    config=config,
)

run_artifact = runner.run()

with open(DEMO_FILE, "w") as f:
    json.dump(run_artifact, f, indent=2)

print(f"\nSaved demo run to {DEMO_FILE}")

# ─────────────────────────────────────────
# Output
# ─────────────────────────────────────────

print("\n=== HISTORY ===")
for step in run_artifact["history"]:
    pprint(step)

print("\n=== TRACE (Nurse cognition) ===")
for step in run_artifact["trace"]:
    pprint(step)

print("\n=== FINAL BELIEF STATE ===")
pprint(run_artifact["belief"])

print("\n=== RED FLAGS LOGGED ===")
pprint(run_artifact["red_flags"])
