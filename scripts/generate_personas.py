import argparse
import yaml
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from triagesim.config import OPENROUTER_API_KEY
from triagesim.personas.schema import PatientPersona, NursePersona


DEFAULT_MODEL = "google/gemini-3-pro-preview"


def main():
    parser = argparse.ArgumentParser(
        description="Generate personas for a single ethnicity"
    )
    parser.add_argument(
        "--ethnicity",
        required=True,
        help="Ethnicity to generate personas for (e.g. Australian, Chinese, Indian)",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="OpenRouter model name",
    )

    args = parser.parse_args()
    ethnicity = args.ethnicity
    model_name = args.model_name

    provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)

    # ─────────────────────────────────────────
    # Agents
    # ─────────────────────────────────────────

    patient_agent = Agent(
        OpenRouterModel(model_name, provider=provider),
        output_type=list[PatientPersona],
    )

    nurse_agent = Agent(
        OpenRouterModel(model_name, provider=provider),
        output_type=list[NursePersona],
    )

    # ─────────────────────────────────────────
    # Prompts
    # ─────────────────────────────────────────

    patient_prompt = f"""
You are an AI that outputs JSON matching the Pydantic schema for PatientPersona.

Generate a JSON array of EXACTLY 10 distinct patient persona objects.

Constraints:
- ethnicity MUST be "{ethnicity}"

Allowed values:

- age_group: ["adult", "elderly"]
- gender: ["male", "female"]
- socioeconomic_status: ["low", "medium", "high"]
- language_proficiency: ["basic", "intermediate", "fluent"]

- recall_accuracy: ["low", "medium", "high"]
- cognitive_state: ["clear", "mildly confused", "confused"]
- trust_in_healthcare: ["low", "medium", "high"]
- pain_expression: ["low", "medium", "high"]
- reactivity_to_clinician_emotion: ["low", "medium", "high"]
- emotion_regulation: ["stable", "reactive", "dysregulated"]

- disfluency_rate: ["low", "medium", "high"]
- topic_drift: ["low", "medium", "high"]
- verbosity: ["low", "medium", "high"]
- instruction: A free-text instruction to let the patient know how to speak. This will be used for text-to-speech control later.

Ensure meaningful variation across all non-ethnicity attributes.
Output ONLY valid JSON.
"""

    nurse_prompt = f"""
You are an AI that outputs JSON matching the Pydantic schema for NursePersona.

Generate a JSON array of EXACTLY 10 distinct nurse persona objects.

Constraints:
- ethnicity MUST be "{ethnicity}"

Allowed values:

- gender: ["male", "female"]
- experience_level: ["novice", "intermediate", "expert"]
- risk_tolerance: ["low", "medium", "high"]
- guideline_adherence: ["strict", "moderate", "flexible"]
- communication_style: ["empathetic", "neutral", "direct", "supportive"]
- verbosity: ["low", "medium", "high"]
- emotional_expression: ["low", "medium", "high"]
- instruction: A free-text instruction to let the nurse know how to speak. This will be used for text-to-speech control later.

Ensure meaningful variation across all non-ethnicity attributes.
Output ONLY valid JSON.
"""

    # ─────────────────────────────────────────
    # Run generation
    # ─────────────────────────────────────────

    print(f"Generating patient personas for ethnicity={ethnicity}")
    patient_resp = patient_agent.run_sync(patient_prompt)
    patient_personas: list[PatientPersona] = patient_resp.output

    print(f"Generating nurse personas for ethnicity={ethnicity}")
    nurse_resp = nurse_agent.run_sync(nurse_prompt)
    nurse_personas: list[NursePersona] = nurse_resp.output

    # ─────────────────────────────────────────
    # Write YAML
    # ─────────────────────────────────────────

    base_dir = Path("./data/personas") / model_name.split("/")[-1] / ethnicity
    base_dir.mkdir(parents=True, exist_ok=True)

    patient_path = base_dir / "patient.yaml"
    nurse_path = base_dir / "nurse.yaml"

    with open(patient_path, "w") as f:
        yaml.safe_dump([p.model_dump() for p in patient_personas], f, sort_keys=False)

    with open(nurse_path, "w") as f:
        yaml.safe_dump([n.model_dump() for n in nurse_personas], f, sort_keys=False)

    print(f"✔ wrote {patient_path}")
    print(f"✔ wrote {nurse_path}")


if __name__ == "__main__":
    main()
