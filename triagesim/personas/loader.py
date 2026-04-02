import random
import yaml
from pathlib import Path
from typing import List, Dict, Any

from triagesim.personas.schema import PatientPersona, NursePersona


# ─────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────


def _fill_defaults(entry: Dict[str, Any], required_fields: set[str]) -> Dict[str, Any]:
    """
    Fill missing persona fields with a neutral default.
    """
    filled = dict(entry)
    for field in required_fields:
        if field not in filled:
            filled[field] = "unspecified"
    return filled


# ─────────────────────────────────────────
# Loader functions
# ─────────────────────────────────────────


def load_patient_personas(filepath: str) -> List[PatientPersona]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Patient persona file not found: {filepath}")

    raw_list = yaml.safe_load(path.read_text())
    if not isinstance(raw_list, list):
        raise ValueError("Patient persona YAML must be a list")

    required_fields = set(PatientPersona.model_fields.keys())

    personas: List[PatientPersona] = []
    for entry in raw_list:
        entry = _fill_defaults(entry, required_fields)
        personas.append(PatientPersona.model_validate(entry))

    return personas


def load_nurse_personas(filepath: str) -> List[NursePersona]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Nurse persona file not found: {filepath}")

    raw_list = yaml.safe_load(path.read_text())
    if not isinstance(raw_list, list):
        raise ValueError("Nurse persona YAML must be a list")

    required_fields = set(NursePersona.model_fields.keys())

    personas: List[NursePersona] = []
    for entry in raw_list:
        entry = _fill_defaults(entry, required_fields)
        personas.append(NursePersona.model_validate(entry))

    return personas


# ─────────────────────────────────────────
# Sampling utilities
# ─────────────────────────────────────────


def sample_patient_personas(
    personas: List[PatientPersona],
    k: int = 1,
    seed: int | None = None,
) -> List[PatientPersona]:
    if k > len(personas):
        raise ValueError(
            f"Cannot sample k={k} personas from population of size {len(personas)}"
        )

    rng = random.Random(seed)
    return rng.sample(personas, k)


def sample_nurse_personas(
    personas: List[NursePersona],
    k: int = 1,
    seed: int | None = None,
) -> List[NursePersona]:
    if k > len(personas):
        raise ValueError(
            f"Cannot sample k={k} personas from population of size {len(personas)}"
        )

    rng = random.Random(seed)
    return rng.sample(personas, k)


# ─────────────────────────────────────────
# Filtering support
# ─────────────────────────────────────────


def filter_patient_personas(
    personas: List[PatientPersona],
    criteria: Dict[str, str],
) -> List[PatientPersona]:
    results = []
    for persona in personas:
        if all(getattr(persona, k) == v for k, v in criteria.items()):
            results.append(persona)
    return results


def filter_nurse_personas(
    personas: List[NursePersona],
    criteria: Dict[str, str],
) -> List[NursePersona]:
    results = []
    for persona in personas:
        if all(getattr(persona, k) == v for k, v in criteria.items()):
            results.append(persona)
    return results
