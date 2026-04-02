import pytest

from triagesim.personas.loader import (
    load_patient_personas,
    load_nurse_personas,
    sample_patient_personas,
)
from triagesim.personas.schema import PatientPersona, NursePersona


@pytest.fixture
def patient_yaml(tmp_path):
    path = tmp_path / "patients.yaml"
    path.write_text(
        """
- age_group: adult
  gender: male
  verbosity: low
  trust_in_healthcare: high
"""
    )
    return path


@pytest.fixture
def nurse_yaml(tmp_path):
    path = tmp_path / "nurses.yaml"
    path.write_text(
        """
- experience_level: senior
  communication_style: calm
"""
    )
    return path


def test_load_patient_personas(patient_yaml):
    personas = load_patient_personas(patient_yaml)
    assert len(personas) == 1
    assert isinstance(personas[0], PatientPersona)


def test_load_nurse_personas(nurse_yaml):
    personas = load_nurse_personas(nurse_yaml)
    assert len(personas) == 1
    assert isinstance(personas[0], NursePersona)


def test_sample_patient_personas_is_deterministic(patient_yaml):
    personas = load_patient_personas(patient_yaml)

    s1 = sample_patient_personas(personas, k=1, seed=42)
    s2 = sample_patient_personas(personas, k=1, seed=42)

    assert s1[0].model_dump() == s2[0].model_dump()


def test_sample_k_greater_than_population(patient_yaml):
    personas = load_patient_personas(patient_yaml)

    with pytest.raises(ValueError):
        sample_patient_personas(personas, k=2, seed=0)
