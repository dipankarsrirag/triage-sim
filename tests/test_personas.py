from pathlib import Path

import pytest
from pydantic import ValidationError

from triagesim import prompts
from triagesim.personas import (
    NURSE_TRAITS,
    PATIENT_TRAITS,
    NursePersona,
    PatientPersona,
    combinations,
    INCOMPATIBLE,
    compatible,
    load_personas,
)

EXAMPLES = Path(__file__).parent.parent / "examples" / "personas"


def test_example_personas_load():
    assert len(load_personas(EXAMPLES / "nurse.yaml", NursePersona)) == 3
    assert len(load_personas(EXAMPLES / "patient.yaml", PatientPersona)) == 3


def test_levels_are_validated(nurse_persona):
    with pytest.raises(ValidationError):
        NursePersona(**{**nurse_persona.model_dump(), "risk_attitude": "low"})
    with pytest.raises(ValidationError):  # a missing trait is an error, not "unspecified"
        NursePersona(**{k: v for k, v in nurse_persona.model_dump().items() if k != "workload"})


def test_jsonl_pools_load(tmp_path, patient_persona):
    path = tmp_path / "patient.jsonl"
    path.write_text(patient_persona.model_dump_json() + "\n")
    assert load_personas(path, PatientPersona) == [patient_persona]


def test_every_trait_level_has_an_anchor_and_sources():
    for traits in (PATIENT_TRAITS, NURSE_TRAITS):
        for name, trait in traits.items():
            assert trait.definition and trait.sources and len(trait.levels) >= 2, name
            assert all(anchor for anchor in trait.levels.values()), name


def test_combinations_cover_the_space():
    assert sum(1 for _ in combinations(NURSE_TRAITS)) == 3 ** len(NURSE_TRAITS)


def test_reviewed_incompatible_pairs_filter_combinations():
    combo = {"verbosity": "terse", "focus": "frequent_tangents", "disfluency": "low"}
    assert not compatible(combo)
    assert compatible({**combo, "focus": "occasional_tangents"})
    for pair in INCOMPATIBLE:  # every excluded pair names real traits and levels
        for trait, level in pair:
            assert level in {**PATIENT_TRAITS, **NURSE_TRAITS}[trait].levels


def test_prompts_show_anchors_but_not_ethnicity_or_instruction(patient_persona):
    text = prompts._persona(patient_persona.model_copy(update={"instruction": "Speak with an accent."}))
    assert "disclosure: open (answers questions on sensitive topics truthfully" in text
    assert "Australian" not in text and "accent" not in text


def test_line_checks_skip_episode_level_traits(nurse_persona):
    full = prompts._persona(nurse_persona)
    per_line = prompts._persona(nurse_persona, per_line_only=True)
    assert "risk attitude" in full and "risk attitude" not in per_line and "questioning style" in per_line


def test_generate_personas_script(tmp_path):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("gen", Path(__file__).parent.parent / "scripts" / "generate_personas.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    sys.argv = ["gen", "--out-dir", str(tmp_path)]
    gen.main()
    nurses = load_personas(tmp_path / "nurse.jsonl", NursePersona)
    assert len(nurses) == 2 * (3 ** len(NURSE_TRAITS) - 3 ** (len(NURSE_TRAITS) - 2))  # minus novice + intuition-led
    assert not any(n.expertise == "novice" and n.algorithm_reliance == "intuition_led" for n in nurses)
