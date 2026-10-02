import pytest
from pydantic import ValidationError

from triagesim.schemas import BeliefExtraction, NurseOutput, PatientOutput, nurse_output_type

BASE = {"understood": None, "vital": None, "utterance": None, "triage": 3, "confidence": "low", "red_flags": [], "explanation": "x"}


def test_narrowed_schema_restricts_enums_and_keeps_field_order():
    schema = nurse_output_type(("utterance", "check_vital", "end"), ("o2sat", "sbp")).model_json_schema()
    props = schema["properties"]
    assert list(props)[:4] == ["understood", "action", "vital", "utterance"]
    assert props["action"]["enum"] == ["utterance", "check_vital", "end"]
    assert props["vital"]["anyOf"][0]["enum"] == ["o2sat", "sbp"]
    assert props["triage"]["enum"] == [1, 2, 3, 4, 5]
    assert set(schema["required"]) == set(props)
    assert schema["additionalProperties"] is False


def test_vital_is_null_when_no_vitals_allowed():
    schema = nurse_output_type(("utterance", "end"), ()).model_json_schema()
    assert schema["properties"]["vital"]["type"] == "null"


def test_narrowed_type_is_cached_and_a_nurse_output():
    t = nurse_output_type(("utterance", "end"), ())
    assert t is nurse_output_type(("utterance", "end"), ())
    assert isinstance(t.model_validate({**BASE, "action": "end"}), NurseOutput)


def test_narrowed_type_rejects_disallowed_action():
    with pytest.raises(ValidationError):
        nurse_output_type(("utterance", "end"), ()).model_validate({**BASE, "action": "log_red_flag", "red_flags": ["x"]})


@pytest.mark.parametrize(
    "fields",
    [
        {"action": "utterance", "utterance": "  "},
        {"action": "check_vital", "vital": None},
        {"action": "log_red_flag", "red_flags": [" "]},
    ],
)
def test_payload_must_match_action(fields):
    with pytest.raises(ValidationError):
        NurseOutput.model_validate({**BASE, **fields})


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        PatientOutput.model_validate({"utterance": "hi", "mood": "sad"})


def test_blank_patient_utterance_is_rejected():
    with pytest.raises(ValidationError):
        PatientOutput(utterance=" ", disclosed="nothing")


def test_belief_extraction_schema_is_strict():
    schema = BeliefExtraction.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])


def test_understood_is_required_only_when_reading():
    read = nurse_output_type(("utterance", "end"), (), True).model_json_schema()["properties"]
    skip = nurse_output_type(("utterance", "end"), (), False).model_json_schema()["properties"]
    assert read["understood"]["type"] == "string" and skip["understood"]["type"] == "null"
    assert list(read)[:2] == ["understood", "action"]  # read before acting
