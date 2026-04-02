import pytest

from pydantic import ValidationError

from triagesim.core.actions import (
    NurseAction,
    ConfidenceLevel,
    TriageAlgorithm,
    NurseUtterance,
    NurseCheckVital,
    NurseLogRedFlag,
    NurseEnd,
    PatientAction,
    make_nurse_utterance,
    make_nurse_check_vital,
    make_nurse_log_red_flag,
    make_nurse_end,
)


# ───────────────────────────────────────────────
# Enum Tests
# ───────────────────────────────────────────────


def test_nurse_action_enum_values():
    assert NurseAction.utterance.value == "utterance"
    assert NurseAction.check_vital.value == "check_vital"
    assert NurseAction.log_red_flag.value == "log_red_flag"
    assert NurseAction.end.value == "end"


def test_confidence_level_enum():
    assert set(item.value for item in ConfidenceLevel) == {"low", "medium", "high"}


def test_triage_algorithm_enum():
    assert set(item.value for item in TriageAlgorithm) == {"esi", "ats"}


# ───────────────────────────────────────────────
# NurseUtterance Validation
# ───────────────────────────────────────────────


def test_nurse_utterance_valid():
    action = NurseUtterance(
        utterance="How are you feeling?",
        triage=3,
        confidence=ConfidenceLevel.medium,
        explanation="Assessing initial pain severity",
        red_flags=["tachypnea"],
    )
    assert action.action == NurseAction.utterance
    assert action.triage == 3
    assert action.confidence == ConfidenceLevel.medium


def test_nurse_utterance_missing_required():
    with pytest.raises(ValidationError):
        NurseUtterance(
            # missing required fields
            utterance="Missing triage & confidence",
        )


def test_nurse_utterance_invalid_triage():
    with pytest.raises(ValidationError):
        NurseUtterance(
            utterance="Invalid triage",
            triage=7,  # out of range
            confidence=ConfidenceLevel.low,
            explanation="Bad triage",
        )


# ───────────────────────────────────────────────
# NurseCheckVital Validation
# ───────────────────────────────────────────────


def test_nurse_check_vital_valid():
    action = NurseCheckVital(utterance="Can I check heart rate?")
    assert action.action == NurseAction.check_vital
    assert "heart rate" in action.utterance.lower()


def test_nurse_check_vital_no_utterance():
    with pytest.raises(ValidationError):
        NurseCheckVital()


# ───────────────────────────────────────────────
# NurseLogRedFlag Tests
# ───────────────────────────────────────────────


def test_nurse_log_red_flag_valid():
    action = NurseLogRedFlag(red_flags=["hypoxia"])
    assert action.action == NurseAction.log_red_flag
    assert "hypoxia" in action.red_flags


def test_nurse_log_red_flag_empty_list():
    action = NurseLogRedFlag(red_flags=[])
    assert action.red_flags == []


# ───────────────────────────────────────────────
# NurseEnd Tests
# ───────────────────────────────────────────────


def test_nurse_end_with_utterance():
    action = NurseEnd(utterance="Thank you")
    assert action.action == NurseAction.end
    assert action.utterance == "Thank you"


def test_nurse_end_without_utterance():
    action = NurseEnd()
    assert action.action == NurseAction.end
    assert action.utterance is None


# ───────────────────────────────────────────────
# PatientAction Tests
# ───────────────────────────────────────────────


def test_patient_action_valid():
    action = PatientAction(utterance="I feel dizzy")
    assert action.utterance == "I feel dizzy"


def test_patient_action_missing():
    with pytest.raises(ValidationError):
        PatientAction()  # utterance required


# ───────────────────────────────────────────────
# Factory Helpers
# ───────────────────────────────────────────────


def test_make_nurse_utterance_helper():
    action = make_nurse_utterance(
        utterance="Hello",
        triage=2,
        confidence="high",
        explanation="Just testing",
        red_flags=["flag"],
    )
    assert isinstance(action, NurseUtterance)
    assert action.confidence == ConfidenceLevel.high


def test_make_nurse_check_vital_helper():
    action = make_nurse_check_vital("Check bp")
    assert isinstance(action, NurseCheckVital)


def test_make_nurse_log_red_flag_helper():
    action = make_nurse_log_red_flag(["flag1", "flag2"])
    assert isinstance(action, NurseLogRedFlag)
    assert "flag2" in action.red_flags


def test_make_nurse_end_helper():
    action = make_nurse_end("Bye")
    assert isinstance(action, NurseEnd)
    assert action.utterance == "Bye"
