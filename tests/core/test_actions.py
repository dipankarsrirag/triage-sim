import pytest

from pydantic import ValidationError

from triagesim.core.actions import (
    NurseAction,
    NurseUtteranceAction,
    NurseCheckVitalAction,
    NurseLogRedFlagAction,
    NurseEndAction,
)


# ─────────────────────────────────────────
# Discriminator tags
# ─────────────────────────────────────────


def test_action_type_tags_are_fixed():
    assert NurseUtteranceAction(utterance="Hello", triage=3).type == "utterance"
    assert NurseCheckVitalAction(vital="o2sat").type == "check_vital"
    assert NurseLogRedFlagAction(red_flags=["shock"]).type == "log_red_flag"
    assert NurseEndAction(triage=2).type == "end"


def test_nurse_action_union_members():
    assert set(NurseAction.__args__) == {
        NurseUtteranceAction,
        NurseCheckVitalAction,
        NurseLogRedFlagAction,
        NurseEndAction,
    }


# ─────────────────────────────────────────
# Utterance action
# ─────────────────────────────────────────


def test_utterance_action_valid():
    action = NurseUtteranceAction(
        utterance="Where is the pain?",
        triage=3,
    )

    assert action.utterance == "Where is the pain?"
    assert action.triage == 3


def test_utterance_action_rejects_empty_utterance():
    with pytest.raises(ValidationError):
        NurseUtteranceAction(utterance="", triage=3)


@pytest.mark.parametrize("triage", [0, 6, -1])
def test_utterance_action_rejects_out_of_range_triage(triage):
    with pytest.raises(ValidationError):
        NurseUtteranceAction(utterance="Where is the pain?", triage=triage)


# ─────────────────────────────────────────
# Check-vital action
# ─────────────────────────────────────────


@pytest.mark.parametrize(
    "vital",
    ["temperature", "heartrate", "resprate", "o2sat", "sbp"],
)
def test_check_vital_accepts_allowed_vitals(vital):
    assert NurseCheckVitalAction(vital=vital).vital == vital


def test_check_vital_rejects_unknown_vital():
    with pytest.raises(ValidationError):
        NurseCheckVitalAction(vital="blood_glucose")


# ─────────────────────────────────────────
# Red-flag action
# ─────────────────────────────────────────


def test_log_red_flag_defaults_to_empty_list():
    assert NurseLogRedFlagAction().red_flags == []


def test_log_red_flag_keeps_provided_flags():
    action = NurseLogRedFlagAction(red_flags=["hypotension", "tachypnea"])
    assert action.red_flags == ["hypotension", "tachypnea"]


# ─────────────────────────────────────────
# End action
# ─────────────────────────────────────────


def test_end_action_requires_triage():
    with pytest.raises(ValidationError):
        NurseEndAction()


# ─────────────────────────────────────────
# Model config guarantees
# ─────────────────────────────────────────


def test_actions_forbid_extra_fields():
    with pytest.raises(ValidationError):
        NurseEndAction(triage=2, confidence="high")


def test_actions_validate_on_assignment():
    action = NurseEndAction(triage=2)

    with pytest.raises(ValidationError):
        action.triage = 9
