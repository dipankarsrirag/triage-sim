import pytest

from triagesim.core.belief_graph import BeliefSlotUpdate
from triagesim.core.environment import TriageEnv, NURSE, PATIENT
from triagesim.core.output_schema import NurseOutput, PatientOutput
from triagesim.core.state_store import StateStore


# ─────────────────────────────────────────
# In-memory mock StateStore
# ─────────────────────────────────────────


class InMemoryStateStore(StateStore):
    def __init__(self):
        self._store = {}

    def set_json(self, key, value):
        self._store[key] = value

    def get_json(self, key):
        return self._store.get(key)

    def delete(self, key):
        self._store.pop(key, None)

    def append_list(self, key, value):
        self._store.setdefault(key, []).append(value)

    def get_list(self, key):
        return self._store.get(key, [])


# ─────────────────────────────────────────
# Mock nurse agent
#
# The environment calls back into the nurse agent to infer belief updates
# from patient utterances. This mock does keyword matching instead of an LLM.
# ─────────────────────────────────────────


class MockNurseAgent:
    def __init__(self):
        self.seen_vital = False
        self.logged_flag = False

    def infer_belief_updates(self, history, last_utterance, turn):
        updates = []

        for keyword in ("breath", "dizzy", "pain"):
            if keyword in last_utterance.lower():
                updates.append(
                    BeliefSlotUpdate(
                        slot="associated_symptom",
                        value=keyword,
                        evidence=last_utterance,
                        source="patient",
                        turn=turn,
                        certainty="explicit",
                    )
                )

        return updates


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────


@pytest.fixture
def ground_truth():
    return {
        "vitals": {
            "heartrate": 120,
            "resprate": 28,
            "o2sat": 90,
            "sbp": 95,
            "temperature": 38.2,
        },
        "acuity": 2,
        "pain": 7,
    }


@pytest.fixture
def env():
    return TriageEnv(
        store=InMemoryStateStore(),
        nurse_agent=MockNurseAgent(),
        max_turns=5,
    )


@pytest.fixture
def run_id(env, ground_truth):
    return env.reset(ground_truth)


# ─────────────────────────────────────────
# Tests
# ─────────────────────────────────────────


def test_nurse_utterance_step(env, run_id):
    nurse_action = NurseOutput(
        action="utterance",
        utterance="Can you tell me where the pain is?",
        triage=3,
        red_flags=[],
        confidence="medium",
        explanation="Pain location not yet clear",
    )

    obs, done = env.step(run_id, NURSE, nurse_action)

    assert not done

    # Nurse actions are micro-turns: only the patient advances the turn counter.
    assert obs["state"]["turn"] == 0

    history = obs["history"]
    assert history[-1]["actor"] == "nurse"
    assert "pain" in history[-1]["utterance"].lower()

    belief = obs["belief"]
    assert belief is not None


def test_patient_response_advances_turn_and_updates_belief(env, run_id):
    # Nurse asks
    env.step(
        run_id,
        NURSE,
        NurseOutput(
            action="utterance",
            utterance="What symptoms are you having?",
            triage=3,
            red_flags=[],
            confidence="low",
            explanation="Need symptoms",
        ),
    )

    # Patient responds
    patient_action = PatientOutput(utterance="I feel short of breath and dizzy.")

    obs, done = env.step(run_id, PATIENT, patient_action)

    assert not done
    assert obs["state"]["turn"] == 1

    symptoms = obs["belief"]["associated_symptoms"]

    assert len(symptoms) > 0
    assert any("breath" in s["value"].lower() for s in symptoms)


def test_check_vital_releases_ground_truth(env, run_id):
    nurse_action = NurseOutput(
        action="check_vital",
        utterance="What is the oxygen saturation?",
        triage=2,
        red_flags=[],
        confidence="medium",
        explanation="Possible hypoxia",
    )

    obs, done = env.step(run_id, NURSE, nurse_action)

    assert not done

    history = obs["history"]
    last_event = history[-1]

    assert last_event["actor"] == "system"
    assert last_event["event"] == "vital"
    assert last_event["name"] == "o2sat"
    assert last_event["value"] == 90

    belief = obs["belief"]
    assert "o2sat" in belief["vitals_known"]


def test_log_red_flag(env, run_id):
    nurse_action = NurseOutput(
        action="log_red_flag",
        utterance=None,
        triage=2,
        red_flags=["severe respiratory distress"],
        confidence="high",
        explanation="Marked tachypnea and hypoxia",
    )

    obs, done = env.step(run_id, NURSE, nurse_action)

    assert not done
    assert "severe respiratory distress" in obs["red_flags"]

    belief = obs["belief"]
    assert any("respiratory" in rf["value"].lower() for rf in belief["red_flags"])


def test_duplicate_red_flags_are_logged_once(env, run_id):
    nurse_action = NurseOutput(
        action="log_red_flag",
        utterance=None,
        triage=2,
        red_flags=["severe respiratory distress"],
        confidence="high",
        explanation="Marked tachypnea and hypoxia",
    )

    env.step(run_id, NURSE, nurse_action)
    obs, _ = env.step(run_id, NURSE, nurse_action)

    assert obs["red_flags"].count("severe respiratory distress") == 1


def test_end_action_terminates_episode(env, run_id):
    nurse_action = NurseOutput(
        action="end",
        utterance=None,
        triage=2,
        red_flags=[],
        confidence="high",
        explanation="Sufficient information gathered",
    )

    obs, done = env.step(run_id, NURSE, nurse_action)

    assert done
    assert obs["state"]["done"] is True

    history = obs["history"]
    assert history[-1]["event"] == "triage_end"


def test_max_turns_termination(env, run_id):
    # Only patient steps advance the turn counter, so termination is
    # driven by patient responses rather than nurse micro-turns.
    for _ in range(env.max_turns):
        env.step(
            run_id,
            PATIENT,
            PatientOutput(utterance="I still feel unwell."),
        )

    obs = env.observe(run_id)
    assert obs["state"]["done"] is True


def test_step_after_done_is_a_noop(env, run_id):
    env.step(
        run_id,
        NURSE,
        NurseOutput(
            action="end",
            utterance=None,
            triage=2,
            red_flags=[],
            confidence="high",
            explanation="Done",
        ),
    )

    before = env.observe(run_id)["history"]

    obs, done = env.step(
        run_id,
        NURSE,
        NurseOutput(
            action="utterance",
            utterance="One more question?",
            triage=3,
            red_flags=[],
            confidence="low",
            explanation="Should be ignored",
        ),
    )

    assert done
    assert obs["history"] == before


def test_wrong_output_type_for_actor_raises(env, run_id):
    with pytest.raises(TypeError):
        env.step(run_id, NURSE, PatientOutput(utterance="I am the patient."))
