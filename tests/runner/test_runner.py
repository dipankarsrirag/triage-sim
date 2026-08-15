import pytest
from typing import List

from triagesim.core.belief_graph import BeliefSlotUpdate
from triagesim.runner import TriageRunner, RunnerConfig
from triagesim.core.output_schema import NurseOutput, PatientOutput


# ─────────────────────────────────────────
# Mock agents
# ─────────────────────────────────────────


class MockNurseAgent:
    """
    Deterministic nurse agent that emits a fixed sequence of actions.
    """

    def __init__(self, actions: List[NurseOutput]):
        self.actions = actions
        self.idx = 0

        # Per-phase flags the runner sets/reads
        self.seen_vital = False
        self.logged_flag = False

    def act(self, history: str, known_vitals: set) -> NurseOutput:
        if self.idx >= len(self.actions):
            # Default safe termination
            return NurseOutput(
                action="end",
                utterance=None,
                triage=3,
                red_flags=[],
                confidence="high",
                explanation="Terminating",
            )
        out = self.actions[self.idx]
        self.idx += 1
        return out

    def infer_belief_updates(
        self,
        history: str,
        last_utterance: str,
        turn: int,
    ) -> List[BeliefSlotUpdate]:
        """Deterministic stand-in for LLM-backed belief inference."""
        if "breath" not in last_utterance.lower():
            return []

        return [
            BeliefSlotUpdate(
                slot="associated_symptom",
                value="shortness of breath",
                evidence=last_utterance,
                source="patient",
                turn=turn,
                certainty="explicit",
            )
        ]


class MockPatientAgent:
    """
    Deterministic patient agent that emits a fixed sequence of utterances.
    """

    def __init__(self, utterances: List[str]):
        self.utterances = utterances
        self.idx = 0

    def act(self, history: str, chief_complaint: str, pain: int) -> PatientOutput:
        if self.idx >= len(self.utterances):
            return PatientOutput(utterance="I have nothing else to add.")
        u = self.utterances[self.idx]
        self.idx += 1
        return PatientOutput(utterance=u)


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────


@pytest.fixture
def ground_truth():
    return {
        "vitals": {
            "heartrate": 110,
            "resprate": 24,
            "o2sat": 92,
            "sbp": 100,
            "temperature": 37.8,
        },
        "chiefcomplaint": "shortness of breath",
        "acuity": 2,
        "pain": 6,
    }


@pytest.fixture
def runner(ground_truth):
    nurse_actions = [
        NurseOutput(
            action="utterance",
            utterance="What brings you in today?",
            triage=3,
            red_flags=[],
            confidence="low",
            explanation="Initial assessment",
        ),
        NurseOutput(
            action="check_vital",
            utterance="What is the oxygen saturation?",
            triage=2,
            red_flags=[],
            confidence="medium",
            explanation="Possible respiratory issue",
        ),
        NurseOutput(
            action="end",
            utterance=None,
            triage=2,
            red_flags=[],
            confidence="high",
            explanation="Sufficient information gathered",
        ),
    ]

    patient_utterances = [
        "I feel short of breath.",
        "It started this morning.",
    ]

    nurse = MockNurseAgent(nurse_actions)
    patient = MockPatientAgent(patient_utterances)

    config = RunnerConfig(
        max_turns=5,
        enable_llm=False,
        store_backend="memory",
        seed=42,
    )

    return TriageRunner(
        nurse_agent=nurse,
        patient_agent=patient,
        ground_truth=ground_truth,
        config=config,
    )


# ─────────────────────────────────────────
# Tests
# ─────────────────────────────────────────


def test_runner_executes_full_episode(runner):
    run = runner.run()

    assert "run_id" in run
    assert run["state"]["done"] is True


def test_runner_history_length(runner):
    run = runner.run()
    history = run["history"]

    # Expected:
    # Nurse utterance
    # Patient response
    # System vital
    # Nurse end
    assert len(history) >= 4

    actors = [h.get("actor") for h in history]
    assert "nurse" in actors
    assert "patient" in actors
    assert "system" in actors


def test_runner_trace_length(runner):
    run = runner.run()
    trace = run["trace"]

    # Trace should log ONLY nurse actions
    assert len(trace) == 3

    for step in trace:
        assert step["actor"] == "nurse"
        assert "action" in step


def test_runner_belief_exists(runner):
    run = runner.run()
    belief = run["belief"]

    assert isinstance(belief, dict)
    assert "vitals_known" in belief
    assert isinstance(belief["vitals_known"], list)


def test_runner_red_flags_structure(runner):
    run = runner.run()

    assert "red_flags" in run
    assert isinstance(run["red_flags"], list)
