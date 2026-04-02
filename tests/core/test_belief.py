import pytest

from triagesim.core.belief_graph import BeliefGraph
from triagesim.core.belief_updater import BeliefUpdater


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────


@pytest.fixture
def graph():
    return BeliefGraph()


@pytest.fixture
def updater():
    # LLM disabled by default
    return BeliefUpdater()


# ─────────────────────────────────────────
# Tests: utterances (LLM disabled)
# ─────────────────────────────────────────


def test_patient_utterance_no_llm_no_change(graph, updater):
    """
    Patient utterance should NOT update belief when LLM is disabled.
    """
    updater.update_from_utterance(
        graph=graph,
        text="I have severe chest pain",
        source="patient",
        turn=0,
    )

    assert graph.slot_values == {}
    assert graph.evidence_nodes == {}
    assert graph.edges == {}
    assert graph.vitals_known == set()
    assert graph.red_flags_logged == set()


def test_nurse_utterance_no_llm_no_change(graph, updater):
    """
    Nurse utterance should NOT update belief when LLM is disabled.
    """
    updater.update_from_utterance(
        graph=graph,
        text="What is the patient's blood pressure?",
        source="nurse",
        turn=1,
    )

    assert graph.slot_values == {}
    assert graph.evidence_nodes == {}
    assert graph.edges == {}


# ─────────────────────────────────────────
# Tests: vitals
# ─────────────────────────────────────────


def test_update_from_vital_logs_vital(graph, updater):
    """
    Vital release should be recorded deterministically.
    """
    updater.update_from_vital(
        graph=graph,
        vital_name="o2sat",
        turn=2,
    )

    assert graph.vitals_known == {"o2sat"}
    assert graph.slot_values == {}
    assert graph.evidence_nodes == {}


# ─────────────────────────────────────────
# Tests: red flags
# ─────────────────────────────────────────


def test_update_from_red_flags_logs_flags(graph, updater):
    """
    Logged red flags should be stored explicitly.
    """
    updater.update_from_red_flags(
        graph=graph,
        red_flags=["severe respiratory distress"],
        turn=3,
    )

    assert graph.red_flags_logged == {"severe respiratory distress"}

    assert "red_flag" in graph.slot_values
    assert graph.slot_values["red_flag"][0]["value"] == "severe respiratory distress"
    assert graph.slot_values["red_flag"][0]["certainty"] == "explicit"


def test_duplicate_red_flags_are_not_duplicated(graph, updater):
    """
    Duplicate red flags should not create duplicate entries.
    """
    updater.update_from_red_flags(
        graph=graph,
        red_flags=["hypotension"],
        turn=1,
    )
    updater.update_from_red_flags(
        graph=graph,
        red_flags=["hypotension"],
        turn=2,
    )

    assert graph.red_flags_logged == {"hypotension"}
    assert len(graph.slot_values["red_flag"]) == 1


# ─────────────────────────────────────────
# Tests: serialization
# ─────────────────────────────────────────


def test_belief_graph_roundtrip_serialization(graph, updater):
    """
    BeliefGraph should serialize and deserialize losslessly.
    """
    updater.update_from_vital(graph, "heartrate", turn=0)
    updater.update_from_red_flags(graph, ["shock"], turn=1)

    data = graph.to_dict()
    restored = BeliefGraph.from_dict(data)

    assert restored.vitals_known == {"heartrate"}
    assert restored.red_flags_logged == {"shock"}
    assert restored.slot_values == graph.slot_values
