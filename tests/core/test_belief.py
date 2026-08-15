import pytest

from triagesim.core.belief_graph import BeliefGraph, BeliefSlotUpdate
from triagesim.core.belief_updater import BeliefUpdater


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────


@pytest.fixture
def graph():
    return BeliefGraph()


@pytest.fixture
def updater():
    return BeliefUpdater()


# ─────────────────────────────────────────
# Tests: externally-produced updates
#
# The updater performs NO inference of its own; updates are produced
# upstream (e.g. by NurseAgent.infer_belief_updates) and applied here.
# ─────────────────────────────────────────


def test_no_updates_leaves_graph_untouched(graph, updater):
    """
    Applying an empty update list must not mutate the graph.
    """
    updater.apply_updates(graph=graph, updates=[])

    assert graph.slot_values == {}
    assert graph.evidence_nodes == {}
    assert graph.edges == {}
    assert graph.vitals_known == set()
    assert graph.red_flags_logged == set()


def test_apply_updates_records_slot_and_evidence(graph, updater):
    """
    An externally-produced update should land in the graph with provenance.
    """
    updater.apply_updates(
        graph=graph,
        updates=[
            BeliefSlotUpdate(
                slot="chief_complaint",
                value="chest pain",
                evidence="I have severe chest pain",
                source="patient",
                turn=0,
                certainty="explicit",
            )
        ],
    )

    assert graph.slot_values["chief_complaint"][0]["value"] == "chest pain"
    assert graph.slot_values["chief_complaint"][0]["certainty"] == "explicit"
    assert len(graph.evidence_nodes) == 1
    assert "chief_complaint" in graph.edges


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
