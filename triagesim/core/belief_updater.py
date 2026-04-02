"""
Belief updater for the triage environment.

This module:
- Applies belief updates that are produced externally (e.g., by NurseAgent)
- Mutates the environment-owned BeliefGraph
- Handles system events (vitals, explicit red-flag logging)

IMPORTANT:
- This module performs NO inference
- It is deterministic and environment-safe
"""

from typing import List

from triagesim.core.belief_graph import BeliefGraph, BeliefSlotUpdate


class BeliefUpdater:
    """
    Stateless belief graph mutator.

    Design principles:
    - No LLMs
    - No detectors
    - No clinical logic
    - Never throws
    """

    def __init__(self) -> None:
        pass

    # ─────────────────────────────────────────
    # Generic update application
    # ─────────────────────────────────────────

    def apply_updates(
        self,
        graph: BeliefGraph,
        updates: List[BeliefSlotUpdate],
    ) -> None:
        """
        Apply externally-produced belief updates to the graph.
        """
        if not updates:
            return
        graph.apply_updates(updates)

    # ─────────────────────────────────────────
    # System-originated events
    # ─────────────────────────────────────────

    def update_from_vital(
        self,
        graph: BeliefGraph,
        vital_name: str,
        turn: int,
    ) -> None:
        """
        Record that a vital sign has been observed.
        """
        graph.log_vital(vital_name)

    def update_from_red_flags(
        self,
        graph: BeliefGraph,
        red_flags: List[str],
        turn: int,
    ) -> None:
        """
        Record nurse-logged red flags as explicit belief entries.
        Assumes red_flags are already de-duplicated for this turn.
        """
        if not red_flags:
            return

        graph.log_red_flags(red_flags)

        updates = [
            BeliefSlotUpdate(
                slot="red_flag",
                value=flag,
                evidence=None,
                source="nurse",
                turn=turn,
                certainty="explicit",
            )
            for flag in red_flags
        ]

        graph.apply_updates(updates)
