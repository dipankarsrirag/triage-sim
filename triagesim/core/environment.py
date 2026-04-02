"""
The core simulation environment for triage conversations.

This environment:
- Orchestrates nurse ↔ patient turns
- Owns the belief graph (single source of truth)
- Applies executable nurse actions
- Releases vitals from ground truth
- Tracks red flags explicitly logged by the nurse
- Persists everything via StateStore (e.g. Redis)
"""

# triagesim/core/environment.py

import uuid
from typing import Any, Dict, Tuple, Union, List

from triagesim.core.state_store import StateStore
from triagesim.core.action_mapper import (
    map_nurse_output_to_action,
    NurseUtteranceAction,
    NurseCheckVitalAction,
    NurseLogRedFlagAction,
    NurseEndAction,
)
from triagesim.core.output_schema import NurseOutput, PatientOutput
from triagesim.core.belief_graph import BeliefGraph
from triagesim.core.belief_updater import BeliefUpdater

NURSE = "nurse"
PATIENT = "patient"
SYSTEM = "system"


class TriageEnv:
    def __init__(
        self,
        store: StateStore,
        nurse_agent,
        max_turns: int = 6,
    ):
        self.store = store
        self.max_turns = max_turns

        self.nurse_agent = nurse_agent
        self.belief_updater = BeliefUpdater()

    # ─────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────

    def reset(self, ground_truth: Dict[str, Any]) -> str:
        run_id = str(uuid.uuid4())

        self.store.set_json(
            f"triage:{run_id}:state",
            {
                "ground_truth": ground_truth,
                "turn": 0,
                "done": False,
            },
        )

        belief = BeliefGraph()
        self.store.set_json(f"triage:{run_id}:belief_graph", belief.to_dict())

        self.store.delete(f"triage:{run_id}:history")
        self.store.delete(f"triage:{run_id}:trace")
        self.store.delete(f"triage:{run_id}:red_flags")

        return run_id

    def observe(self, run_id: str) -> Dict[str, Any]:
        state = self.store.get_json(f"triage:{run_id}:state")
        history = self.store.get_list(f"triage:{run_id}:history")
        trace = self.store.get_list(f"triage:{run_id}:trace")
        red_flags = self.store.get_list(f"triage:{run_id}:red_flags")

        belief = BeliefGraph.from_dict(
            self.store.get_json(f"triage:{run_id}:belief_graph")
        )

        return {
            "state": state,
            "history": history,
            "trace": trace,
            "belief": belief.to_observation(),
            "red_flags": red_flags,
        }

    # ─────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────

    def step(
        self,
        run_id: str,
        actor: str,
        action: Union[NurseOutput, PatientOutput],
    ) -> Tuple[Dict[str, Any], bool]:
        state = self.store.get_json(f"triage:{run_id}:state")

        if state["done"]:
            return self.observe(run_id), True

        turn = state["turn"]
        belief = BeliefGraph.from_dict(
            self.store.get_json(f"triage:{run_id}:belief_graph")
        )

        done = False  # default

        # ─────────────────────────────────────────
        # Nurse action (micro-turn, does NOT advance turn)
        # ─────────────────────────────────────────
        if actor == NURSE:
            if not isinstance(action, NurseOutput):
                raise TypeError("Nurse must act with NurseOutput")

            executable = map_nurse_output_to_action(action)

            # Record nurse cognition trace
            self.store.append_list(
                f"triage:{run_id}:trace",
                {
                    "turn": turn,
                    "actor": NURSE,
                    "action": executable.model_dump(),
                    "triage": action.triage,
                    "confidence": action.confidence,
                    "explanation": action.explanation,
                    "red_flags_proposed": action.red_flags,
                },
            )

            done = self._apply_nurse_action(
                run_id=run_id,
                action=executable,
                belief=belief,
                turn=turn,
            )

        # ─────────────────────────────────────────
        # Patient action (ADVANCES turn)
        # ─────────────────────────────────────────
        elif actor == PATIENT:
            if not isinstance(action, PatientOutput):
                raise TypeError("Patient must act with PatientOutput")

            self.store.append_list(
                f"triage:{run_id}:history",
                {
                    "turn": turn,
                    "actor": PATIENT,
                    "utterance": action.utterance,
                },
            )

            # Nurse infers belief updates from patient utterance
            updates = self.nurse_agent.infer_belief_updates(
                history=self._format_history(run_id),
                last_utterance=action.utterance,
                turn=turn,
            )

            self.belief_updater.apply_updates(belief, updates)

            # ✅ ONLY HERE do we advance the turn
            state["turn"] += 1

        else:
            raise ValueError(f"Unknown actor: {actor}")

        # ─────────────────────────────────────────
        # Persist belief graph
        # ─────────────────────────────────────────
        self.store.set_json(
            f"triage:{run_id}:belief_graph",
            belief.to_dict(),
        )

        # ─────────────────────────────────────────
        # Termination logic (based on PATIENT turns)
        # ─────────────────────────────────────────
        if state["turn"] >= self.max_turns or done:
            state["done"] = True

        self.store.set_json(f"triage:{run_id}:state", state)

        return self.observe(run_id), state["done"]

    # ─────────────────────────────────────────
    # Nurse action execution
    # ─────────────────────────────────────────

    def _normalize_flag(self, s: str) -> str:
        words = s.lower().strip().split()
        
        return " ".join(s.lower().strip().split())

    def _apply_nurse_action(
        self,
        run_id: str,
        action,
        belief: BeliefGraph,
        turn: int,
    ) -> bool:

        if isinstance(action, NurseUtteranceAction):
            self.store.append_list(
                f"triage:{run_id}:history",
                {
                    "turn": turn,
                    "actor": NURSE,
                    "utterance": action.utterance,
                    "triage": action.triage,
                },
            )
            return False

        if isinstance(action, NurseCheckVitalAction):
            vitals = self.store.get_json(f"triage:{run_id}:state")["ground_truth"][
                "vitals"
            ]
            value = vitals.get(action.vital)

            self.store.append_list(
                f"triage:{run_id}:history",
                {
                    "turn": turn,
                    "actor": SYSTEM,
                    "event": "vital",
                    "name": action.vital,
                    "value": value,
                },
            )

            self.belief_updater.update_from_vital(
                graph=belief,
                vital_name=action.vital,
                turn=turn,
            )
            return False

        if isinstance(action, NurseLogRedFlagAction):
            key = f"triage:{run_id}:red_flags"

            # Existing logged flags (normalized)
            existing_raw = self.store.get_list(key) or []
            existing_norm = {self._normalize_flag(f) for f in existing_raw}

            # Per-turn deduplication
            seen_this_turn = set()
            accepted_flags = []

            MAX_FLAGS_PER_TURN = 5

            for flag in action.red_flags:
                if not flag:
                    continue

                norm = self._normalize_flag(flag)

                # Skip duplicates within the turn
                if norm in seen_this_turn:
                    continue

                # Skip exact repeats across previous turns
                if norm in existing_norm:
                    continue

                seen_this_turn.add(norm)
                accepted_flags.append(flag)

                if len(accepted_flags) >= MAX_FLAGS_PER_TURN:
                    break

            # Persist accepted flags
            for flag in accepted_flags:
                self.store.append_list(key, flag)

            # Update belief graph ONLY with accepted flags
            self.belief_updater.update_from_red_flags(
                graph=belief,
                red_flags=accepted_flags,
                turn=turn,
            )

            return False

        if isinstance(action, NurseEndAction):
            self.store.append_list(
                f"triage:{run_id}:history",
                {
                    "turn": turn,
                    "actor": SYSTEM,
                    "event": "triage_end",
                },
            )
            return True

        raise RuntimeError("Unhandled nurse action type")

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _format_history(self, run_id: str) -> str:
        """
        Render dialogue + system events as a single text transcript
        for nurse belief inference.
        """
        history = self.store.get_list(f"triage:{run_id}:history")
        lines: List[str] = []

        for h in history:
            if h["actor"] == NURSE:
                lines.append(f"Nurse: {h['utterance']}")
            elif h["actor"] == PATIENT:
                lines.append(f"Patient: {h['utterance']}")
            elif h.get("event") == "vital":
                lines.append(f"[Vital] {h['name']} = {h['value']}")
            elif h.get("event") == "triage_end":
                lines.append("[Triage ended]")

        return "\n".join(lines)
