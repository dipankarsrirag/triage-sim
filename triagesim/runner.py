"""
Simulation runner for TriageSim.

The runner:
- Instantiates the environment and agents
- Drives nurse ↔ patient interaction with nurse micro-turns
- Persists all interaction state via StateStore
- Returns a complete, replayable run artifact

Design principles:
- Environment is the single source of truth
- Runner contains NO clinical logic
- LLM usage is controlled via a feature flag
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from triagesim.core.environment import TriageEnv, NURSE, PATIENT
from triagesim.core.state_store import (
    StateStore,
    RedisStateStore,
    InMemoryStateStore,
)
from triagesim.agents.nurse_agent import NurseAgent
from triagesim.agents.patient_agent import PatientAgent
from triagesim.core.output_schema import NurseOutput, PatientOutput


@dataclass
class RunnerConfig:
    """
    Configuration for a single simulation run.
    """

    max_turns: int = 6
    enable_llm: bool = False
    store_backend: str = "memory"  # "memory" | "redis"
    redis_db: int = 0
    seed: Optional[int] = None


class TriageRunner:
    """
    Orchestrates a single triage simulation episode.
    """

    def __init__(
        self,
        nurse_agent: NurseAgent,
        patient_agent: PatientAgent,
        ground_truth: Dict[str, Any],
        config: RunnerConfig,
    ):
        self.nurse_agent = nurse_agent
        self.patient_agent = patient_agent
        self.ground_truth = ground_truth
        self.config = config

        self.store = self._init_store()

        self.env = TriageEnv(
            nurse_agent=nurse_agent,
            store=self.store,
            max_turns=config.max_turns,
        )

        # Feature flag: enable or disable LLM-backed belief updates
        self.env.belief_updater.enable_llm = config.enable_llm

        if self.config.seed is not None:
            import random

            random.seed(self.config.seed)

    def _init_store(self) -> StateStore:
        if self.config.store_backend == "redis":
            return RedisStateStore(db=self.config.redis_db)
        elif self.config.store_backend == "memory":
            return InMemoryStateStore()
        else:
            raise ValueError(f"Unknown store backend: {self.config.store_backend}")

    def run(self) -> Dict[str, Any]:
        """
        Execute one full triage simulation.

        Nurse may act multiple times per turn (e.g. check vital → speak).
        Patient acts once per cycle.
        """

        run_id = self.env.reset(self.ground_truth)
        done = False

        # Hard safety cap (nurse + patient steps)
        max_steps = self.config.max_turns * 4
        steps = 0

        while not done and steps < max_steps:
            steps += 1

            # ─────────────────────────────────────────
            # Nurse phase (may contain multiple actions)
            # ─────────────────────────────────────────
            self.nurse_agent.seen_vital = False
            self.nurse_agent.logged_flag = False

            while True:
                obs = self.env.observe(run_id)
                belief = obs["belief"]
                known_vitals = set(belief.get("vitals_known", []))
                history_text = self._format_history(obs["history"])

                nurse_action: NurseOutput = self.nurse_agent.act(
                    history=history_text,
                    known_vitals=known_vitals,
                )

                obs, done = self.env.step(
                    run_id=run_id,
                    actor=NURSE,
                    action=nurse_action,
                )

                # print(obs)

                if done:
                    break

                # print(f"------- NURSE ACTION: {nurse_action.action}")
                # Enforce at most ONE vital request per nurse phase
                if nurse_action.action == "check_vital":
                    self.nurse_agent.seen_vital = True
                    # latest = obs["history"][-1]
                    # print(f"[VITAL | {latest['name']}]: {latest['value']}")
                    continue  # nurse gets to act again after system vital

                # Nurse utterance hands control to patient
                if nurse_action.action == "utterance":
                    # print("Nurse:", nurse_action.utterance)
                    break

                # Logging red flags does NOT end nurse phase
                if nurse_action.action == "log_red_flag":
                    self.nurse_agent.logged_flag = True
                    # latest = obs["trace"][-1]["action"]["red_flags"]
                    # print(f"[RED FLAGS]: {latest}")
                    continue

                # Explicit end
                if nurse_action.action == "end":
                    done = True
                    break

            if done:
                break

            # ─────────────────────────────────────────
            # Patient turn (exactly one)
            # ─────────────────────────────────────────
            obs = self.env.observe(run_id)
            history_text = self._format_history(obs["history"])

            # print(self.ground_truth)

            patient_action: PatientOutput = self.patient_agent.act(
                history=history_text,
                chief_complaint=self.ground_truth["chiefcomplaint"],
                pain=self.ground_truth["pain"],
            )

            # print("Patient:", patient_action.utterance)

            obs, done = self.env.step(
                run_id=run_id,
                actor=PATIENT,
                action=patient_action,
            )
        # print("------------------------------")
        # history_text = self._format_history(obs["history"])
        # print(history_text)
        # print("------------------------------")
        return self._finalize_run(run_id)

    def _format_history(self, history: list[dict]) -> str:
        """
        Convert environment history into a clean dialogue transcript
        suitable for LLM prompting.
        """
        lines: list[str] = []

        for h in history:
            actor = h.get("actor")

            if actor == "nurse":
                lines.append(f"Nurse: {h['utterance']}")

            elif actor == "patient":
                lines.append(f"Patient: {h['utterance']}")

            elif h.get("event") == "vital":
                lines.append(f"[Vital] {h['name']} = {h['value']}")

            elif h.get("event") == "triage_end":
                lines.append("[Triage ended]")

        formatted_history = "\n".join(lines)

        return formatted_history

    def _finalize_run(self, run_id: str) -> Dict[str, Any]:
        """
        Return a complete snapshot of the run.
        """

        obs = self.env.observe(run_id)

        return {
            "run_id": run_id,
            "ground_truth": self.ground_truth,
            "state": obs["state"],
            "history": obs["history"],
            "trace": obs["trace"],
            "belief": obs["belief"],
            "red_flags": obs["red_flags"],
        }
