"""
Environment-owned belief graph.

This module defines a belief graph that is:
- Incrementally updated per new event (utterance or vital)
- Provenance-aware: each slot value is linked to evidence and turn
- Deterministic in merge behavior (even if extraction is probabilistic)

The environment should store this graph in Redis and update it each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field


SourceType = Literal["patient", "nurse", "system_vital", "system_other"]
CertaintyType = Literal["explicit", "inferred", "suspected"]


# ─────────────────────────────────────────
# LLM-facing update schema
# ─────────────────────────────────────────


class BeliefSlotUpdate(BaseModel):
    """
    One incremental update proposed by a detector (LLM or heuristic).
    """

    slot: str = Field(
        ...,
        description="Target slot name, e.g. chief_complaint, pain_severity, risk_factors",
    )
    value: Optional[str] = Field(
        None, description="Slot value, or null if detector cannot extract"
    )
    evidence: Optional[str] = Field(
        None, description="Short quote or paraphrase supporting the update"
    )
    source: SourceType = Field(..., description="Where the evidence came from")
    turn: int = Field(..., ge=0, description="Turn index")
    certainty: CertaintyType = Field(..., description="explicit | inferred | suspected")


# ─────────────────────────────────────────
# Graph internal representation
# ─────────────────────────────────────────


@dataclass(frozen=True)
class EvidenceNode:
    evidence_id: str
    text: str
    source: SourceType
    turn: int


@dataclass
class BeliefGraph:
    """
    Stores belief as a provenance-aware graph.

    - evidence_nodes: evidence_id -> EvidenceNode
    - slot_values: slot -> list of belief items
    - edges: slot -> list of evidence_ids supporting that slot value
    """

    # Evidence
    evidence_nodes: Dict[str, EvidenceNode] = field(default_factory=dict)

    # Slots: store as lists to allow multiple hypotheses if needed
    slot_values: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Slot -> evidence ids (aligned with slot_values order)
    edges: Dict[str, List[str]] = field(default_factory=dict)

    # Known vitals and nurse-logged red flags are treated as special tracked sets
    vitals_known: Set[str] = field(default_factory=set)
    red_flags_logged: Set[str] = field(default_factory=set)

    # Deterministic counters for evidence ids
    _evidence_counter: int = 0

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def add_evidence(self, text: str, source: SourceType, turn: int) -> str:
        """
        Add an evidence node and return its id.
        """
        self._evidence_counter += 1
        evidence_id = f"e{self._evidence_counter:06d}"
        self.evidence_nodes[evidence_id] = EvidenceNode(
            evidence_id=evidence_id,
            text=text,
            source=source,
            turn=turn,
        )
        return evidence_id

    def apply_updates(self, updates: List[BeliefSlotUpdate]) -> None:
        """
        Deterministically merge a list of updates into the belief graph.

        Merge policy:
        - Ignore updates with value=None
        - De-duplicate exact (slot, value) pairs
        - Keep multiple values if they differ (allows ambiguity)
        - Always attach evidence if provided
        """
        for upd in updates:
            if upd.value is None:
                continue

            slot = upd.slot.strip()
            value = upd.value.strip()

            if not slot or not value:
                continue

            existing = self.slot_values.get(slot, [])
            if any(item.get("value") == value for item in existing):
                continue

            # Add evidence node if evidence is provided, else store without evidence link
            evidence_id = None
            if upd.evidence:
                evidence_id = self.add_evidence(upd.evidence, upd.source, upd.turn)

            item = {
                "value": value,
                "source": upd.source,
                "turn": upd.turn,
                "certainty": upd.certainty,
            }
            self.slot_values.setdefault(slot, []).append(item)
            self.edges.setdefault(slot, []).append(evidence_id or "")

    def log_vital(self, vital_name: str) -> None:
        self.vitals_known.add(vital_name)

    def log_red_flags(self, flags: List[str]) -> None:
        for f in flags:
            f2 = f.strip()
            if f2:
                self.red_flags_logged.add(f2)

    # ─────────────────────────────────────────
    # Serialization (Redis-friendly)
    # ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_nodes": {
                k: {
                    "evidence_id": v.evidence_id,
                    "text": v.text,
                    "source": v.source,
                    "turn": v.turn,
                }
                for k, v in self.evidence_nodes.items()
            },
            "slot_values": self.slot_values,
            "edges": self.edges,
            "vitals_known": sorted(self.vitals_known),
            "red_flags_logged": sorted(self.red_flags_logged),
            "_evidence_counter": self._evidence_counter,
        }

    def to_observation(self) -> Dict[str, Any]:
        """
        Flattened, consumer-friendly belief view for environment observation.
        This is intentionally lossy (no provenance graph).
        """

        obs: Dict[str, Any] = {}

        for slot, values in self.slot_values.items():
            obs_key = slot
            if slot == "associated_symptom":
                obs_key = "associated_symptoms"
            elif slot == "red_flag":
                obs_key = "red_flags"

            obs[obs_key] = values

        obs["vitals_known"] = sorted(self.vitals_known)
        obs["red_flags_logged"] = sorted(self.red_flags_logged)

        return obs

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BeliefGraph":
        g = cls()
        g._evidence_counter = int(d.get("_evidence_counter", 0))

        ev = d.get("evidence_nodes", {})
        for k, v in ev.items():
            g.evidence_nodes[k] = EvidenceNode(
                evidence_id=v["evidence_id"],
                text=v["text"],
                source=v["source"],
                turn=int(v["turn"]),
            )

        g.slot_values = d.get("slot_values", {}) or {}
        g.edges = d.get("edges", {}) or {}
        g.vitals_known = set(d.get("vitals_known", []) or [])
        g.red_flags_logged = set(d.get("red_flags_logged", []) or [])
        return g
