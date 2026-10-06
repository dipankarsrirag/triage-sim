"""
Structured outputs the LLMs must produce.

Every field is required (nullable where optional) and extra keys are forbidden, so the JSON schemas
work for constrained decoding in vLLM and for OpenRouter's strict `json_schema` response format.
"""

from functools import cache
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from triagesim.personas import CASE_STATES, levels

VITALS = ("temperature", "heartrate", "resprate", "o2sat", "sbp")

Vital = Literal["temperature", "heartrate", "resprate", "o2sat", "sbp"]
Action = Literal["utterance", "check_vital", "log_red_flag", "end"]


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NurseOutput(_Output):
    # read before acting: constrained decoding generates fields in this order
    understood: Optional[str] = Field(
        description="First step after the patient speaks: one sentence on what their last reply told you "
        "(that reply only); otherwise null"
    )
    action: Action = Field(description="Next action chosen by the nurse")
    vital: Optional[Vital] = Field(description="Vital sign to check if action is 'check_vital', else null")
    utterance: Optional[str] = Field(description="What the nurse says (with 'end', a closing line); null if action is 'log_red_flag'")
    red_flags: list[str] = Field(description="All red flags identified so far")
    # reasoning before the estimate: constrained decoding generates fields in this order
    explanation: str = Field(description="Clinical reasoning grounded in the triage algorithm")
    triage: Literal[1, 2, 3, 4, 5] = Field(description="Current triage estimate (ESI or ATS level)")
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence in the triage estimate")

    @model_validator(mode="after")
    def _payload_matches_action(self):
        if self.action == "utterance" and not (self.utterance or "").strip():
            raise ValueError("action 'utterance' needs a non-empty utterance")
        if self.action == "check_vital" and self.vital is None:
            raise ValueError("action 'check_vital' needs a vital")
        return self


@cache
def nurse_output_type(actions: tuple[str, ...], vitals: tuple[str, ...], reading: bool = False) -> type[NurseOutput]:
    """NurseOutput narrowed to the actions and vitals allowed at this step; `understood` is required
    when the nurse is reading a new patient reply (`reading`) and null otherwise.

    Under constrained decoding the model cannot pick anything else; on other backends validation
    rejects it and the request is retried.
    """
    fields = NurseOutput.model_fields
    return create_model(
        "NurseOutput",
        __base__=NurseOutput,
        understood=(str if reading else type(None), Field(description=fields["understood"].description)),
        action=(Literal[actions], Field(description=fields["action"].description)),
        vital=(
            Optional[Literal[vitals]] if vitals else type(None),
            Field(description=fields["vital"].description),
        ),
    )


class PatientOutput(_Output):
    utterance: str
    disclosed: str = Field(description="One sentence: the information you revealed in this reply")

    @field_validator("utterance")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("empty utterance")
        return v


# The nurse's triage record (its belief state), kept by the dialogue master from what the patient
# says; vital signs are added as the nurse takes them.
RECORD_FIELDS = {
    "chief_complaint": "the main problem, in the patient's words",
    "onset_and_course": "when and how it started, and how it has changed since",
    "pain": "pain score (0-10), location and character",
    "associated_symptoms": "other symptoms the patient has, and relevant ones they deny",
    "relevant_history": "past medical history, earlier episodes or emergency visits",
    "medications": "medicines the patient takes",
    "allergies": "allergies, or that there are none",
}


RECORD_UNKNOWN = "needs information from patient"

TriageRecord = create_model(
    "TriageRecord",
    __base__=_Output,
    __doc__=f'The full triage record: every field\'s current value, or "{RECORD_UNKNOWN}".',
    **{name: (str, Field(description=desc)) for name, desc in RECORD_FIELDS.items()},
)


class Reading(_Output):
    """The dialogue master's reading of one exchange (the nurse's question and the patient's reply),
    from its words alone: what the reply conveys, and the full triage record after it."""

    conveyed: str = Field(description="One sentence: the information the patient's reply conveys")
    record: TriageRecord


# ─────────────────────────────────────────
# Dialogue master and patient scripts
# ─────────────────────────────────────────


class InSitu(_Output):
    """The dialogue master's check that a case would have a triage conversation at all."""

    reasoning: str
    conversation: bool = Field(description="Whether the patient would talk with the triage nurse in situ")


class Verdict(_Output):
    """The dialogue master's check of one utterance."""

    critique: str = Field(description="Brief analysis; if a check fails, what must change")
    faithful: bool = Field(description="true if the utterance passes the faithfulness check")
    informative: bool = Field(description="true if the utterance passes the informativeness check")
    in_persona: bool = Field(description="true if the utterance passes the persona check")
    plausible: bool = Field(description="true if the utterance passes the plausibility check")
    no_names: bool = Field(description="true if the utterance uses no personal names")

    @property
    def passed(self) -> bool:
        return self.faithful and self.informative and self.in_persona and self.plausible and self.no_names


class NurseVerdict(Verdict):
    """The dialogue master's check of one nurse utterance, which must also ask for one thing only."""

    one_question: bool = Field(description="true if the utterance asks for one thing only")

    @property
    def passed(self) -> bool:
        return super().passed and self.one_question


class LineEdit(_Output):
    """The dialogue master's own version of a line that failed its checks twice."""

    utterance: str

    @field_validator("utterance")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("empty utterance")
        return v


class PatientLineEdit(LineEdit):
    disclosed: str = Field(description="One sentence: the information the edited line reveals")


class PatientScript(_Output):
    """Standardized-patient script: what this patient experienced, from their perspective."""

    # triage signs the dialogue master sets from the case, not from the persona
    cognitive_state: levels(CASE_STATES["cognitive_state"]) = Field(description=CASE_STATES["cognitive_state"].definition)
    intoxication: levels(CASE_STATES["intoxication"]) = Field(description=CASE_STATES["intoxication"].definition)
    story: str = Field(description="What happened, first person, 2-4 sentences in plain words")
    onset: str = Field(description="When and how it started, e.g. 'suddenly, about two hours ago'")
    duration: str = Field(description="How long it has lasted; constant or comes and goes")
    location: Optional[str] = Field(description="Where in the body, e.g. 'centre of the chest, into the left arm'")
    quality: Optional[str] = Field(description="What the main symptom feels like, e.g. 'tight, like a band'")
    severity: str = Field(description="How bad it feels and what it stops the patient doing, in their words")
    associated_symptoms: list[str]
    pertinent_negatives: list[str] = Field(
        description="Relevant symptoms the patient does NOT have; never one the complaint or diagnoses imply"
    )
    medical_history: list[str]
    medications: list[str]
    allergies: list[str]
    social_history: str
