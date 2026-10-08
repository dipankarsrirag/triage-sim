import json
from collections import Counter

import pytest

from triagesim.cases import Case
from triagesim.personas import NursePersona, PatientPersona
from triagesim.schemas import RECORD_FIELDS, RECORD_UNKNOWN, Appearance, InSitu, LineEdit, NurseOutput, PatientOutput, PatientScript, Reading, Verdict


def allowed(request, field):
    """Enum values the request's schema allows for a NurseOutput field."""
    prop = request.output_type.model_json_schema()["properties"][field]
    options = prop.get("anyOf", [prop])
    return [v for o in options for v in o.get("enum", [o["const"]] if "const" in o else [])]


def reading(request):
    """Whether the nurse must say what it understood from a new patient reply at this step."""
    return request.output_type.model_json_schema()["properties"]["understood"].get("type") == "string"


def nurse_says(request, **fields):
    """A complete nurse output: defaults, `understood` filled in only when the step requires it."""
    return {"understood": "The patient fainted." if reading(request) else None, "vital": None, "utterance": None,
            "red_flags": [], "explanation": "x", "triage": 3, "confidence": "medium", **fields}


def nurse_policy(request):
    """Check heart rate, log a red flag, ask a question, end after two patient replies."""
    actions, vitals = allowed(request, "action"), allowed(request, "vital")
    dialogue = request.messages[1]["content"]
    base = nurse_says(request)
    if "check_vital" in actions and "heartrate" in vitals:
        return {**base, "action": "check_vital", "vital": "heartrate", "utterance": "Let me check your pulse."}
    if "log_red_flag" in actions and "[Vital] heartrate" in dialogue:
        return {**base, "action": "log_red_flag", "red_flags": ["Tachycardia", " tachycardia "]}
    if dialogue.count("Patient:") >= 2 and "end" in actions:
        return {**base, "action": "end", "triage": 2}
    return {**base, "action": "utterance", "utterance": "Hello. What brings you in today?"}


def patient_policy(request):
    return {"utterance": "I fainted this morning.", "disclosed": "I fainted this morning."}


def reader_policy(request):
    """The dialogue master's reading of an exchange: what it conveyed and what it adds to the record."""
    record = dict.fromkeys(RECORD_FIELDS, RECORD_UNKNOWN) | {"chief_complaint": "fainted this morning"}
    return {"conveyed": "The patient fainted this morning.", "record": record}


SCRIPT = {
    "cognitive_state": "alert", "intoxication": "none",
    "story": "I fainted at the bus stop.", "onset": "this morning", "duration": "a few seconds", "location": None,
    "quality": None, "severity": "shaken", "associated_symptoms": ["palpitations"], "pertinent_negatives": ["chest pain"],
    "medical_history": [], "medications": [], "allergies": [], "social_history": "lives alone",
}


APPEARANCE = {
    "arrival": "walked in slowly, steadied by a friend", "breathing": "breathing fast, short sentences",
    "skin": "pale and sweaty", "behaviour": "alert, answers straight away", "visible_distress": "grimacing",
    "visible_injuries": None,
}


def verdict(request, critique="fine", **checks):
    """The dialogue master's answer for the requested verdict type: every check passes unless given."""
    return {"critique": critique, **{name: checks.get(name, True) for name in request.output_type.model_fields if name != "critique"}}


def judge_policy(request):
    return verdict(request)


def editor_policy(request):
    """The dialogue master's own version of a line that failed twice."""
    edit = {"utterance": "Edited by the master."}
    return edit | {"disclosed": "x"} if "disclosed" in request.output_type.model_fields else edit


class FakeBackend:
    """Backend that answers with policies keyed on the requested output type."""

    batched = True  # driven in rounds, like in-process vLLM

    def __init__(self, model="fake", nurse=nurse_policy, patient=patient_policy,
                 judge=judge_policy, reader=reader_policy, editor=editor_policy,
                 in_situ=lambda r: {"reasoning": "walk-in, talking", "conversation": True}, script=lambda r: SCRIPT,
                 appearance=lambda r: APPEARANCE):
        self.model = model
        self.policies = {
            NurseOutput: nurse, PatientOutput: patient,
            Verdict: judge, Reading: reader, InSitu: in_situ, PatientScript: script, LineEdit: editor,
            Appearance: appearance,
        }
        self.batches = []
        self.usage = Counter()

    def complete(self, requests):
        self.batches.append(requests)
        out = []
        for r in requests:
            policy = next(p for t, p in self.policies.items() if issubclass(r.output_type, t))
            answer = policy(r)
            out.append(answer if isinstance(answer, (str, Exception)) else json.dumps(answer))
        return out


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def case():
    return Case(
        case_id="c1",
        chief_complaint="Syncope",
        vitals={"temperature": 99.1, "heartrate": 112, "resprate": 26, "o2sat": 91, "sbp": 98},
        acuity=2,
        pain="7",
    )


@pytest.fixture
def nurse_persona():
    return NursePersona(
        gender="female",
        ethnicity="Australian",
        expertise="expert",
        algorithm_reliance="blended",
        risk_attitude="balanced",
        self_report_credence="corroborating",
        questioning_style="mixed",
        emotional_responsiveness="acknowledging",
        workload="busy",
    )


@pytest.fixture
def patient_persona():
    return PatientPersona(
        gender="male",
        ethnicity="Australian",
        age_group="adult",
        english_proficiency="fluent",
        health_literacy="adequate",
        recall_reliability="reliable",
        symptom_reporting="faithful",
        distress="calm",
        disclosure="open",
        demeanour="cooperative",
        verbosity="terse",
        focus="on_target",
        disfluency="low",
    )
