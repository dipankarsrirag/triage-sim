"""
Personas: combinations of literature-grounded traits that shape how the nurse and patient talk.

Each trait has a definition, levels anchored in observable behaviour, and its sources (full
references and the evidence in docs/persona-traits-review.md). The registries below are the single
source of truth: the persona models, the prompts, the dialogue master's checks and the combinatorial
generation (scripts/generate_personas.py) all read them. Personas are every combination of levels
except those containing an INCOMPATIBLE pair.

Nurse traits shape only the interview (which questions are asked and how); they must not change the
triage level the nurse assigns. Clinical states that are triage signs (cognitive state,
intoxication) are not persona traits: the dialogue master sets them from the case, in the script.
"""

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, create_model


@dataclass(frozen=True)
class Trait:
    definition: str
    levels: dict[str, str]  # level -> behavioural anchor shown to the models
    sources: str
    per_line: bool = True  # observable in a single utterance (else only across an episode)


PATIENT_TRAITS: dict[str, Trait] = {
    "age_group": Trait(
        "The patient's age band (sampled until the cases carry age).",
        {"adult": "an adult under 65", "older": "65 or older"},
        "AIHW 2026",
        per_line=False,
    ),
    "english_proficiency": Trait(
        "Ability to communicate in English, the language of the triage interview.",
        {
            "fluent": "speaks English fluently",
            "functional": "uses simplified grammar and vocabulary and occasionally misunderstands a question",
            "limited": "speaks in fragments and often misunderstands; would normally need an interpreter",
        },
        "Flores 2005; Ramirez 2008; Mahmoud 2013; ABS 2022",
    ),
    "health_literacy": Trait(
        "Ability to understand and talk about health information.",
        {
            "limited": "uses only lay words, does not know the names of conditions or medicines, struggles with a "
            "0-10 pain scale, and asks what medical words mean",
            "marginal": "mostly uses lay words, knows some names of conditions or medicines only approximately, and "
            "needs help with a 0-10 scale",
            "adequate": "uses the names of conditions and medicines, uses a 0-10 scale readily, and understands "
            "common medical terms",
        },
        "ABS 2006; Chung 2002; Herndon 2011; Street 2005",
    ),
    "recall_reliability": Trait(
        "How reliably the patient recalls facts of their history: symptom timing, current medicines, past "
        "conditions (never by inventing new symptoms).",
        {
            "reliable": "gives clear timings and recalls all their current medicines (by name or description) and "
            "past conditions when asked",
            "patchy": "gives approximate timings ('sometime yesterday'), recalls major conditions and regular medicines "
            "but forgets some (e.g. over-the-counter or occasional ones), and is unsure of doses",
            "unreliable": "cannot say when symptoms began or in what order, recalls few of their medicines or past "
            "conditions, and gives a different time or detail when asked again",
        },
        "Mazer 2011; Caglar 2011; Kreshak 2015; Monte 2015; Iliceto 2016; Brandberg 2024; Chung 2002",
    ),
    "symptom_reporting": Trait(
        "How the patient's report of pain and symptoms relates to what they feel.",
        {
            "minimising": "reports pain about 2 points lower than felt and plays symptoms down ('just a bit sore'), "
            "offering harmless explanations",
            "faithful": "reports pain and symptoms as felt",
            "amplifying": "reports pain about 2 points higher than felt (at most 10) and describes symptoms in "
            "catastrophic terms",
        },
        "Barsky 1988; Sullivan 2001; Yong 2006; Puntillo 2003; Marquié 2003; Capponi 2016",
    ),
    "distress": Trait(
        "Emotional distress shown during the interview.",
        {
            "calm": "is composed and rarely voices worry",
            "anxious": "voices worries and asks for reassurance, and settles when reassured",
            "highly_distressed": "repeatedly expresses fear or upset and seeks reassurance, and settles only partly "
            "when reassured",
        },
        "Fleet 1996; Fulbrook 2015; Zimmermann 2011",
    ),
    "disclosure": Trait(
        "Willingness to disclose sensitive topics (alcohol and drugs, mental health and self-harm, violence at "
        "home, not taking medicines); other history is unaffected.",
        {
            "open": "answers questions on sensitive topics truthfully when asked, but does not raise them unprompted",
            "selective": "denies or plays down a sensitive topic when asked briefly or with a closed question (e.g. "
            "admits drinking but not drug use), and discloses it if the nurse follows up, asks openly, picks up a hint "
            "or reassures about privacy",
            "guarded": "denies or deflects sensitive topics even when the nurse follows up ('why do you need to know "
            "that?'), out of fear of judgement, the police or being overheard, and answers other questions normally",
        },
        "Rockett 2006; Chen 2006; Cherpitel 2007; Rhodes 2006, 2007; Claassen 2005; Boudreaux 2016; Hankin 2015; "
        "Karro 2005; Kimberg 2021; Steinhauser 2024",
    ),
    "demeanour": Trait(
        "The patient's stance towards the nurse.",
        {
            "cooperative": "is polite and cooperative",
            "impatient": "presses to be seen sooner and shows irritation at waiting",
            "hostile": "is verbally confrontational towards the nurse",
        },
        "Groves 1978; Hahn 1996; Kyung 2025; ACEM 2023a",
    ),
    "verbosity": Trait(
        "How much the patient says per answer, within short spoken turns (open questions draw longer answers).",
        {
            "terse": "answers in a few words",
            "typical": "answers in one or two short sentences",
            "expansive": "answers in up to three sentences, adding detail",
        },
        "Heritage 2006a",
    ),
    "focus": Trait(
        "How closely answers stay on the question.",
        {
            "on_target": "stays on the question",
            "occasional_tangents": "sometimes drifts into unrelated detail",
            "frequent_tangents": "often drifts into side stories before answering",
        },
        "Arbuckle 1993; Wang 2024",
    ),
    "disfluency": Trait(
        "Speech disfluency: fillers (um, uh) and self-repairs.",
        {
            "low": "rarely uses fillers or self-corrections",
            "moderate": "uses some fillers and self-corrections",
            "high": "uses frequent fillers, false starts and self-repairs",
        },
        "Bortfeld 2001",
    ),
}

NURSE_TRAITS: dict[str, Trait] = {
    "expertise": Trait(
        "Triage experience, shown in how the nurse interviews (experience does not predict triage accuracy).",
        {
            "novice": "asks about each history topic in turn rather than prioritising, and asks for vital signs early",
            "competent": "asks structured questions focused on the presenting problem",
            "expert": "asks fewer, targeted, hypothesis-driven questions and screens for red flags early",
        },
        "Cioffi 1998; Considine 2007; Gorick 2026",
    ),
    "algorithm_reliance": Trait(
        "How closely the interview follows the structure of the triage algorithm.",
        {
            "protocol_driven": "works through the algorithm's criteria in order and obtains vital signs before "
            "concluding",
            "blended": "mixes structured questions with following the patient's story",
            "intuition_led": "follows the patient's story and overall presentation rather than the algorithm's "
            "order",
        },
        "Gerdtz 2001; Cioffi 1998; Göransson 2008; Platts-Mills 2010; Grossmann 2012; Murdoch 2015",
        per_line=False,
    ),
    "risk_attitude": Trait(
        "How widely the nurse screens for danger under uncertainty.",
        {
            "risk_averse": "screens widely for red flags and dangerous causes before concluding",
            "balanced": "screens for the red flags relevant to the complaint",
            "risk_accepting": "focuses on the presenting problem with little extra screening",
        },
        "Pearson 1995; Arslanian-Engoren 2000; Considine 2004",
        per_line=False,
    ),
    "self_report_credence": Trait(
        "How the nurse treats the patient's own account of pain and severity.",
        {
            "accepting": "takes the stated pain and severity at face value",
            "corroborating": "checks reported severity against function, appearance and vital signs",
            "discounting": "probes reported severity for concrete examples and consistency",
        },
        "ENA 2023; Puntillo 2003; Marquié 2003; Capponi 2016; Grover 2012",
    ),
    "questioning_style": Trait(
        "The form of the nurse's questions.",
        {
            "open_facilitative": "opens with a general question, asks open questions, back-channels and "
            "summarises",
            "mixed": "combines open and closed questions",
            "closed_directive": "asks mostly yes/no and either/or questions",
        },
        "Heritage 2006a; Ernesäter 2014, 2016; Erkelens 2021; Lee 2015; Johri 2025",
    ),
    "emotional_responsiveness": Trait(
        "How the nurse responds when the patient voices worry or emotion.",
        {
            "task_focused": "moves past the patient's worries to the next question",
            "acknowledging": "briefly acknowledges worries before moving on",
            "empathic": "responds explicitly to worries and explains the process and the likely wait",
        },
        "Del Piccolo 2011; Ernesäter 2016; Street 2005; Mead 2000; Slade 2008; Janerka 2025",
    ),
    "workload": Trait(
        "How busy the triage desk is in this episode.",
        {
            "quiet": "has time for an unhurried interview",
            "busy": "keeps a brisk pace",
            "overloaded": "is rushed and moves on quickly",
        },
        "ACEM 2023a, 2023b; Gerdtz 2001; Hitchcock 2014; Reay 2020; Fekonja 2023; Gorick 2026",
        per_line=False,
    ),
}

# Triage signs the dialogue master sets from the case (in the script), never sampled
CASE_STATES: dict[str, Trait] = {
    "cognitive_state": Trait(
        "Mental status, as supported by the case.",
        {
            "alert": "is alert and oriented",
            "impaired": "is slow, repeats themselves, has lapses in orientation, and needs questions repeated",
            "delirious": "is inattentive and disorganised: drowsy, gives sparse answers and loses the question",
        },
        "Hustey 2002; Han 2009; Inouye 1990",
    ),
    "intoxication": Trait(
        "Alcohol or drug intoxication, as supported by the case.",
        {
            "none": "is sober",
            "mild": "is disinhibited but coherent",
            "marked": "is slurred, repetitive and inconsistent",
        },
        "Egerton-Warburton 2018; Klein 2018",
    ),
}


# Each anchor describes only its own trait, so traits combine freely. These pairs of levels still
# contradict each other by definition and are never combined (review of all trait pairs:
# docs/persona-pair-review.md).
INCOMPATIBLE: dict[frozenset, str] = {
    frozenset({("verbosity", "terse"), ("focus", "frequent_tangents")}):
        "A side story cannot fit in a few words: off-target talk is itself a form of verbosity (Arbuckle 1993).",
    frozenset({("expertise", "novice"), ("algorithm_reliance", "intuition_led")}):
        "Novices work by rules; intuition develops with experience (Benner 1982; Cioffi 1998).",
}


def levels(trait: Trait):
    return Literal[tuple(trait.levels)]


class _Persona(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # For later post-editing and TTS (accent, voice direction); never shown to the models.
    ethnicity: Optional[str] = None
    instruction: Optional[str] = None


PatientPersona = create_model(
    "PatientPersona",
    __base__=_Persona,
    gender=(Optional[str], Field(None, description="Set from the case")),
    **{name: (levels(t), Field(description=t.definition)) for name, t in PATIENT_TRAITS.items()},
)
NursePersona = create_model(
    "NursePersona",
    __base__=_Persona,
    gender=(Literal["female", "male"], Field(description="For the nurse's name and voice only")),
    **{name: (levels(t), Field(description=t.definition)) for name, t in NURSE_TRAITS.items()},
)

P = TypeVar("P", bound=_Persona)


def traits_of(persona: BaseModel) -> dict[str, Trait]:
    return PATIENT_TRAITS if isinstance(persona, PatientPersona) else NURSE_TRAITS


def combinations(traits: dict[str, Trait]) -> Iterator[dict[str, str]]:
    """Every combination of trait levels."""
    names = list(traits)
    for chosen in itertools.product(*(traits[n].levels for n in names)):
        yield dict(zip(names, chosen))


def compatible(combo: dict[str, str], incompatible=INCOMPATIBLE) -> bool:
    """Whether a combination avoids every incompatible pair of trait levels."""
    return not any(frozenset({(a, combo[a]), (b, combo[b])}) in incompatible for a, b in itertools.combinations(combo, 2))


def load_personas(path: str | Path, persona_type: type[P]) -> list[P]:
    """Load personas from a YAML list or a JSONL file (as written by scripts/generate_personas.py)."""
    path = Path(path)
    if path.suffix == ".jsonl":
        raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of personas")
    return [persona_type.model_validate(entry) for entry in raw]
