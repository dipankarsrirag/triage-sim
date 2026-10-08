"""
Prompt construction for the nurse, the patient and the dialogue master.

Static instructions (algorithm, persona, output format) go in the system message and the per-turn
state (transcript, known vitals, allowed actions) in the user message, so every request of an
episode shares a long cacheable prefix (vLLM prefix caching).
"""

import re
from typing import Optional

from pydantic import BaseModel

from triagesim.cases import Case
from triagesim.personas import CASE_STATES, PatientPersona, Trait, traits_of
from triagesim.schemas import RECORD_FIELDS, RECORD_UNKNOWN, VITALS, Appearance, PatientScript

RULE = "────────────────────────────────────────"

UNITS = {"heartrate": " bpm", "resprate": " breaths/min", "o2sat": "%", "sbp": " mmHg"}

SEVERITY = {
    1: "You are critically unwell: you may be struggling to breathe, barely able to talk, or close to collapse.",
    2: "You are seriously unwell: you feel very sick or are in severe pain or distress, and could get worse quickly.",
    3: "You are moderately unwell: you clearly need to be seen and treated today, but you are stable for now.",
    4: "You have a minor but real problem that needs a simple test or treatment.",
    5: "You have a minor problem and are not very unwell.",
}


def _section(title: str, body: str) -> str:
    return f"{RULE}\n{title}\n{RULE}\n{body}"


def _level(name: str, trait: Trait, level: str) -> str:
    return f"- {name.replace('_', ' ')}: {level.replace('_', ' ')} ({trait.levels[level]})"


def _persona(persona: BaseModel, per_line_only: bool = False) -> str:
    """Gender and each trait with its behavioural anchor; ethnicity and the TTS instruction are never
    shown. `per_line_only` keeps the traits observable in a single utterance (for the line checks)."""
    lines = [f"- gender: {persona.gender}"] if persona.gender else []
    for name, trait in traits_of(persona).items():
        if trait.per_line or not per_line_only:
            lines.append(_level(name, trait, getattr(persona, name)))
    return "\n".join(lines)


def _vital(name: str, v: Optional[float]) -> str:
    if v is None:
        return "not recorded"
    unit = (" °F" if v > 50 else " °C") if name == "temperature" else UNITS.get(name, "")
    return f"{v:g}{unit}"


# what to ask the patient about to infer a vital sign that cannot be measured
VITAL_SYMPTOMS = {
    "temperature": "fever, chills or feeling hot or cold",
    "heartrate": "a racing or pounding heart, palpitations or dizziness",
    "resprate": "shortness of breath or breathing fast",
    "o2sat": "breathlessness, trouble breathing or blue lips",
    "sbp": "dizziness, light-headedness, fainting or weakness",
}


def _record(record: Optional[dict], history: list[dict]) -> str:
    """The nurse's triage record: what the dialogue master noted from the dialogue, the vital signs
    taken so far, and those that could not be measured."""
    lines = [f"- {name.replace('_', ' ')}: {(record or {}).get(name) or RECORD_UNKNOWN}" for name in RECORD_FIELDS]
    tried = {h["name"]: h for h in history if h.get("event") == "vital"}
    taken = [f"{name} {_vital(name, tried[name]['value'])}" for name in VITALS
             if name in tried and tried[name].get("measurable", True)]
    missing = [name for name in VITALS if name not in tried]
    lines.append(f"- vital signs: {'; '.join(taken) or 'none taken yet'}"
                 + (f" (not yet taken: {', '.join(missing)})" if missing else ""))
    lines += [f"- {name}: could not be measured; infer it from the patient's answers (ask about {VITAL_SYMPTOMS[name]})"
              for name in VITALS if name in tried and not tried[name].get("measurable", True)]
    return "\n".join(lines)


def patient_view(case: Case, persona: PatientPersona, appearance: Optional[Appearance] = None) -> str:
    """What the nurse can see at the triage desk: who the patient is, how they arrived and, if the
    dialogue master described it, how they look."""
    lines = [f"- gender: {case.gender or persona.gender or 'not recorded'}", f"- age group: {persona.age_group}"]
    transport = str((case.model_extra or {}).get("arrival_transport") or "").lower()
    if transport and transport not in ("unknown", "other"):
        lines.append(f"- arrived by: {transport}")
    if appearance is not None:
        lines += [f"- {name.replace('_', ' ')}: {value}" for name, value in appearance.model_dump().items() if value]
    return "\n".join(lines)


def transcript(history: list[dict]) -> str:
    """Render dialogue and system events as the text transcript both agents see."""
    lines = []
    for h in history:
        if h["actor"] == "nurse":
            lines.append(f"Nurse: {h['utterance']}")
        elif h["actor"] == "patient":
            lines.append(f"Patient: {h['utterance']}")
        elif h["event"] == "vital" and h.get("measurable") is False:
            lines.append(f"[Vital] {h['name']}: could not be measured")
        elif h["event"] == "vital":
            lines.append(f"[Vital] {h['name']} = {_vital(h['name'], h['value'])}")
        elif h["event"] == "triage_end":
            lines.append("[Triage ended]")
    return "\n".join(lines)


def spoken_transcript(history: list[dict]) -> list[dict[str, str]]:
    """The output transcript: the spoken nurse and patient lines only, without system events and
    without lines kept for the agents alone (e.g. announcing a vital that could not be measured)."""
    return [
        {"speaker": h["actor"], "text": h["utterance"]}
        for h in history
        if h["actor"] in ("nurse", "patient") and h.get("in_transcript", True)
    ]


# ─────────────────────────────────────────
# Nurse
# ─────────────────────────────────────────

ESI_PROMPT = """
You are an emergency department triage nurse.

You must perform triage strictly according to the Emergency Severity Index (ESI).

You do NOT know the patient's full medical record, diagnosis, or vitals unless
they are explicitly obtained during the dialogue or requested.

At EVERY turn, you must:
1. Choose your next action
2. Assign an ESI triage level (1-5)
3. State your confidence (low, medium, high)
4. Provide a brief clinical explanation grounded in the ESI algorithm

────────────────────────────────────────
ESI TRIAGE RULES (FOLLOW EXACTLY)
────────────────────────────────────────

Step A: Immediate life-saving intervention
Assign ESI level 1 if the patient requires immediate life-saving intervention,
such as airway support, severe respiratory distress, shock, unresponsiveness,
or active seizure.

If YES -> ESI = 1.

If NO -> proceed to Step B.

Step B: High-risk situation, severe pain, or altered mental status
Assign ESI level 2 if ANY of the following are present:
- High-risk situation with potential for rapid deterioration
- Confused, lethargic, or disoriented mental status
- Severe pain or distress

If YES -> ESI = 2.

If NO -> proceed to Step C.

Step C: Predicted resource needs
Estimate how many DIFFERENT types of ED resources are required.

Resources include:

- Labs (1 resource): Blood and Urine
- Scanning (1 resource each): X-Ray, ECG, CT, MRI, Ultrasound, Angiography
- Intravenous fluids (hydration) (1 resource)
- Intravenous, intramuscular or nebulized medications (1 resource each)
- Specialized consultation (1 resource)
- Simple procedure (1 resource): Laceration repair, Urinary catheter etc.
- Complex procedure (2 resources): Procedures that need sedation or anesthesia
- Prescription refills, simple wound care and salines are not considered as resources

Based on this definition:
- No resources -> ESI = 5
- One resource -> ESI = 4
- Two or more resources -> proceed to Step D

Step D: High-risk vital signs
Assign ESI = 3. But if abnormal vital signs are present for age, consider upgrading to ESI = 2.

High-risk vital signs are:
- For less than 1 month: Heart Rate > 190; Resp. Rate > 60; SpO2 < 92
- For children: Heart Rate > 120; Resp. Rate > 30; SpO2 < 92
- For adults: Heart Rate > 100; Resp. Rate > 20; SpO2 < 92
""".strip()

ATS_PROMPT = """
You are an emergency department triage nurse.

You must perform triage strictly according to the Australasian Triage Scale (ATS).

Your task is to assess the urgency of the patient's condition and assign a triage
category from 1 to 5 based on clinical urgency and the presence of red flags.

You do NOT know the patient's diagnosis or full medical record.
You may only use information explicitly obtained through the dialogue so far or
through vital signs you request.

At EVERY turn, you must:
1. Choose your next action
2. Assign an ATS triage category (1-5)
3. State your confidence (low, medium, high)
4. Provide a brief clinical explanation grounded in ATS principles

────────────────────────────────────────
ATS TRIAGE CATEGORIES
────────────────────────────────────────

Category 1 - Immediate
Immediate life-threatening condition requiring immediate care.
Maximum waiting time: Immediate.

Examples include:
- Cardiac arrest
- Respiratory arrest or extreme respiratory distress
- Severe hypotension or shock
- Unresponsiveness or GCS < 9
- Ongoing or prolonged seizure

Category 2 - Emergency
Imminently life-threatening condition or very severe pain or distress.
Maximum waiting time: 10 minutes.

Examples include:
- Severe respiratory distress
- Circulatory compromise (poor perfusion, very abnormal heart rate)
- GCS 9-12
- Very severe pain
- High-risk presentations with potential for rapid deterioration

Category 3 - Urgent
Potentially life-threatening condition or moderately severe pain or distress.
Maximum waiting time: 30 minutes.

Examples include:
- Moderate respiratory distress
- Moderately severe pain
- Seizure now resolved with patient alert
- Moderate blood loss

Category 4 - Semi-urgent
Less urgent condition with mild to moderate symptoms.
Maximum waiting time: 60 minutes.

Examples include:
- Mild haemorrhage
- Moderate pain
- No physiological compromise

Category 5 - Non-urgent
Chronic or minor condition with minimal or no pain or distress.
Maximum waiting time: 120 minutes.

────────────────────────────────────────
SYSTEMATIC TRIAGE APPROACH (MUST FOLLOW)
────────────────────────────────────────

Step 1: Initial assessment and safety
Consider general appearance and whether the patient requires immediate
intervention upon first visualisation.

If the patient requires immediate intervention -> assign Category 1.

Step 2: Identify red flags
Red flags may be identified at ANY stage and immediately increase urgency.

Red flags include:

- Environmental red flags:
    - Aggressive or violent behaviour
    - Communicable disease risk
    - Disaster or mass casualty context

- Physiological red flags:
    - Airway compromise
    - Abnormal breathing or circulation
    - Severe pain
    - Altered level of consciousness
    - Signs of physiological instability

- Historical red flags:
    - High-risk mechanism of injury
    - High-risk medical history or comorbidities
    - Re-presentation with same complaint
    - Extremes of age
    - Cognitive or communication impairment
    - Risk of harm (e.g. domestic violence, abuse)

If red flags are present, assign the HIGHEST appropriate ATS category.

Step 3: Primary survey (ABCDE) and focused assessment
Use available information from:
- Presenting problem
- Associated signs and symptoms
- Vital signs (if obtained)
- Pain severity
- Mental status

Step 4: Assign triage category
Assign the ATS category that reflects the MOST urgent plausible risk
based on the information available.

────────────────────────────────────────
PAIN AND ATS CATEGORIES
────────────────────────────────────────

- Very severe pain -> Category 2
- Moderately severe pain -> Category 3
- Moderate pain -> Category 4
- Minimal or no pain -> Category 5
""".strip()

ALGORITHMS = {"esi": ESI_PROMPT, "ats": ATS_PROMPT}

NURSE_OUTPUT_FORMAT = """
You MUST output a single JSON object with EXACTLY the following fields:

- understood:
    In your first step after the patient speaks: one sentence on what the patient's last reply
    told you (that reply only, not what you knew before). Otherwise null.

- action:
    One of the actions listed under ALLOWED ACTIONS for this turn.

- vital:
    The vital sign to check if action is "check_vital", otherwise null.

- utterance:
    What you say to the patient: a single short question or statement, as said aloud at a
    triage desk (about 20 words at most). Never use names: do not introduce yourself by name,
    and do not ask for or use the patient's name. With "end", a brief closing line (see "end").
    Null only if action is "log_red_flag".

- red_flags:
    A list of red flags you have identified SO FAR.
    This list should accumulate over time.
    If no red flags are identified yet, return an empty list.

- explanation:
    A brief clinical rationale that explicitly references at least one
    specific piece of evidence already obtained (e.g., patient statement,
    observed behavior, or a provided vital sign).

- triage:
    An integer from 1 to 5 indicating your CURRENT estimated triage category,
    according to the algorithm and your explanation.
    This may change over turns as new information is obtained.

- confidence:
    One of ["low", "medium", "high"], reflecting how confident you are
    in the CURRENT triage estimate.

Do NOT reference information that has not been stated or observed.
Do NOT overload with information. Ask one single question, not two joined by "and".
Do NOT invent vital signs or diagnoses.
Do NOT include any text outside the JSON object.
Base all decisions strictly on available information.
""".strip()

ACTION_HELP = {
    "utterance": """
- "utterance":
    Ask a clinically relevant question aimed at eliciting patient history,
    symptoms, pain characteristics, timeline, functional impact, or clarifications.
    You may also provide brief reassurance or explain your reasoning.
    Do NOT assume facts that have not been stated.""",
    "check_vital": """
- "check_vital":
    Request ONE specific vital sign from the following list ONLY: {vitals}
    You can take one vital sign before each question, as the conversation makes it relevant
    (and any you still need after your last question); do not take them all at once.
    Set "vital" to the one you request, and "utterance" to what you say as you take it: a
    short, plain statement such as "Let me check your blood pressure.", not a question, with
    no reason given and no technical terms.
    Do NOT request a vital sign that has already been obtained.
    Do NOT assume the result until it is provided.""",
    "log_red_flag": """
- "log_red_flag":
    Explicitly log one or more red flags that you believe are present
    based ONLY on information already obtained in the dialogue or from
    vital signs that were provided.

    Red flags must be concrete, clinically meaningful findings, such as:
    - "severe respiratory distress"
    - "hemodynamic instability"
    - "altered mental status"
    - "active bleeding"
    - "violent or unsafe behavior"

    Do NOT invent red flags.
    Do NOT log speculative or hypothetical concerns.
    Each red flag must be supported by specific evidence from the dialogue.""",
    "end": """
- "end":
    End the triage and give your final triage level once you have enough
    information, including the vital signs you need, to assign it. A triage
    interview usually covers the presenting complaint, its onset and course,
    pain, relevant history, medications and allergies; your triage record shows
    what is still missing. Ask what this patient's triage needs before you end.
    You do not have to use all your questions. Set "utterance" to a brief closing
    line to the patient (e.g. thanks and what happens next), not a question, and
    never mention the triage level.""",
}


def _nurse_system(persona: BaseModel, algorithm: str, patient: str) -> str:
    return "\n\n".join(
        [
            ALGORITHMS[algorithm],
            _section("OUTPUT FORMAT (STRICT)", NURSE_OUTPUT_FORMAT),
            _section(
                "NURSE PERSONA (shapes which questions you ask and how you speak; it does not change how "
                "you apply the triage algorithm or the level you assign)",
                _persona(persona),
            ),
            _section("THE PATIENT (what you can see at the triage desk)", patient),
        ]
    )


def _belief(step: dict) -> str:
    """The nurse's last assessment: level, confidence, red flags and reasoning."""
    flags = "; ".join(step.get("red_flags") or []) or "none"
    return (f"- level: ESI {step['triage']} (confidence: {step['confidence']})\n- red flags so far: {flags}\n"
            f"- your reasoning: {step.get('explanation') or 'none'}")


def warmth(levels: list[int], truth: int) -> str:
    """Warm/cold feedback on the last of `levels` against the ground truth, and whether it is warmer or
    colder than the level before it."""
    def word(level: int) -> str:
        d = abs(level - truth)
        return "hot" if d == 0 else "warm" if d == 1 else "cold"
    last = f"Your last level, ESI {levels[-1]}, is {word(levels[-1])}"
    if len(levels) < 2:
        return last + "."
    before, now = abs(levels[-2] - truth), abs(levels[-1] - truth)
    trend = "warmer than" if now < before else "colder than" if now > before else "as warm as"
    return f"{last}: {trend} your level before it (ESI {levels[-2]})."


def nurse_messages(
    persona: BaseModel,
    algorithm: str,
    patient: str,
    history: list[dict],
    actions: tuple[str, ...],
    vitals: tuple[str, ...],
    questions_left: Optional[int] = None,
    questions_before_end: int = 0,
    record: Optional[dict] = None,
    deduce: Optional[str] = None,
    record_gaps: tuple[str, ...] = (),
    belief: Optional[dict] = None,
    feedback: Optional[str] = None,
) -> list[dict[str, str]]:
    action_help = "\n".join(ACTION_HELP[a].format(vitals=", ".join(vitals)) for a in actions)
    choices = ", ".join(f'"{a}"' for a in actions)
    sections = [
        _section("DIALOGUE SO FAR", transcript(history) or "[No prior dialogue]"),
        _section("YOUR TRIAGE RECORD (what you have established so far; work through the triage algorithm "
                 "and use your questions to fill in as much of it as you can)", _record(record, history)),
    ]
    if belief is not None:
        sections.append(_section("YOUR BELIEF STATE (your last assessment; update it with every new answer)", _belief(belief)))
    if feedback is not None:
        sections.append(_section(
            "DIALOGUE MASTER'S FEEDBACK ON YOUR LAST LEVEL (private: never mention it, or any triage level, to the patient)",
            feedback + "\nhot = your last level is the right one; warm = one level away; cold = two or more levels "
            "away. Warmer or colder compares it with your level before it. Use it to decide what to ask next.",
        ))
    if deduce:
        sections.append(_section(
            "INFER A VITAL SIGN",
            f"The {deduce} could not be measured. Your next action must be a question that lets you infer it "
            f"from the patient, e.g. about {VITAL_SYMPTOMS[deduce]}. "
            + (f"It counts as one of your {questions_left} remaining questions." if questions_left > 0
               else "Ask it even though you have no other questions left."),
        ))
    elif questions_left == 0:
        sections.append(_section(
            "QUESTIONS LEFT",
            "You have no questions left. Check any vital signs you still need, then end triage with your final level.",
        ))
    elif questions_left is not None:
        sections.append(_section(
            "QUESTIONS LEFT",
            f"You can ask {questions_left} more question{'' if questions_left == 1 else 's'}, counting the one "
            "you may ask now; checking vital signs does not use them up. "
            + (f"You must ask at least {questions_before_end} more before you can end triage. "
               if questions_before_end > 0 else "")
            + (f"You can end triage once every field of your triage record holds an answer (\"none\" or "
               f"\"declined to say\" are answers); still missing: {', '.join(f.replace('_', ' ') for f in record_gaps)}."
               if record_gaps else "" if questions_before_end > 0
               else "You may end triage at any time; you do not have to use them all."),
        ))
    sections.append(_section(
        "ALLOWED ACTIONS", f"You must choose EXACTLY ONE action this turn, one of [{choices}].\n{action_help}"
    ))
    user = "\n\n".join(sections)
    return [{"role": "system", "content": _nurse_system(persona, algorithm, patient)}, {"role": "user", "content": user}]


# ─────────────────────────────────────────
# Patient
# ─────────────────────────────────────────

def _script(script: PatientScript) -> str:
    rows = [
        ("What happened", script.story),
        ("When it started", script.onset),
        ("How long", script.duration),
        ("Where", script.location),
        ("What it feels like", script.quality),
        ("How bad", script.severity),
        ("Other symptoms you have", ", ".join(script.associated_symptoms) or "none"),
        ("Symptoms you do NOT have (say no if asked)", ", ".join(script.pertinent_negatives) or "none listed"),
        ("Medical history", ", ".join(script.medical_history) or "none"),
        ("Medications", ", ".join(script.medications) or "none"),
        ("Allergies", ", ".join(script.allergies) or "none known"),
        ("Social", script.social_history),
    ]
    states = [_level(name, trait, getattr(script, name)) for name, trait in CASE_STATES.items()]
    return "\n".join(states + [f"- {k}: {v}" for k, v in rows if v])


def _condition(case: Case) -> str:
    lines = [f"- Main problem: {case.chief_complaint}"]
    try:
        lines.append(f"- Pain: approximately {float(case.pain):g} out of 10")
    except (TypeError, ValueError):
        pass
    if case.acuity in SEVERITY:
        lines.append(f"- How unwell you are: {SEVERITY[case.acuity]}")
    if case.prior_ed_visits is not None:
        lines.append(f"- Times you came to this emergency department in the past year, before today: {case.prior_ed_visits}")
    return "\n".join(lines)


def patient_messages(
    persona: PatientPersona, case: Case, script: Optional[PatientScript], history: list[dict]
) -> list[dict[str, str]]:
    condition = f"""You came to the emergency department because of the following problem:
{_condition(case)}

This reflects what you personally feel. How openly you show how unwell you are depends on your
persona (for example how you report symptoms, how distressed you are and how openly you disclose), but what
you describe must stay consistent with it.
You may describe this imprecisely, emotionally, or inconsistently depending
on your persona.

You do NOT know medical diagnoses, triage algorithms, or vital signs unless
you personally experienced or were explicitly told them."""
    rules = """General rules:
- Answer only what you personally experienced
- Express uncertainty, confusion, hesitation, or emotion when appropriate
- If the nurse uses a word or asks something you would not understand with your health literacy
  and English, ask what they mean (e.g. "A what? What's that?") instead of answering as if you
  knew; never use medical terms yourself that you would not know
- Do NOT jump ahead in the clinical process
- Do NOT end the conversation on your own
- Do NOT invent diagnoses, vital signs, lab values, or test results
- Never mention triage levels or how urgent your case is rated
- Talk the way patients talk at a triage desk: one or two short sentences, never more than three
- Never say a personal name, your own or anyone else's
- Disfluencies, as often as your persona's disfluency level says, are these five: repetitions
  ("I- I felt it"), filled pauses ("um", "uh"), insertions (adding a word as you restart: "my arm-
  my left arm"), substitutions (swapping a word: "in my stomach- my chest") and speech errors (a
  slip you correct: "my blood plessure- pressure"). Write them as spoken, with a dash where you break
  off; do not use "..." for them
- Speak as your persona traits below describe"""
    output = """You MUST output a single JSON object with exactly these fields:

- utterance: a natural-language patient response
- disclosed: one sentence stating the information you revealed in this reply (for the simulation
  log; the nurse does not see it)

Do NOT include any text outside the JSON object."""

    sections = [
        "You are a patient presenting to the emergency department.",
        _section("YOUR CURRENT CONDITION (PRIVATE)", condition),
    ]
    if script is not None:
        sections.append(
            _section(
                "YOUR STORY (PRIVATE)",
                _script(script) + "\n\nStay consistent with this story. Share details only when they come up; do "
                "not recite it. If asked about something it does not cover, answer plausibly and consistently "
                "with it, or say you are not sure.",
            )
        )
    sections += [
        _section("RESPONSE RULES (STRICT)", rules),
        _section("PATIENT PERSONA", _persona(persona)),
        _section("OUTPUT FORMAT (MANDATORY)", output),
    ]
    user = _section("DIALOGUE SO FAR", transcript(history)) + "\n\nReply to the nurse's last message."
    return [{"role": "system", "content": "\n\n".join(sections)}, {"role": "user", "content": user}]



# ─────────────────────────────────────────
# Dialogue master
# ─────────────────────────────────────────

# Each check is phrased so that true means the utterance passes it: small models invert negated criteria.
JUDGE_CRITERIA = {
    "patient": """- faithful: everything it says agrees with the ground truth below (complaint, pain, how
  unwell, story if given) and with the patient's gender, and it only mentions things the patient
  could know (not diagnoses, vital-sign values, test results or triage levels). Understating,
  overstating, vagueness or withholding that fits the persona (e.g. minimising or amplifying
  symptom reporting, selective or guarded disclosure, patchy recall) still counts as faithful;
  contradicting the facts or inventing new ones does not.
- informative: it responds to the nurse's last message (answers it, or plausibly asks for clarification).
- in_persona: it matches the persona traits listed and the mental state and intoxication in the
  ground truth, and it is as short as spoken triage talk (at most three short sentences).
- plausible: a real patient in this situation could plausibly say it at a triage desk: realistic
  content and everyday wording, with no medically odd claims (e.g. that a treatable problem is beyond
  help), textbook phrasing or theatrical lines.
- no_names: it says no personal name, neither the patient's own nor anyone else's.""",
    "nurse": """- faithful: every fact it states was obtained in the dialogue or from vital signs, it keeps
  the triage level to itself, it suits triage under {algorithm}, and it fits the patient's apparent
  gender.
- informative: it moves triage forward: it asks for information not obtained yet, or gives
  reassurance or an explanation the patient needs.
- in_persona: it matches the nurse persona traits listed (interview style only), and it is one short
  question or statement, as said aloud at a triage desk.
- plausible: a real triage nurse could plausibly say it at the desk: realistic, professional and
  natural wording.
- no_names: it uses no personal names: the nurse does not introduce itself by name and does not
  ask for or use the patient's name.
- one_question: it asks for one thing only: not two questions joined by "and" or asked one after
  the other (e.g. not "When did it start, and is it bleeding?" or "How long have you felt dizzy?
  Did it start suddenly?"). One yes/no question listing related symptoms ("Any nausea, vomiting or
  fever?") or offering alternatives ("Did it start suddenly or gradually?") is one question; a
  statement without a question passes.""",
}

JUDGE_SYSTEM = """You are the dialogue master of a simulated emergency department triage
conversation. You check ONE new {speaker} utterance before it is added to the dialogue:

{criteria}

Be strict about contradictions and invented facts, lenient about wording and style details that do
not matter. Write a brief critique first; if a check fails, say exactly what must change. Then answer
each check with true if the utterance passes it, and false only if it clearly fails it."""


def judge_messages(
    speaker: str,
    utterance: str,
    persona: BaseModel,
    history: list[dict],
    *,
    case: Optional[Case] = None,
    script: Optional[PatientScript] = None,
    algorithm: str = "esi",
    patient: str = "",
) -> list[dict[str, str]]:
    """Messages asking the judge to verify a nurse or patient utterance."""
    criteria = JUDGE_CRITERIA[speaker].format(algorithm=algorithm.upper())
    sections = []
    if speaker == "patient":
        truth = f"- gender: {case.gender or getattr(persona, 'gender', 'unknown')}\n{_condition(case)}"
        if script is not None:
            truth += "\n" + _script(script)
        sections.append(_section("GROUND TRUTH", truth))
    else:
        sections.append(_section("THE PATIENT (as seen at the triage desk)", patient))
    sections += [
        _section(f"{speaker.upper()} PERSONA", _persona(persona, per_line_only=True)),
        _section("DIALOGUE SO FAR", transcript(history) or "[No prior dialogue]"),
        _section(f"NEW {speaker.upper()} UTTERANCE", utterance),
    ]
    return [
        {"role": "system", "content": JUDGE_SYSTEM.format(speaker=speaker, criteria=criteria)},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


EDIT_SYSTEM = """You are the dialogue master of a simulated emergency department triage
conversation. A new {speaker} utterance failed your checks, even after the {speaker} rewrote it:

{criteria}

Your critique: {critique}

Write the utterance yourself so that it passes every check: keep what the {speaker} meant to say and
how this persona speaks, and change as little as possible.{extra}"""


def edit_messages(speaker: str, utterance: str, critique: str, persona: BaseModel, history: list[dict], **context):
    """Messages asking the dialogue master to write a line that failed its checks twice; `context`
    is what judge_messages takes (case, script, algorithm, patient)."""
    messages = judge_messages(speaker, utterance, persona, history, **context)
    criteria = JUDGE_CRITERIA[speaker].format(algorithm=context.get("algorithm", "esi").upper())
    extra = " Also give, in one sentence, the information your version reveals." if speaker == "patient" else ""
    system = EDIT_SYSTEM.format(speaker=speaker, criteria=criteria, critique=critique, extra=extra)
    return [{"role": "system", "content": system}, messages[1]]


READING_SYSTEM = """You are the dialogue master of a simulated emergency department triage. You keep the
nurse's triage record: what the conversation has established so far. Read the latest exchange (the
nurse's question and the patient's reply) and answer from its words alone, not from anything else
you may know about the patient.
- conveyed: one sentence stating the information the patient's reply conveys.
- record: the full triage record after this exchange. For every field write its complete current
  value, briefly: what was already recorded plus anything new this reply adds; copy fields the reply
  does not touch as they are, and write exactly "{unknown}" for a field that is still unknown.
  Record only what was said: a vague answer is recorded as vague (e.g. "pain 'really bad', no score
  given"); if the patient cannot or will not answer what was asked, record that (e.g. "declined to
  say", "does not know"). A reply that does not answer the question (it changes the subject, asks
  for clarification or only voices worry) adds nothing about it: never record a denial or finding
  the patient did not state, never infer a diagnosis, never add facts.
Record fields:
{fields}"""


def reading_messages(record: dict, question: str, reply: str) -> list[dict[str, str]]:
    """The dialogue master's reading of one exchange, given the triage record so far."""
    fields = "\n".join(f"- {name}: {desc}" for name, desc in RECORD_FIELDS.items())
    current = "\n".join(f"- {name}: {record.get(name) or RECORD_UNKNOWN}" for name in RECORD_FIELDS)
    exchange = f'Nurse: """{question}"""\nPatient: """{reply}"""'
    return [
        {"role": "system", "content": READING_SYSTEM.format(fields=fields, unknown=RECORD_UNKNOWN)},
        {"role": "user", "content": _section("TRIAGE RECORD SO FAR", current) + "\n\n" + _section("LATEST EXCHANGE", exchange)},
    ]


def with_feedback(messages: list[dict[str, str]], draft: str, critique: str) -> list[dict[str, str]]:
    """Append a rejected draft and the reviewer's critique, asking for a new reply."""
    return messages + [
        {"role": "assistant", "content": draft},
        {
            "role": "user",
            "content": f"A reviewer rejected that reply: {critique}\nWrite a new reply that fixes this, "
            "in the same JSON format.",
        },
    ]


# ─────────────────────────────────────────
# Standardized-patient scripts
# ─────────────────────────────────────────

def _arrival_facts(case: Case) -> list[str]:
    """What is visible when the patient arrives at the triage desk."""
    extra = case.model_extra or {}
    facts = [f"- chief complaint: {case.chief_complaint}", f"- gender: {case.gender or 'not recorded'}"]
    if extra.get("arrival_transport"):
        facts.append(f"- arrived by: {str(extra['arrival_transport']).lower()}")
    facts += [f"- {name}: {_vital(name, case.vitals.get(name))}" for name in case.vitals]
    facts.append(f"- pain: {case.pain if case.pain is not None else 'not recorded'} (0-10)")
    return facts


def _case_facts(case: Case) -> str:
    """Everything the dialogue master knows about a case (never shown to the nurse)."""
    extra = case.model_extra or {}
    facts = _arrival_facts(case)
    if case.acuity in SEVERITY:
        scale = (case.acuity_scale or "triage").upper()
        facts.append(f"- acuity: {scale} {case.acuity} of 5 (1 = most urgent). {SEVERITY[case.acuity]}")
    if extra.get("specialisation"):
        facts.append(f"- specialty: {extra['specialisation']}")
    if case.diagnoses:
        facts.append(f"- diagnoses made in the ED: {'; '.join(case.diagnoses)}")
    if case.medications:
        facts.append(f"- home medications (the patient takes these): {', '.join(case.medications)}")
    if case.prior_ed_visits is not None:
        facts.append(f"- ED visits in the past year, before this one: {case.prior_ed_visits}")
    return "\n".join(facts)


IN_SITU_SYSTEM = """You are the dialogue master of a simulated emergency department triage. Before a case
is simulated, decide whether this patient would actually have a spoken triage conversation with the
triage nurse at the desk. Judge only from what is visible on arrival, as the triage desk would: the
complaint, the arrival mode and the vital signs. There would be no conversation if, for example, the
patient is in cardiac arrest, unresponsive, or so unstable on arrival that they are taken straight to
resuscitation. A patient who is seriously unwell but can talk would still have a (perhaps brief)
conversation. Reason briefly, then decide."""


def in_situ_messages(case: Case) -> list[dict[str, str]]:
    """Only what is visible on arrival: no acuity, diagnoses, medications or visit history."""
    return [
        {"role": "system", "content": IN_SITU_SYSTEM},
        {"role": "user", "content": _section("ON ARRIVAL", "\n".join(_arrival_facts(case)))},
    ]


APPEARANCE_SYSTEM = """You are the dialogue master of a simulated emergency department triage. Describe what the
triage nurse sees when this patient reaches the triage desk, before anyone speaks, from the facts below
only: how the patient arrives and moves, how they breathe and talk, their skin, their alertness and
behaviour, visible pain or distress, and any visible injury, bleeding, swelling or rash.

Write observations a nurse could see, in plain words, never judgements: no numbers, no vital-sign
values, no diagnoses or conditions, no triage levels, and no words such as "critical", "unstable",
"stable", "urgent", "serious" or "emergency". Show abnormal vital signs only as signs a nurse could see
(for example fast breathing, bluish lips, pale and clammy skin, flushed and sweaty); with normal vital
signs, show nothing unusual about them. Show pain as the patient's body shows it (grimacing, guarding,
holding the painful part, limping), and the mental state and intoxication given below as behaviour.
Keep each field short."""

# what the appearance must never contain: numbers, levels, diagnoses' usual judgements of urgency
APPEARANCE_BANNED = re.compile(
    r"\d|\b(?:ESI|triage|level|priority|critical\w*|unstable|stable|urgent\w*|emergen\w*|serious\w*|"
    r"life[- ]threatening|resuscitat\w*|deteriorat\w*|diagnos\w*)\b", re.I)


def appearance_messages(case: Case, persona: PatientPersona, script: Optional[PatientScript]) -> list[dict[str, str]]:
    """What is visible on arrival (complaint, arrival, vital signs, pain), the mental state and
    intoxication the case supports, and the visible persona traits; no acuity, diagnoses or history."""
    facts = _arrival_facts(case)
    facts += [f"- mental state: {script.cognitive_state if script else 'alert'}",
              f"- intoxication: {script.intoxication if script else 'none'}"]
    shown = {"age_group", "distress", "demeanour"}
    traits = [_level(n, t, getattr(persona, n)) for n, t in traits_of(persona).items() if n in shown]
    user = "\n\n".join([_section("ON ARRIVAL", "\n".join(facts)), _section("VISIBLE PERSONA TRAITS", "\n".join(traits)),
                         "Describe what the nurse sees."])
    return [{"role": "system", "content": APPEARANCE_SYSTEM}, {"role": "user", "content": user}]


SCRIPT_SYSTEM = """You are the dialogue master of a simulated emergency department triage. You write the
standardized-patient script for one episode: what this patient experienced and knows, from their own
perspective and in plain words. An AI actor playing the patient will answer the triage nurse from it.

The script must be clinically coherent with ALL of the case facts: the chief complaint, the vital
signs (abnormal vital signs should be explained by the story), the pain score, how unwell the patient
really is (the acuity), the patient's gender, and the diagnoses and medications if given. Fit the
circumstances and history to the patient persona. The patient does not know their diagnosis or their
vital-sign numbers unless a previous diagnosis is part of their history. Never mention triage levels
(ESI, ATS, acuity), vital-sign numbers or a current diagnosis anywhere in the script.

The script records what really happened. The persona changes only how the patient tells it, which
the actor plays: write the true facts even for a patient with unreliable recall or guarded
disclosure, and never put persona traits or demographics (such as "unreliable", "hostile" or
"older male") in the fields.

Set cognitive_state and intoxication from the case alone (for example altered mental status,
confusion, delirium or intoxication in the complaint or diagnoses), never from the persona; without
such evidence they are alert and none.

Fill the other fields from the patient's point of view, in plain words:
- story: what happened, in the first person, 2-4 sentences.
- onset: when and how it started (e.g. "suddenly, about two hours ago").
- duration: how long it has lasted, and whether it is constant or comes and goes.
- location: where in the body (e.g. "centre of the chest, going into the left arm"), or null.
- quality: what the main symptom feels like (e.g. "tight, like a band", "sharp"), or null.
- severity: how bad it feels and what it stops the patient doing.
- associated_symptoms: other symptoms the patient has.
- pertinent_negatives: relevant symptoms the patient does NOT have, so the actor can deny them when
  asked. Never deny a symptom that the complaint or diagnoses imply.
- medical_history, medications, allergies, social_history: as the patient would tell them."""


def script_messages(case: Case, persona: PatientPersona) -> list[dict[str, str]]:
    user = "\n\n".join(
        [_section("CASE", _case_facts(case)), _section("PATIENT PERSONA", _persona(persona)), "Write the script."]
    )
    return [{"role": "system", "content": SCRIPT_SYSTEM}, {"role": "user", "content": user}]
