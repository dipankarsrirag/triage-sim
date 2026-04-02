from typing import List

from triagesim.personas.schema import NursePersona
from triagesim.core.output_schema import NurseOutput
from triagesim.agents.base_agent import BaseLLM
from triagesim.core.belief_graph import BeliefSlotUpdate
from triagesim.core.llm_detectors import LLMEvidenceDetector


class NurseAgent:
    """
    Nurse agent responsible for:
    - Choosing the next triage action (NurseOutput)
    - Inferring belief updates from the nurse's perspective

    The SAME LLM + persona is used for both decision-making
    and belief inference.
    """

    def __init__(
        self,
        llm: BaseLLM,
        persona: NursePersona,
        algorithm: str = "esi",
    ):
        self.llm = llm
        self.persona = persona
        self.seen_vital = False
        self.logged_flag = False
        self.algorithm = algorithm.lower()

        if self.algorithm not in {"esi", "ats"}:
            raise ValueError(f"Unsupported triage algorithm: {algorithm}")

        # Belief inference uses the SAME LLM and persona
        self._belief_detector = LLMEvidenceDetector(llm=self.llm)

    # ─────────────────────────────────────────
    # Action selection
    # ─────────────────────────────────────────

    def act(self, history: str, known_vitals: set[str]) -> NurseOutput:
        """
        Given dialogue history, produce the next nurse action
        as a structured NurseOutput.
        """
        self.start = True if not history else False

        prompt = self._build_prompt(history, known_vitals)

        # IMPORTANT:
        # llm.generate() MUST return a NurseOutput
        # because output_type=NurseOutput was set on the backend
        result = self.llm.generate(prompt)
        return result  # type: ignore[return-value]

    # ─────────────────────────────────────────
    # Belief inference (nurse-owned)
    # ─────────────────────────────────────────

    def infer_belief_updates(
        self,
        history: str,
        last_utterance: str,
        turn: int,
    ) -> List[BeliefSlotUpdate]:
        """
        Infer belief updates from the nurse's perspective.

        This is called by the environment AFTER a patient utterance.
        Failures must never crash the episode.
        """
        updates: List[BeliefSlotUpdate] = []

        try:
            # Pain
            pain = self._belief_detector.detect_pain(last_utterance)
            if getattr(pain, "pain_severity", None):
                updates.append(
                    BeliefSlotUpdate(
                        slot="pain_severity",
                        value=pain.pain_severity,
                        evidence=last_utterance,
                        source="patient",
                        turn=turn,
                        certainty="explicit",
                    )
                )

            if getattr(pain, "pain_location", None):
                updates.append(
                    BeliefSlotUpdate(
                        slot="pain_location",
                        value=pain.pain_location,
                        evidence=last_utterance,
                        source="patient",
                        turn=turn,
                        certainty="explicit",
                    )
                )

            # Symptoms
            symptoms = self._belief_detector.detect_symptoms(last_utterance)
            for s in getattr(symptoms, "symptoms", []) or []:
                updates.append(
                    BeliefSlotUpdate(
                        slot="associated_symptom",
                        value=s,
                        evidence=last_utterance,
                        source="patient",
                        turn=turn,
                        certainty="explicit",
                    )
                )

            # Chief complaint
            chief = self._belief_detector.detect_chief_complaint(last_utterance)
            if getattr(chief, "chief_complaint", None):
                updates.append(
                    BeliefSlotUpdate(
                        slot="chief_complaint",
                        value=chief.chief_complaint,
                        evidence=last_utterance,
                        source="patient",
                        turn=turn,
                        certainty="explicit",
                    )
                )

            # Duration
            duration = self._belief_detector.detect_duration(last_utterance)
            if getattr(duration, "duration", None):
                updates.append(
                    BeliefSlotUpdate(
                        slot="duration",
                        value=duration.duration,
                        evidence=last_utterance,
                        source="patient",
                        turn=turn,
                        certainty="explicit",
                    )
                )

            # Red flags (suspected)
            red_flags = self._belief_detector.detect_red_flags(last_utterance)
            for rf in getattr(red_flags, "red_flags", []) or []:
                updates.append(
                    BeliefSlotUpdate(
                        slot="red_flag",
                        value=rf,
                        evidence=last_utterance,
                        source="patient",
                        turn=turn,
                        certainty="suspected",
                    )
                )

        except Exception:
            # Safety guarantee: belief inference must never crash the episode
            return []

        return updates

    # ─────────────────────────────────────────
    # Prompt construction
    # ─────────────────────────────────────────

    def _format_known_vitals(self, known_vitals: set[str]) -> str:
        if not known_vitals:
            return "- None obtained yet"

        return "\n".join(f"- {v} (obtained)" for v in sorted(known_vitals))

    def _get_actions(self, known_vitals: set[str]) -> str:
        """
        Build the ALLOWED ACTIONS block, masking already-known vitals.
        """

        ALL_VITALS = {"temperature", "heartrate", "resprate", "o2sat", "sbp"}
        remaining_vitals = sorted(ALL_VITALS - set(known_vitals))

        if remaining_vitals and not self.seen_vital:
            # print("##############", remaining_vitals, "############")
            check_vital_block = f"""
    - "check_vital":
        Request ONE specific vital sign from the following list ONLY:
        {", ".join(remaining_vitals)}

        Do NOT request a vital sign that has already been obtained.
        Do NOT request multiple vitals in a single turn.
        Do NOT assume the result until it is provided.
    """
            check_vital = ', "check_vital"'
            utterance_rule = ' "utterance" or "check_vital"'
        else:
            check_vital_block = ""
            check_vital = ""
            utterance_rule = ' "utterance"'

        if self.logged_flag:
            log_red_flags_block = ""
            action_choices = '"utterance"' + check_vital + ', "end"'

        else:
            log_red_flags_block = """
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
            Each red flag must be supported by specific evidence from the dialogue.
        """
            action_choices = '"utterance"' + check_vital + ', "log_red_flag", "end"'

        # print("------------------------")
        # print("Actions:", action_choices)
        # print(f"Flags:- Vitals ({self.seen_vital}) | Red Flag: ({self.logged_flag})")
        # print("------------------------")
        return f"""
    ────────────────────────────────────────
    ALLOWED ACTIONS
    ────────────────────────────────────────

    You must choose EXACTLY ONE action each turn.

    Available actions:

    - "utterance":
        Ask a clinically relevant question aimed at eliciting patient history,
        symptoms, pain characteristics, timeline, functional impact, or clarifications.
        You may also provide brief reassurance or explain your reasoning.
        Do NOT assume facts that have not been stated.

    {check_vital_block}

    {log_red_flags_block}

    - "end":
        End the triage interaction when you believe sufficient information has
        been gathered to assign a triage category.
        You should only choose this when no further clarification or vital signs
        are required.

    ────────────────────────────────────────
    OUTPUT FORMAT (STRICT)
    ────────────────────────────────────────

    You MUST output a single JSON object with EXACTLY the following fields:

    - action:
        One of [{action_choices}]

    - utterance:
        A string containing a single question or a single statement if action is{utterance_rule}.
        {'Always state your name in the beginning of the interaction.' if self.start else ""}

    - triage:
        An integer from 1 to 5 indicating your CURRENT estimated triage category,
        according to the algorithm.
        This may change over turns as new information is obtained.

    - red_flags:
        A list of red flags you have identified SO FAR.
        This list should accumulate over time.
        If no red flags are identified yet, return an empty list.

    - confidence:
        One of ["low", "medium", "high"], reflecting how confident you are
        in the CURRENT triage estimate.

    - explanation:
        A brief clinical rationale that explicitly references at least one
        specific piece of evidence already obtained (e.g., patient statement,
        observed behavior, or a provided vital sign).

    Do NOT reference information that has not been stated or observed.
    Do NOT overload with information. Ask one single question.
    Do NOT invent vital signs or diagnoses.
    Do NOT include any text outside the JSON object.
    Base all decisions strictly on available information.
    """.strip()

    def _build_prompt(self, history: str, known_vitals: set[str]) -> str:
        persona_desc = "\n".join(
            f"- {k}: {v}" for k, v in self.persona.model_dump().items()
        )
        algorithm_prompt = (
            self._esi_prompt() if self.algorithm == "esi" else self._ats_prompt()
        )

        known_vitals_block = self._format_known_vitals(known_vitals=known_vitals)
        actions_block = self._get_actions(known_vitals=known_vitals)

        return f"""
{algorithm_prompt}

────────────────────────────────────────
NURSE PERSONA
────────────────────────────────────────
{persona_desc}

────────────────────────────────────────
KNOWN VITAL SIGNS
────────────────────────────────────────
{known_vitals_block}

────────────────────────────────────────
DIALOGUE SO FAR
────────────────────────────────────────
{history if history else "[No prior dialogue]"}

{actions_block}

────────────────────────────────────────
REMINDER
────────────────────────────────────────
You MUST output a JSON object matching the required schema.
""".strip()

    def _esi_prompt(self) -> str:
        return """
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

    def _ats_prompt(self) -> str:

        return """
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
