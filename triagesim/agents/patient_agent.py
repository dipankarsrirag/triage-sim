"""
Patient agent implementation for TriageSim.

The patient agent:
- Responds ONLY to nurse prompts
- Is constrained by a categorical persona
- Produces structured outputs (PatientOutput)
- Does not maintain belief state (environment does)
"""

from triagesim.personas.schema import PatientPersona
from triagesim.core.output_schema import PatientOutput
from triagesim.agents.base_agent import BaseLLM


class PatientAgent:
    """
    PatientAgent generates patient utterances conditioned on:
    - dialogue history
    - a categorical persona
    - explicit response budget rules
    """

    def __init__(self, llm: BaseLLM, persona: PatientPersona):
        self.llm = llm
        self.persona = persona

    def act(self, history: str, chief_complaint: str, pain: int) -> PatientOutput:
        """
        Generate the patient's next utterance as a structured PatientOutput.

        The LLM backend MUST have been initialized with:
            output_type = PatientOutput
        """
        prompt = self._build_prompt(history, chief_complaint, pain)

        # Structured generation enforced by pydantic-ai
        result = self.llm.generate(prompt)
        return result  # type: ignore[return-value]

    # ─────────────────────────────────────────
    # Prompt construction
    # ─────────────────────────────────────────

    def _build_prompt(self, history: str, chief_complaint: str, pain: int) -> str:
        persona = self.persona.model_dump()
        persona_desc = "\n".join(f"- {k}: {v}" for k, v in persona.items())

        verbosity = persona["verbosity"]
        trust = persona["trust_in_healthcare"]

        # -------------------------------------------------
        # Response budget rules (STRICT)
        # -------------------------------------------------

        if verbosity == "low":
            budget_rules = (
                "- Answer only the nurse's current question\n"
                "- Use 1 to 2 sentences\n"
                "- Do NOT introduce new symptoms or history unless directly asked\n"
                "- Be concise and focused"
            )
        elif verbosity == "medium":
            budget_rules = (
                "- Answer the nurse's current question\n"
                "- You MAY add at most one minor clarification\n"
                "- Use 2 to 4 sentences\n"
                "- Do NOT introduce multiple new symptoms unless asked"
            )
        else:  # high
            budget_rules = (
                "- Answer the nurse's current question\n"
                "- You MAY add context, emotions, or one extra relevant detail\n"
                "- Use up to 6 sentences\n"
                "- Avoid listing many symptoms unless explicitly asked"
            )

        # -------------------------------------------------
        # Trust-based withholding behavior
        # -------------------------------------------------

        if trust == "high":
            withholding_rule = (
                "- Be open, cooperative, and forthcoming when answering questions"
            )
        elif trust == "medium":
            withholding_rule = "- Answer honestly, but avoid volunteering sensitive or alarming details unless asked"
        else:  # low
            withholding_rule = (
                "- Be guarded and cautious\n"
                "- Do NOT volunteer concerning or personal details unless explicitly asked"
            )

        # -------------------------------------------------
        # Final prompt
        # -------------------------------------------------

        return f"""
You are a patient presenting to the emergency department.

────────────────────────────────────────
YOUR CURRENT CONDITION (PRIVATE)
────────────────────────────────────────
You came to the emergency department because of the following problem:
- Main problem: {chief_complaint}

Your current pain level is approximately {pain} out of 10.

This reflects what you personally feel.
You may describe this imprecisely, emotionally, or inconsistently depending
on your persona.

You do NOT know medical diagnoses, triage algorithms, or vital signs unless
you personally experienced or were explicitly told them.

────────────────────────────────────────
RESPONSE RULES (STRICT)
────────────────────────────────────────
{budget_rules}

Additional rule:
{withholding_rule}

General rules:
- Answer only what you personally experienced
- Express uncertainty, confusion, hesitation, or emotion when appropriate
- Ask for clarification if you do not understand the nurse's question
- Do NOT jump ahead in the clinical process
- Do NOT end the conversation on your own
- Do NOT invent diagnoses, vital signs, lab values, or test results

────────────────────────────────────────
PATIENT PERSONA
────────────────────────────────────────
{persona_desc}

────────────────────────────────────────
DIALOGUE SO FAR
────────────────────────────────────────
{history if history else "[No prior dialogue]"}

────────────────────────────────────────
OUTPUT FORMAT (MANDATORY)
────────────────────────────────────────
You MUST output a single JSON object with exactly one field:

- utterance: a natural-language patient response

Do NOT include any text outside the JSON object.
""".strip()
