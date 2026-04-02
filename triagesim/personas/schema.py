from typing import Optional
from pydantic import BaseModel, Field


class PatientPersona(BaseModel):
    age_group: str
    gender: str
    ethnicity: str
    socioeconomic_status: str
    language_proficiency: str

    recall_accuracy: str
    cognitive_state: str
    trust_in_healthcare: str
    pain_expression: str
    reactivity_to_clinician_emotion: str
    emotion_regulation: str

    disfluency_rate: str
    topic_drift: str
    verbosity: str
    instruction: Optional[str] = Field(
        default=None,
        description="Free-text instruction describing how the patient should speak",
    )


class NursePersona(BaseModel):
    gender: str
    ethnicity: str
    experience_level: str
    risk_tolerance: str
    guideline_adherence: str
    communication_style: str
    verbosity: str
    emotional_expression: str

    instruction: Optional[str] = Field(
        default=None,
        description="Free-text instruction describing how the nurse should speak",
    )
