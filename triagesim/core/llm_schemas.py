from typing import Literal, List, Optional
from pydantic import BaseModel, Field

SeverityLabel = Literal["mild", "moderate", "severe", "unknown"]

ALLOWED_VITALS = {"temperature", "heartrate", "resprate", "o2sat", "sbp"}


class LLMVitalsDetection(BaseModel):

    vitals: List[str] = Field(description="List of vitals the nurse intends to request")

    @classmethod
    def validate_vitals(cls, values: List[str]) -> List[str]:
        """
        Only return vitals that are in the allowed set.
        """
        return [v for v in values if v in ALLOWED_VITALS]


class LLMPainDetection(BaseModel):
    pain_severity: Optional[SeverityLabel] = Field(None, description="Severity label")
    pain_location: Optional[str] = Field(None, description="Anatomical location")


class LLMSymptomDetection(BaseModel):
    symptoms: List[str] = Field(default_factory=list)


class LLMRedFlagDetection(BaseModel):
    red_flags: List[str] = Field(default_factory=list)


class LLMDurationDetection(BaseModel):
    duration: Optional[str] = Field(None, description="Acute/subacute/chronic etc.")


class LLMChiefComplaint(BaseModel):
    chief_complaint: Optional[str] = Field(
        None, description="Patient's chief complaint phrase"
    )
