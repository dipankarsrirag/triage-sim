"""
Ground-truth cases (clinical vignettes). The nurse never sees a case directly: it only learns what
the patient says and the vitals it asks for. The patient knows its complaint, pain and how unwell
it is (the acuity), plus the script the dialogue master writes for it.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from triagesim.schemas import VITALS


def _blank_to_none(v: Any) -> Any:
    if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


class Case(BaseModel):
    """Other fields in the source row (e.g. arrival_transport, specialisation, dataset) are kept and
    passed through to the episode artifacts."""

    model_config = ConfigDict(extra="allow")

    case_id: str
    chief_complaint: str = Field(validation_alias=AliasChoices("chief_complaint", "chiefcomplaint"))
    vitals: dict[str, Optional[float]] = Field(default_factory=dict)
    acuity: Optional[int] = None
    acuity_scale: Optional[Literal["esi", "ats"]] = None
    pain: Optional[str] = None
    gender: Optional[str] = None
    diagnoses: list[str] = Field(default_factory=list, description="e.g. MIMIC-IV-ED ICD titles; seed patient scripts")
    medications: list[str] = Field(default_factory=list, description="home medications (e.g. MIMIC-IV-ED medrecon)")
    prior_ed_visits: Optional[int] = Field(None, description="the patient's ED visits in the 12 months before this one")

    @field_validator("case_id", mode="before")
    @classmethod
    def _id_to_str(cls, v: Any) -> str:
        return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)

    @field_validator("vitals", mode="before")
    @classmethod
    def _clean_vitals(cls, v: dict) -> dict:
        return {k: _blank_to_none(x) for k, x in v.items()}

    @field_validator("acuity", mode="before")
    @classmethod
    def _clean_acuity(cls, v: Any) -> Optional[int]:
        v = _blank_to_none(v)
        return None if v is None else int(float(v))

    @field_validator("gender", mode="before")
    @classmethod
    def _clean_gender(cls, v: Any) -> Optional[str]:
        v = _blank_to_none(v)
        if v is None:
            return None
        v = str(v).strip().lower()
        return {"m": "male", "f": "female"}.get(v, v)

    @field_validator("pain", mode="before")
    @classmethod
    def _clean_pain(cls, v: Any) -> Optional[str]:
        v = _blank_to_none(v)
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        return None if v is None else str(v).strip()


def load_cases(path: str | Path) -> list[Case]:
    """Load cases from .jsonl, .json (a list) or .csv.

    Rows may nest vitals under "vitals" or give them as flat columns (as in MIMIC-IV-ED triage.csv,
    whose `stay_id` becomes the case id; gender "M"/"F" as in edstays.csv is normalised). Rows
    without an id are numbered.
    """
    path = Path(path)
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())

    cases = []
    for i, row in enumerate(rows):
        row = dict(row)
        if "case_id" not in row:
            row["case_id"] = row.pop("stay_id", i)
        if "vitals" not in row:
            row["vitals"] = {v: row.pop(v, None) for v in VITALS}
        cases.append(Case.model_validate(row))
    return cases
