from pydantic import BaseModel


class BaseAgentOutput(BaseModel):
    """
    Base class for all agent outputs.
    Enforces JSON-only, structured communication.
    """

    class ConfigDict:
        extra = "forbid"  # CRITICAL: no hallucinated fields
