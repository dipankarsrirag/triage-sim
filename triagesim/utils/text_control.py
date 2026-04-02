def enforce_response_budget(text: str, response_length: str) -> str:
    """
    Trims patient utterance ONLY if it clearly violates
    the persona response_length budget.
    """
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    limits = {
        "low": 2,
        "medium": 4,
        "high": 6,
    }

    max_sentences = limits.get(response_length, 4)

    if len(sentences) <= max_sentences:
        return text  # no violation

    # Trim conservatively
    trimmed = ". ".join(sentences[:max_sentences]).strip()
    if not trimmed.endswith("."):
        trimmed += "."

    return trimmed
