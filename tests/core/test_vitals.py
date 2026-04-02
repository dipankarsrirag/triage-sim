from triagesim.core.vitals import rule_extract_vitals, extract_requested_vitals


def test_rule_extract_vitals_simple():
    txt = "Check blood pressure and o2 sat please."
    assert set(rule_extract_vitals(txt)) == {"sbp", "o2sat"}


def test_rule_extract_none():
    txt = "Let me ask what is going on."
    assert rule_extract_vitals(txt) == []


class DummyDetector:
    def detect_vitals(self, text):
        return type("X", (), {"vitals": ["heartrate", "temperature"]})


def test_llm_extract(monkeypatch):
    txt = "Ask for everything"
    det = DummyDetector()
    result = extract_requested_vitals(txt, detector=det)
    assert set(result) == {"heartrate", "temperature"}


def test_fallback(monkeypatch):
    txt = "Ask for everything"
    # no detector provided => blank
    assert extract_requested_vitals(txt, detector=None) == []
