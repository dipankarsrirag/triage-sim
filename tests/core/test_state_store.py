import pytest

from triagesim.core.environment import TriageEnv, NURSE, PATIENT
from triagesim.core.output_schema import NurseOutput, PatientOutput
from triagesim.core.state_store import RedisStateStore


class StubNurseAgent:
    """Nurse agent that proposes no belief updates."""

    def __init__(self):
        self.seen_vital = False
        self.logged_flag = False

    def infer_belief_updates(self, history, last_utterance, turn):
        return []


def _redis_available() -> bool:
    try:
        RedisStateStore(db=15).client.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis is not reachable on localhost:6379",
)


@pytest.fixture
def env():
    store = RedisStateStore(db=15)
    store.client.flushdb()
    return TriageEnv(store=store, nurse_agent=StubNurseAgent(), max_turns=2)


def test_environment_redis_roundtrip(env):
    run_id = env.reset(
        {
            "vitals": {"o2sat": 95},
            "acuity": 4,
            "pain": 3,
        }
    )

    obs, done = env.step(
        run_id,
        NURSE,
        NurseOutput(
            action="utterance",
            utterance="Hello",
            triage=4,
            red_flags=[],
            confidence="low",
            explanation="Greeting",
        ),
    )

    assert not done

    # Nurse micro-turns do not advance the turn counter.
    assert obs["state"]["turn"] == 0
    assert obs["belief"] is not None
    assert obs["history"][-1]["utterance"] == "Hello"

    # A patient turn advances the counter, and survives the Redis roundtrip.
    obs, done = env.step(
        run_id,
        PATIENT,
        PatientOutput(utterance="Hi, I have a headache."),
    )

    assert obs["state"]["turn"] == 1
    assert [h["actor"] for h in obs["history"]] == [NURSE, PATIENT]
