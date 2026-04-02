import pytest

from triagesim.core.environment import TriageEnv, NURSE
from triagesim.core.output_schema import NurseOutput
from triagesim.core.state_store import RedisStateStore


@pytest.fixture
def env():
    store = RedisStateStore(db=15)
    store.client.flushdb()
    return TriageEnv(store=store, max_turns=2)


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

    assert obs["state"]["turn"] == 1
    assert obs["belief"] is not None
