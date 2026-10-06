import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import openai
import pytest

from pydantic import BaseModel, field_validator

from triagesim.llm import GenerationError, OpenRouter, Request, generate, load_backend


class Say(BaseModel):
    utterance: str

    @field_validator("utterance")
    @classmethod
    def _not_blank(cls, v):
        if not v.strip():
            raise ValueError("empty")
        return v


class Scripted:
    """Backend returning queued answers and recording the seeds it was called with."""

    model = "scripted"

    def __init__(self, answers):
        self.answers = list(answers)
        self.seeds = []

    def complete(self, requests):
        self.seeds.append([r.seed for r in requests])
        return [self.answers.pop(0) for _ in requests]


def req(seed=None):
    return Request([{"role": "user", "content": "hi"}], Say, {}, seed)


def test_generate_retries_invalid_output_with_new_seed():
    backend = Scripted(['{"utterance": ""}', '```json\n{"utterance": "hello"}\n```'])
    [out] = generate(backend, [req(seed=10)])
    assert out == Say(utterance="hello")
    assert backend.seeds == [[10], [11]]


def test_generate_only_retries_failures_and_keeps_order():
    backend = Scripted(['{"utterance": "a"}', "not json", '{"utterance": "b"}'])
    a, b = generate(backend, [req(), req()])
    assert (a.utterance, b.utterance) == ("a", "b")
    assert backend.seeds == [[None, None], [None]]


def test_generate_gives_up_after_attempts():
    backend = Scripted([RuntimeError("boom")] * 3)
    [out] = generate(backend, [req()], attempts=3)
    assert isinstance(out, GenerationError) and "boom" in str(out)


def test_load_backend_rejects_unknown_spec():
    with pytest.raises(ValueError):
        load_backend("hf:gpt2")


# ─────────────────────────────────────────
# OpenRouter, against a fake OpenAI client
# ─────────────────────────────────────────


def response(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3),
    )


@pytest.fixture
def fake_openai(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    client = SimpleNamespace(calls=[], responses=[], init={})

    def create(**kwargs):
        client.calls.append(kwargs)
        return client.responses.pop(0)

    def make_client(**kwargs):
        client.init = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(openai, "OpenAI", make_client)
    return client


def test_openrouter_request_shape(fake_openai):
    fake_openai.responses = [response('{"utterance": "hi"}')]
    backend = OpenRouter("anthropic/claude-sonnet-4.5")
    request = Request([{"role": "user", "content": "x"}], Say, {"temperature": 0.5, "top_k": 20}, seed=7)
    [out] = generate(backend, [request])

    assert out.utterance == "hi"
    assert fake_openai.init["base_url"] == "https://openrouter.ai/api/v1"
    call = fake_openai.calls[0]
    assert (call["model"], call["temperature"], call["seed"]) == ("anthropic/claude-sonnet-4.5", 0.5, 7)
    assert call["extra_body"] == {"reasoning": {"effort": "low"}, "top_k": 20}
    fmt = call["response_format"]["json_schema"]
    assert fmt["strict"] and fmt["schema"] == Say.model_json_schema()
    assert backend.usage == {"calls": 1, "prompt_tokens": 10, "completion_tokens": 3}


def test_openrouter_truncated_output_is_an_error(fake_openai):
    fake_openai.responses = [response('{"utter', finish_reason="length")]
    [out] = generate(OpenRouter("m"), [req()], attempts=1)
    assert isinstance(out, GenerationError) and "truncated" in str(out)


def test_openrouter_needs_api_key(fake_openai, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouter("m")


def test_openrouter_response_without_choices_is_retryable(fake_openai):
    fake_openai.responses = [SimpleNamespace(choices=None, usage=None), response('{"utterance": "ok"}')]
    [out] = generate(OpenRouter("m"), [req()])
    assert out.utterance == "ok"


def test_openrouter_caps_requests_in_flight_across_concurrent_calls(fake_openai):
    backend = OpenRouter("m", max_workers=3)
    lock, in_flight, peak = threading.Lock(), [0], [0]

    def create(**kwargs):
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.02)
        with lock:
            in_flight[0] -= 1
        return response('{"utterance": "hi"}')

    backend.client.chat.completions.create = create
    with ThreadPoolExecutor(4) as pool:  # e.g. four episodes, each sending its own call
        list(pool.map(lambda _: backend.complete([req()] * 4), range(4)))
    assert peak[0] == 3
    assert backend.usage["calls"] == 16
