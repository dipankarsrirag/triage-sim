"""
LLM backends: vLLM (in-process) and OpenRouter.

A backend takes a batch of chat requests and returns one raw completion per request, constrained
to the request's pydantic schema (vLLM: structured outputs; OpenRouter: `json_schema` response
format). `generate` parses and validates the completions and retries the ones that fail.
"""

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import cache
from typing import Any, Optional

from pydantic import BaseModel


class GenerationError(RuntimeError):
    """A request produced no valid output after all attempts."""


@dataclass
class Request:
    messages: list[dict[str, str]]
    output_type: type[BaseModel]
    sampling: dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None


@cache
def _schema(output_type: type[BaseModel]) -> dict:
    return output_type.model_json_schema()


def _json_span(text: str) -> str:
    """Strip anything around the outermost JSON object (e.g. markdown fences)."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if 0 <= start < end else text


class VLLM:
    """In-process vLLM engine. Each call to `complete` is one batched `LLM.chat`.

    Args:
        model: local path or Hugging Face id.
        chat_template_kwargs: passed to the chat template, e.g. {"enable_thinking": False}.
        **engine_args: passed to `vllm.LLM`, e.g. max_model_len=8192, gpu_memory_utilization=0.9.
    """

    def __init__(self, model: str, *, chat_template_kwargs: Optional[dict] = None, **engine_args):
        from vllm import LLM

        self.model = model
        self.chat_template_kwargs = chat_template_kwargs
        self.llm = LLM(model=model, **engine_args)
        # the model's generation_config (temperature, top_p, ...), as `vllm serve` would use
        self.defaults = self.llm.model_config.get_diff_sampling_param()
        self.usage: Counter = Counter()

    def complete(self, requests: list[Request]) -> list[str | Exception]:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        params = [
            SamplingParams(
                **{**self.defaults, **r.sampling},
                seed=r.seed,
                structured_outputs=StructuredOutputsParams(
                    json=_schema(r.output_type), disable_any_whitespace=True
                ),
            )
            for r in requests
        ]
        try:
            outputs = self.llm.chat(
                [r.messages for r in requests],
                params,
                use_tqdm=False,
                chat_template_kwargs=self.chat_template_kwargs,
            )
        except Exception as e:
            # one bad request (e.g. a prompt longer than max_model_len) fails the whole batch
            if len(requests) == 1:
                return [e]
            return [out for r in requests for out in self.complete([r])]

        results: list[str | Exception] = []
        for out, p in zip(outputs, params):
            completion = out.outputs[0]
            self.usage.update(
                calls=1, prompt_tokens=len(out.prompt_token_ids), completion_tokens=len(completion.token_ids)
            )
            if completion.finish_reason == "length":
                results.append(GenerationError(f"output truncated at max_tokens={p.max_tokens}"))
            else:
                results.append(completion.text)
        return results


# chat.completions.create arguments; any other sampling key (top_k, min_p, ...) goes in extra_body
_OPENAI_PARAMS = {"temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "stop"}


class OpenRouter:
    """OpenRouter chat completions, `max_workers` requests in flight at a time.

    The API key comes from `api_key` or OPENROUTER_API_KEY (environment or a .env file). The
    default `extra_body` asks reasoning models for low reasoning effort.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        max_workers: int = 16,
        extra_body: Optional[dict] = None,
    ):
        from dotenv import find_dotenv, load_dotenv
        from openai import OpenAI

        load_dotenv(find_dotenv(usecwd=True))
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter needs an API key: set OPENROUTER_API_KEY or pass api_key")

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=5)
        self.max_workers = max_workers
        self.extra_body = {"reasoning": {"effort": "low"}} if extra_body is None else extra_body
        self.usage: Counter = Counter()

    def _complete_one(self, r: Request) -> tuple[str | Exception, Counter]:
        kwargs = {k: v for k, v in r.sampling.items() if k in _OPENAI_PARAMS}
        extra = {k: v for k, v in r.sampling.items() if k not in _OPENAI_PARAMS}
        if r.seed is not None:
            kwargs["seed"] = r.seed
        usage = Counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=r.messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": r.output_type.__name__, "strict": True, "schema": _schema(r.output_type)},
                },
                extra_body={**self.extra_body, **extra},
                **kwargs,
            )
            usage.update(calls=1)
            if resp.usage:
                usage.update(prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens)
            choice = resp.choices[0]  # error payloads can come back without choices
            if choice.finish_reason == "length":
                return GenerationError(f"output truncated at max_tokens={kwargs.get('max_tokens')}"), usage
            return choice.message.content or "", usage
        except Exception as e:
            return e, usage

    def complete(self, requests: list[Request]) -> list[str | Exception]:
        with ThreadPoolExecutor(self.max_workers) as pool:
            done = list(pool.map(self._complete_one, requests))
        for _, usage in done:
            self.usage.update(usage)
        return [text for text, _ in done]


Backend = VLLM | OpenRouter


def load_backend(spec: str, **kwargs) -> Backend:
    """Build a backend from "vllm:<model>" or "openrouter:<model>"; kwargs go to its constructor."""
    kind, sep, model = spec.partition(":")
    if not sep or kind not in ("vllm", "openrouter"):
        raise ValueError(f"model spec must be 'vllm:<model>' or 'openrouter:<model>', got {spec!r}")
    return VLLM(model, **kwargs) if kind == "vllm" else OpenRouter(model, **kwargs)


def generate(backend: Backend, requests: list[Request], attempts: int = 3) -> list[BaseModel | GenerationError]:
    """Run requests through a backend, retrying (with a new seed) the ones that fail validation."""
    results: list[Any] = [None] * len(requests)
    todo = list(range(len(requests)))
    for attempt in range(attempts):
        batch = [requests[i] if requests[i].seed is None else replace(requests[i], seed=requests[i].seed + attempt) for i in todo]
        retry = []
        for i, text in zip(todo, backend.complete(batch)):
            try:
                if isinstance(text, Exception):
                    raise text
                results[i] = requests[i].output_type.model_validate_json(_json_span(text))
            except Exception as e:
                results[i] = GenerationError(f"{type(e).__name__}: {e}")
                retry.append(i)
        todo = retry
        if not todo:
            break
    return results
