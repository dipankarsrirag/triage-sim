from abc import ABC, abstractmethod
from typing import List, Optional, Type, Union

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

from triagesim.config import OPENROUTER_API_KEY
from triagesim.core.output_schema import BaseAgentOutput


class BaseLLM(ABC):
    """
    Abstract interface for ALL language model backends.
    Agents will call `.generate()` to produce structured outputs.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[str, BaseAgentOutput]:
        """
        Generate a response given a prompt.

        This method must return either:
        - raw text (for unstructured outputs), or
        - a structured Pydantic model (if output_type was provided).

        The calling agent assumes the correct model/schema is applied
        via the backend's `output_type` during instantiation.

        Args:
            prompt: the model input text
            max_tokens: optional token cap
            stop: optional stop sequences
            **kwargs: backend-specific overrides

        Returns:
            Model output (string or Pydantic output model)
        """
        pass


class OpenRouterLLM(BaseLLM):
    """
    LLM backend using OpenRouter via the `pydantic_ai` SDK.

    This wraps an `Agent` configured with a given output model type
    so that `.generate()` returns a structured output or text.

    Args:
        model_name: OpenRouter model specifier (e.g., "google/gemini-3-pro-preview")
        output_type: a subclass of BaseAgentOutput Pydantic model
                     that the Agent will produce directly.

    Usage:
        llm = OpenRouterLLM("google/gemini-3-pro-preview", output_type=NurseOutput)
        nurse_action = llm.generate(prompt)
    """

    def __init__(
        self,
        model_name: str,
        output_type: Type[BaseAgentOutput],
        **agent_kwargs,
    ):
        """
        Create an Agent configured to produce the right output type.

        The API key is resolved from triagesim.config.OPENROUTER_API_KEY.

        Args:
            model_name: the OpenRouter model identifier
            output_type: Pydantic model to enforce output structure
            agent_kwargs: any extra fields passed to Agent()
        """
        # Instantiate the provider with the API key
        provider = OpenRouterProvider(api_key=OPENROUTER_API_KEY)
        settings = OpenRouterModelSettings(openrouter_reasoning={"effort": "low"})
        # Create the model via pydantic_ai
        model = OpenRouterModel(model_name, provider=provider)

        # Create the agent that enforces the output_type schema
        self.agent = Agent(
            model, output_type=output_type, model_settings=settings, **agent_kwargs
        )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[str, BaseAgentOutput]:
        """
        Run a synchronous request against the configured agent.

        Because the agent is configured with `output_type`, this method
        should return a structured Pydantic model or a simple string

        Args:
            prompt: the text prompt
            max_tokens: optionally limit tokens
            stop: optional stop sequences
            **kwargs: passed to agent.run_sync()

        Returns:
            Model output from the LLM call
        """
        # Build kwargs for pydantic_ai if provided
        run_kwargs: dict[str, object] = {}
        if max_tokens is not None:
            run_kwargs["max_tokens"] = max_tokens
        if stop is not None:
            run_kwargs["stop"] = stop

        run_kwargs.update(kwargs)

        # Run the model
        result = self.agent.run_sync(prompt, **run_kwargs)

        # Return whatever the parsed agent returns; if output_type is set,
        # the result should be an instance of that Pydantic model.
        return result.output
