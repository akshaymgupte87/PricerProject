import os
import re
from collections.abc import Callable

from litellm import completion

from agents.agent import Agent


def local_ollama_model(value: str) -> str:
    """Return the provider-qualified model name LiteLLM expects."""
    value = value.strip()
    return value if value.startswith("ollama/") else f"ollama/{value}"


DEFAULT_MODEL = local_ollama_model(
    os.getenv(
        "PRICER_SPECIALIST_MODEL",
        os.getenv("PRICER_QWEN_MODEL", "qwen3.6:latest"),
    )
)
DEFAULT_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
DEFAULT_BACKEND = os.getenv("PRICER_SPECIALIST_BACKEND", "ollama").strip().lower()


class SpecialistAgent(Agent):
    """Estimate prices with fine-tuned Qwen or an optional Ollama fallback."""

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(
        self,
        backend: str = DEFAULT_BACKEND,
        price_fn: Callable[[str], float] | None = None,
        model_name: str = DEFAULT_MODEL,
        api_base: str = DEFAULT_API_BASE,
        completion_fn: Callable = completion,
    ):
        backend = backend.strip().lower()
        if backend not in {"finetuned", "ollama"}:
            raise ValueError("Specialist backend must be 'finetuned' or 'ollama'.")
        self.backend = backend
        self.price_fn = price_fn
        self.MODEL = local_ollama_model(model_name)
        self.api_base = api_base
        self.completion_fn = completion_fn
        if self.backend == "finetuned":
            self.log("Specialist Agent is ready using local fine-tuned Qwen2.5")
        else:
            self.log(f"Specialist Agent is ready using Ollama fallback {self.MODEL}")

    @staticmethod
    def _extract_price(reply: str) -> float:
        cleaned = reply.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.?\d+", cleaned)
        if not match:
            raise ValueError(f"Local Qwen returned no numeric price: {reply!r}")
        result = float(match.group())
        if result <= 0:
            raise ValueError(f"Local Qwen returned a non-positive price: {result}")
        return result

    def _fine_tuned_price(self, description: str) -> float:
        if self.price_fn is None:
            from pricer_ephemeral import price

            self.price_fn = price
        return self.price_fn(description)

    def _ollama_price(self, description: str) -> float:
        response = self.completion_fn(
            model=self.MODEL,
            api_base=self.api_base,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You estimate the fair US retail value of products. "
                        "Return only one price in US dollars as a number."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Estimate the fair retail value of this product:\n\n{description}",
                },
            ],
            temperature=0,
            think=False,
            keep_alive="30m",
        )
        reply = response.choices[0].message.content or ""
        return self._extract_price(reply)

    def price(self, description: str) -> float:
        """Return a price from the configured specialist backend."""
        self.log(f"Specialist Agent is calling {self.backend} backend")
        result = (
            self._fine_tuned_price(description)
            if self.backend == "finetuned"
            else self._ollama_price(description)
        )
        if result <= 0:
            raise ValueError(f"Specialist returned a non-positive price: {result}")
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result
