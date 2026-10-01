"""Configuration helpers for the local LLM serving provider."""

from __future__ import annotations

import os


SUPPORTED_PROVIDERS = {"ollama", "vllm"}


def provider() -> str:
    value = os.getenv("PRICER_LLM_PROVIDER", "ollama").strip().lower()
    if value not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"PRICER_LLM_PROVIDER must be one of {sorted(SUPPORTED_PROVIDERS)}, got {value!r}"
        )
    return value


def api_base() -> str:
    configured = os.getenv("PRICER_LLM_API_BASE", "").strip()
    if configured:
        return configured.rstrip("/")
    if provider() == "vllm":
        return os.getenv("VLLM_API_BASE", "http://localhost:8001/v1").rstrip("/")
    return os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")


def model_name(value: str) -> str:
    """Qualify a model name for LiteLLM's Ollama or OpenAI-compatible provider."""
    value = value.strip()
    prefix = "ollama" if provider() == "ollama" else "openai"
    if value.startswith(("ollama/", "openai/")):
        return value
    return f"{prefix}/{value}"
