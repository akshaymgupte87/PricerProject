"""Configuration helpers for the local LLM serving provider."""

from __future__ import annotations

import os


SUPPORTED_PROVIDERS = {"ollama", "vllm", "gateway"}


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
    if provider() == "gateway":
        return os.getenv(
            "PRICER_LLM_GATEWAY_API_BASE", "http://localhost:4000/v1"
        ).rstrip("/")
    if provider() == "vllm":
        return os.getenv("VLLM_API_BASE", "http://localhost:8001/v1").rstrip("/")
    return os.getenv("OLLAMA_API_BASE", "http://localhost:11434").rstrip("/")


def model_name(value: str) -> str:
    """Qualify an upstream model or map it to the configured gateway alias."""
    value = value.strip()
    if provider() == "gateway":
        alias = os.getenv("PRICER_LLM_GATEWAY_MODEL", "pricer-local").strip()
        return f"openai/{alias}"
    prefix = "ollama" if provider() == "ollama" else "openai"
    if value.startswith(("ollama/", "openai/")):
        return value
    return f"{prefix}/{value}"


def api_key() -> str | None:
    """Return the client credential for the gateway; direct providers need none."""
    if provider() == "gateway":
        value = os.getenv("PRICER_LLM_API_KEY", "").strip()
        return value or None
    return None
