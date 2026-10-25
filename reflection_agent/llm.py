"""Chat model factory: local ollama by default, Anthropic optional, a fake for offline runs."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

from langchain_core.language_models import BaseChatModel

from reflection_agent.config import Settings
from reflection_agent.fake import FakeReflectionModel

__all__ = ["ModelConfigError", "is_loopback_url", "make_model"]


class ModelConfigError(RuntimeError):
    """The configured provider cannot be used as configured."""


def is_loopback_url(url: str) -> bool:
    host = urlparse(url if "://" in url else f"http://{url}").hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_model(
    settings: Settings, *, json_mode: bool = False, temperature: float | None = None
) -> BaseChatModel:
    """Build a chat model. ``json_mode`` asks ollama for JSON-only output (the critic, the
    Reflexion actor and revisor); ``temperature`` overrides the setting (LATS samples
    its candidates at a higher temperature so they differ)."""
    temp = settings.temperature if temperature is None else temperature
    if settings.provider == "fake":
        return FakeReflectionModel(latency=settings.fake_latency)
    if settings.provider == "ollama":
        if not settings.allow_remote_ollama and not is_loopback_url(settings.ollama_host):
            raise ModelConfigError(
                f"OLLAMA_HOST={settings.ollama_host!r} is not loopback; set "
                "REFLECTION_ALLOW_REMOTE_OLLAMA=1 to send prompts to another machine"
            )
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.model,
            base_url=settings.ollama_host,
            temperature=temp,
            reasoning=False,
            format="json" if json_mode else None,
        )
    if settings.provider == "anthropic":  # pragma: no cover - needs the optional extra + a key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ModelConfigError("ANTHROPIC_API_KEY is not set")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ModelConfigError(
                "pip install 'langgraph-reflection-agent[anthropic]' to use Anthropic"
            ) from exc
        return ChatAnthropic(model=settings.model, temperature=temp)
    raise ModelConfigError(f"unknown provider {settings.provider!r}")
