"""Settings and loop limits, read from ``REFLECTION_*`` environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

__all__ = ["DEFAULT_MODELS", "PROVIDERS", "VARIANTS", "Limits", "Provider", "Settings", "Variant"]

Provider = Literal["ollama", "anthropic", "fake"]
PROVIDERS: tuple[Provider, ...] = ("ollama", "anthropic", "fake")
Variant = Literal["reflect", "reflexion", "lats"]
VARIANTS: tuple[Variant, ...] = ("reflect", "reflexion", "lats")

DEFAULT_MODELS: dict[Provider, str] = {
    "ollama": "qwen3:4b",
    "anthropic": "claude-sonnet-5-5",
    "fake": "fake-reflection",
}


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    return default if raw is None or raw == "" else int(raw)


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    return default if raw is None or raw == "" else float(raw)


@dataclass(frozen=True, slots=True)
class Limits:
    """Every loop is bounded by these, enforced by the graph and never by the model."""

    threshold: float = 8.0
    """Critic score (0-10) that ends basic reflection and LATS with ``status="passed"``."""
    max_iterations: int = 4
    """Basic reflection: drafts (the first one included)."""
    max_attempts: int = 3
    """Reflexion: trials, each draft -> tools -> revise -> evaluate (-> lesson)."""
    max_queries: int = 3
    """Reflexion: search queries plus calculations run per draft."""
    candidates: int = 3
    """LATS: children sampled per expansion (breadth)."""
    max_depth: int = 3
    """LATS: tree depth below the root."""
    max_expansions: int = 4
    """LATS: select -> expand -> reflect rounds."""
    exploration: float = 1.0
    """LATS: the UCT exploration weight."""

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 10:
            raise ValueError("threshold must be between 0 and 10")
        for name in (
            "max_iterations",
            "max_attempts",
            "candidates",
            "max_depth",
            "max_expansions",
        ):
            value = getattr(self, name)
            if not 1 <= value <= 20:
                raise ValueError(f"{name} must be between 1 and 20, got {value}")
        if not 0 <= self.max_queries <= 10:
            raise ValueError("max_queries must be between 0 and 10")
        if self.exploration < 0:
            raise ValueError("exploration must be >= 0")


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the three graphs and the CLI can be tuned with."""

    provider: Provider = "ollama"
    model: str = DEFAULT_MODELS["ollama"]
    ollama_host: str = "http://127.0.0.1:11434"
    allow_remote_ollama: bool = False
    temperature: float = 0.0
    limits: Limits = field(default_factory=Limits)
    corpus_dir: str | None = None
    sandbox_timeout: float = 5.0
    fake_latency: float = 0.0
    """Seconds per model call for the ``fake`` provider."""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env
        provider = e.get("REFLECTION_PROVIDER", "ollama").strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"REFLECTION_PROVIDER must be one of {PROVIDERS}")
        p: Provider = provider
        d = Limits()
        return cls(
            provider=p,
            model=e.get("REFLECTION_MODEL") or DEFAULT_MODELS[p],
            ollama_host=e.get("OLLAMA_HOST") or "http://127.0.0.1:11434",
            allow_remote_ollama=e.get("REFLECTION_ALLOW_REMOTE_OLLAMA", "") == "1",
            temperature=_float(e, "REFLECTION_TEMPERATURE", 0.0),
            limits=Limits(
                threshold=_float(e, "REFLECTION_THRESHOLD", d.threshold),
                max_iterations=_int(e, "REFLECTION_MAX_ITERATIONS", d.max_iterations),
                max_attempts=_int(e, "REFLECTION_MAX_ATTEMPTS", d.max_attempts),
                max_queries=_int(e, "REFLECTION_MAX_QUERIES", d.max_queries),
                candidates=_int(e, "REFLECTION_CANDIDATES", d.candidates),
                max_depth=_int(e, "REFLECTION_MAX_DEPTH", d.max_depth),
                max_expansions=_int(e, "REFLECTION_MAX_EXPANSIONS", d.max_expansions),
                exploration=_float(e, "REFLECTION_EXPLORATION", d.exploration),
            ),
            corpus_dir=e.get("REFLECTION_CORPUS_DIR") or None,
            sandbox_timeout=_float(e, "REFLECTION_SANDBOX_TIMEOUT", 5.0),
            fake_latency=_float(e, "REFLECTION_FAKE_LATENCY", 0.0),
        )

    @property
    def recursion_limit(self) -> int:
        """A backstop sized so the limits above always trip first."""
        lim = self.limits
        return 25 + 3 * lim.max_iterations + 6 * lim.max_attempts + 4 * lim.max_expansions
