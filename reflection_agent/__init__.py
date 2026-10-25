"""Reflection agents in LangGraph: basic reflection, Reflexion and LATS-lite.

>>> from reflection_agent import Settings, build_graph
>>> graph = build_graph("reflect", Settings(provider="fake"))
"""

from reflection_agent.config import Limits
from reflection_agent.config import Settings
from reflection_agent.graphs import build_graph
from reflection_agent.rubric import RUBRIC
from reflection_agent.rubric import Critique
from reflection_agent.sandbox import Sandbox
from reflection_agent.sandbox import SandboxResult
from reflection_agent.tasks import CODE_TASKS
from reflection_agent.tasks import CodeTask

__all__ = [
    "CODE_TASKS",
    "RUBRIC",
    "CodeTask",
    "Critique",
    "Limits",
    "Sandbox",
    "SandboxResult",
    "Settings",
    "build_graph",
]
__version__ = "0.1.0"
