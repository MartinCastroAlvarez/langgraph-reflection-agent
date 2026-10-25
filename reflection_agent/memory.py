"""Reflexion's episodic memory: verbal lessons per task, optionally kept in a LangGraph store.

Within one run, lessons live in the graph state and every new attempt sees all of them.
With a ``BaseStore`` (``InMemoryStore``, or a persistent one such as
``SqliteStore``/``PostgresStore``), lessons also survive the run: the next run on the same
task key starts with them. Lessons are model output, so they are capped in number and
length and are only ever put into prompts inside ``<data>`` blocks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime

from langgraph.store.base import BaseStore

__all__ = ["MAX_LESSON_CHARS", "MAX_LESSONS", "LessonMemory", "task_key"]

MAX_LESSONS = 10
MAX_LESSON_CHARS = 500
NAMESPACE = ("reflexion", "lessons")
_KEY = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


def task_key(task: str, name: str | None = None) -> str:
    """A stable store key: ``name`` when it's a safe identifier, else a hash of the task."""
    if name is not None:
        if not _KEY.match(name):
            raise ValueError("a task key is 1-64 chars of a-z, 0-9, '.', '_' or '-'")
        return name
    digest = hashlib.sha256(" ".join(task.split()).lower().encode()).hexdigest()
    return f"task-{digest[:16]}"


def clean(lessons: list[str]) -> list[str]:
    """Strip, cut, de-duplicate and keep the most recent :data:`MAX_LESSONS`."""
    out: list[str] = []
    for lesson in lessons:
        text = " ".join(str(lesson).split())[:MAX_LESSON_CHARS]
        if text and text not in out:
            out.append(text)
    return out[-MAX_LESSONS:]


@dataclass(frozen=True)
class LessonMemory:
    store: BaseStore | None = None

    async def recall(self, key: str) -> list[str]:
        if self.store is None:
            return []
        item = await self.store.aget(NAMESPACE, key)
        if item is None:
            return []
        lessons = item.value.get("lessons", [])
        return clean([str(x) for x in lessons]) if isinstance(lessons, list) else []

    async def remember(self, key: str, lessons: list[str]) -> None:
        if self.store is None:
            return
        await self.store.aput(
            NAMESPACE,
            key,
            {"lessons": clean(lessons), "updated": datetime.now(UTC).isoformat()},
        )
