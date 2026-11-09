"""Central task registry."""

from __future__ import annotations

from functools import lru_cache

from agenteval.tasks.base import Task
from agenteval.tasks.booking import booking_tasks
from agenteval.tasks.coding import coding_tasks
from agenteval.tasks.support import support_tasks

DOMAINS: tuple[str, ...] = ("booking", "support", "coding")


@lru_cache(maxsize=1)
def _all_tasks() -> tuple[Task, ...]:
    tasks = (*booking_tasks(), *support_tasks(), *coding_tasks())
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate task IDs in registry")
    return tasks


def load_tasks(domain: str | None = None, ids: list[str] | None = None) -> list[Task]:
    """Load tasks, optionally filtered by domain and/or explicit IDs."""
    tasks = list(_all_tasks())
    if domain and domain != "all":
        if domain not in DOMAINS:
            raise ValueError(f"Unknown domain '{domain}'. Choose from {DOMAINS}")
        tasks = [t for t in tasks if t.domain == domain]
    if ids:
        wanted = set(ids)
        tasks = [t for t in tasks if t.id in wanted]
    return tasks


def get_task(task_id: str) -> Task:
    for t in _all_tasks():
        if t.id == task_id:
            return t
    raise KeyError(f"Unknown task '{task_id}'")
