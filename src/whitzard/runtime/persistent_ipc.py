from __future__ import annotations

import secrets
from functools import partial
from multiprocessing.managers import BaseManager
from queue import Queue


class PersistentWorkerQueueManager(BaseManager):
    """Shared manager class for queue-backed persistent worker control."""


_QUEUE_OBJECTS: dict[str, object] = {}


def create_queue_method_names() -> tuple[str, str, str]:
    token = secrets.token_hex(8)
    return (
        f"get_command_queue_{token}",
        f"get_event_queue_{token}",
        f"get_log_queue_{token}",
    )


def _get_registered_queue(token: str):
    return _QUEUE_OBJECTS[token]


def register_parent_queues(
    *,
    command_method: str,
    event_method: str,
    log_method: str,
    command_queue: Queue[dict[str, object]],
    event_queue: Queue[dict[str, object]],
    log_queue: Queue[str],
) -> None:
    _QUEUE_OBJECTS[command_method] = command_queue
    _QUEUE_OBJECTS[event_method] = event_queue
    _QUEUE_OBJECTS[log_method] = log_queue
    PersistentWorkerQueueManager.register(
        command_method,
        callable=partial(_get_registered_queue, command_method),
    )
    PersistentWorkerQueueManager.register(
        event_method,
        callable=partial(_get_registered_queue, event_method),
    )
    PersistentWorkerQueueManager.register(
        log_method,
        callable=partial(_get_registered_queue, log_method),
    )


def register_client_queues(
    *,
    command_method: str,
    event_method: str,
    log_method: str,
) -> None:
    PersistentWorkerQueueManager.register(command_method)
    PersistentWorkerQueueManager.register(event_method)
    PersistentWorkerQueueManager.register(log_method)


def unregister_parent_queues(*method_names: str) -> None:
    for method_name in method_names:
        _QUEUE_OBJECTS.pop(method_name, None)
