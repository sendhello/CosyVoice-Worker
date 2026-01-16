from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class WorkItem:
    """Represents a unit of work parsed from an SQS message.

    Attributes
    -----------
    payload: Dict[str, Any]
        JSON payload describing the task. The expected structure is documented
        in the worker README section. At minimum it should contain a `mode`
        and fields required by the particular CosyVoice inference path.
    receipt_handle: str
        SQS receipt handle to acknowledge (delete) after successful processing.
    message_id: str
        SQS message ID for logging/diagnostics.
    """

    payload: Dict[str, Any]
    receipt_handle: str
    message_id: str


class InternalQueue:
    """Thread-safe internal queue for buffering `WorkItem`s.

    This is a thin wrapper over `queue.Queue` with a convenience method to
    gather a batch of items waiting up to a specified window duration.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._q: queue.Queue[WorkItem] = queue.Queue(maxsize=maxsize)

    def put(self, item: WorkItem, block: bool = True, timeout: Optional[float] = None) -> None:
        self._q.put(item, block=block, timeout=timeout if timeout is not None else 0.0)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> WorkItem:
        return self._q.get(block=block, timeout=timeout)

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()

    def gather(self, max_items: int, window_sec: float) -> List[WorkItem]:
        """Gather up to `max_items` items, waiting up to `window_sec` seconds.

        Always returns at least one item if the queue is not empty at call time,
        unless `window_sec` elapses and nothing could be fetched.
        """
        items: List[WorkItem] = []
        end_ts = time.monotonic() + max(0.0, window_sec)
        # Try to get the first item quickly
        try:
            first = self._q.get(timeout=max(0.0, window_sec))
            items.append(first)
        except Exception:
            return items
        # Drain additional items until we reach max_items or time window ends
        while len(items) < max(1, max_items) and time.monotonic() < end_ts:
            try:
                remaining = end_ts - time.monotonic()
                next_item = self._q.get(timeout=max(0.0, min(0.01, remaining)))
                items.append(next_item)
            except Exception:
                break
        return items
