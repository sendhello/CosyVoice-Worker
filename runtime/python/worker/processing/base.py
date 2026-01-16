from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List


class Processor(ABC):
    """Abstract processing backend.

    Implementations should provide single-item and batch processing methods.
    The methods should be idempotent where reasonably possible since SQS can
    deliver messages at-least-once.
    """

    @abstractmethod
    def process_one(self, payload: Dict[str, Any]) -> None:
        """Process a single payload.

        Implementations may store results externally (e.g., upload to S3)
        according to fields of the payload (like output path/URL).
        """

    @abstractmethod
    def process_batch(self, payloads: Iterable[Dict[str, Any]]) -> None:
        """Process multiple payloads as a batch.

        The default expectation is to try to leverage model-level batching or
        shared resources to amortize costs.
        """
