"""Episodic memory buffer for hippocampus.

Stores and retrieves spatial events. No dependencies on tbp.monty.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable

from tbp.hippocampus.types import SpatialEvent


class EpisodicMemory:
    """Thread-safe episodic memory buffer for spatial events.

    Stores events in a circular buffer with support for querying
    by recency, source, or custom filters.

    Attributes:
        max_size: Maximum number of events to retain.
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize episodic memory.

        Args:
            max_size: Maximum events to store. Oldest are discarded
                when capacity is exceeded.
        """
        self.max_size = max_size
        self._buffer: deque[SpatialEvent] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._total_received = 0

    def store(self, event: SpatialEvent) -> None:
        """Store a single event.

        Args:
            event: SpatialEvent to store.
        """
        with self._lock:
            self._buffer.append(event)
            self._total_received += 1

    def store_batch(self, events: list[SpatialEvent]) -> None:
        """Store multiple events.

        Args:
            events: List of SpatialEvents to store.
        """
        with self._lock:
            self._buffer.extend(events)
            self._total_received += len(events)

    def get_recent(self, n: int | None = None) -> list[SpatialEvent]:
        """Retrieve most recent events.

        Args:
            n: Number of events. If None, return all.

        Returns:
            List of events, most recent last.
        """
        with self._lock:
            if n is None or n >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-n:]

    def get_all(self) -> list[SpatialEvent]:
        """Retrieve all stored events."""
        return self.get_recent(None)

    def get_by_source(self, source_id: str) -> list[SpatialEvent]:
        """Retrieve events from a specific source.

        Args:
            source_id: Source identifier to filter by.

        Returns:
            List of events from that source.
        """
        with self._lock:
            return [e for e in self._buffer if e.source_id == source_id]

    def get_by_object(self, object_id) -> list[SpatialEvent]:
        """Retrieve events associated with a specific object.

        Args:
            object_id: Object identifier to filter by.

        Returns:
            List of events for that object.
        """
        with self._lock:
            return [e for e in self._buffer if e.object_id == object_id]

    def query(
        self, predicate: Callable[[SpatialEvent], bool]
    ) -> list[SpatialEvent]:
        """Query events using a custom predicate.

        Args:
            predicate: Function that returns True for events to include.

        Returns:
            List of matching events.
        """
        with self._lock:
            return [e for e in self._buffer if predicate(e)]

    def flush(self) -> list[SpatialEvent]:
        """Clear buffer and return all events.

        Returns:
            List of all events that were stored.
        """
        with self._lock:
            events = list(self._buffer)
            self._buffer.clear()
            return events

    def clear(self) -> None:
        """Clear buffer without returning events."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        """Return current number of stored events."""
        with self._lock:
            return len(self._buffer)

    @property
    def total_received(self) -> int:
        """Total events ever stored (including discarded)."""
        with self._lock:
            return self._total_received

    def __repr__(self) -> str:
        return (
            f"EpisodicMemory(size={len(self)}, "
            f"max_size={self.max_size}, "
            f"total_received={self.total_received})"
        )
