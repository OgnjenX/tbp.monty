"""Entorhinal Cortex implementation.

Processes spatial events using grid cells and place cells.
No dependencies on tbp.monty.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from tbp.hippocampus.memory import EpisodicMemory
from tbp.hippocampus.types import GridCell, PlaceCell, SpatialEvent

logger = logging.getLogger(__name__)


class EntorhinalCortex:
    """Entorhinal Cortex: processes spatial events.

    This is the main entry point for the hippocampus. It receives
    SpatialEvents and encodes them using grid and place cells.

    The EC can:
    - Receive spatial events from any source (via adapters)
    - Encode locations using grid/place cell populations
    - Store events in episodic memory
    - Notify listeners when events are received

    Attributes:
        memory: EpisodicMemory for storing events.
        place_cells: List of place cells.
        grid_cells: List of grid cells.
    """

    def __init__(
        self,
        memory: EpisodicMemory | None = None,
        n_place_cells: int = 100,
        n_grid_cells: int = 50,
        place_cell_radius: float = 0.1,
        grid_cell_spacings: list[float] | None = None,
        spatial_extent: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Initialize Entorhinal Cortex.

        Args:
            memory: EpisodicMemory instance. Created if None.
            n_place_cells: Number of place cells to create.
            n_grid_cells: Number of grid cells to create.
            place_cell_radius: Radius of place cell fields.
            grid_cell_spacings: List of grid spacings. If None, uses
                default geometric series.
            spatial_extent: Extent of space for random place cell centers.
        """
        self.memory = memory or EpisodicMemory()
        self._listeners: list[Callable[[SpatialEvent], None]] = []
        self._event_count = 0

        # Initialize place cells with random centers
        self.place_cells: list[PlaceCell] = []
        for i in range(n_place_cells):
            center = np.random.uniform(
                low=-np.array(spatial_extent) / 2,
                high=np.array(spatial_extent) / 2,
            )
            self.place_cells.append(
                PlaceCell(cell_id=i, center=center, radius=place_cell_radius)
            )

        # Initialize grid cells with varying spacings
        if grid_cell_spacings is None:
            # Default: geometric series of spacings
            grid_cell_spacings = [0.1 * (1.4**i) for i in range(5)]

        self.grid_cells: list[GridCell] = []
        for i in range(n_grid_cells):
            spacing = grid_cell_spacings[i % len(grid_cell_spacings)]
            orientation = np.random.uniform(0, np.pi / 3)  # 0-60 degrees
            phase = np.random.uniform(-spacing / 2, spacing / 2, size=2)
            self.grid_cells.append(
                GridCell(
                    cell_id=i,
                    spacing=spacing,
                    orientation=orientation,
                    phase=phase,
                )
            )

    def receive_event(self, event: SpatialEvent) -> dict[str, np.ndarray]:
        """Process an incoming spatial event.

        This is the main entry point. When an event is received:
        1. Encode location using place and grid cells
        2. Store in episodic memory
        3. Notify listeners

        Args:
            event: SpatialEvent to process.

        Returns:
            Dictionary with:
                - 'place_activations': array of place cell activations
                - 'grid_activations': array of grid cell activations
        """
        self._event_count += 1

        # Compute place cell activations
        place_activations = np.array(
            [pc.compute_activation(event.location) for pc in self.place_cells]
        )

        # Compute grid cell activations
        grid_activations = np.array(
            [gc.compute_activation(event.location) for gc in self.grid_cells]
        )

        # Store in episodic memory
        self.memory.store(event)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.warning(f"Listener error: {e}")

        logger.debug(
            f"EC received event {self._event_count}: "
            f"loc={event.location}, conf={event.confidence:.2f}"
        )

        return {
            "place_activations": place_activations,
            "grid_activations": grid_activations,
        }

    def receive_events(
        self, events: list[SpatialEvent]
    ) -> list[dict[str, np.ndarray]]:
        """Process multiple events.

        Args:
            events: List of SpatialEvents.

        Returns:
            List of activation dictionaries.
        """
        return [self.receive_event(e) for e in events]

    def add_listener(self, callback: Callable[[SpatialEvent], None]) -> None:
        """Add a listener to be notified of new events.

        Args:
            callback: Function called with each new event.
        """
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[SpatialEvent], None]) -> None:
        """Remove a listener.

        Args:
            callback: Function to remove.
        """
        if callback in self._listeners:
            self._listeners.remove(callback)

    def get_place_representation(self, location: np.ndarray) -> np.ndarray:
        """Get place cell population vector for a location.

        Args:
            location: 3D position.

        Returns:
            Array of place cell activations.
        """
        return np.array(
            [pc.compute_activation(location) for pc in self.place_cells]
        )

    def get_grid_representation(self, location: np.ndarray) -> np.ndarray:
        """Get grid cell population vector for a location.

        Args:
            location: 3D position.

        Returns:
            Array of grid cell activations.
        """
        return np.array(
            [gc.compute_activation(location) for gc in self.grid_cells]
        )

    def reset(self) -> None:
        """Reset activations (not memory or cells)."""
        for pc in self.place_cells:
            pc.activation = 0.0
        for gc in self.grid_cells:
            gc.activation = 0.0
        self._event_count = 0

    @property
    def event_count(self) -> int:
        """Total events received since last reset."""
        return self._event_count

    def __repr__(self) -> str:
        return (
            f"EntorhinalCortex("
            f"place_cells={len(self.place_cells)}, "
            f"grid_cells={len(self.grid_cells)}, "
            f"events={self._event_count})"
        )
