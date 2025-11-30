"""Core types for hippocampus module.

These types have NO dependencies on tbp.monty and define the interface
for spatial events that the hippocampus processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SpatialEvent:
    """A spatial event received by the hippocampus.

    This is the core input type for hippocampal processing. It represents
    a sensory event with spatial context (location and orientation).

    Attributes:
        timestamp: Unix timestamp when the event occurred.
        location: 3D position vector [x, y, z]. Reference frame is defined
            by the source (e.g., body-relative from Monty).
        orientation: 3x3 rotation matrix representing orientation.
            Columns are orthonormal basis vectors.
        features: Dictionary of features observed at this location.
            Keys and values are source-dependent.
        source_id: Identifier of the source that generated this event
            (e.g., learning module ID).
        confidence: Confidence score for this event, typically [0, 1].
        event_type: Type of event ('observation', 'hypothesis', etc.).
        object_id: Optional object identifier if known.
        extra: Additional metadata.

    Example:
        >>> event = SpatialEvent(
        ...     timestamp=time.time(),
        ...     location=np.array([0.1, 0.2, 0.3]),
        ...     orientation=np.eye(3),
        ...     features={"color": "red"},
        ...     source_id="lm_0",
        ...     confidence=0.95,
        ... )
    """

    timestamp: float
    location: np.ndarray  # shape (3,)
    orientation: np.ndarray  # shape (3, 3)
    source_id: str
    confidence: float
    features: dict[str, Any] = field(default_factory=dict)
    event_type: str = "observation"
    object_id: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and convert arrays."""
        self.location = np.asarray(self.location, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)

        if self.location.shape != (3,):
            raise ValueError(f"location must have shape (3,), got {self.location.shape}")
        if self.orientation.shape != (3, 3):
            raise ValueError(
                f"orientation must have shape (3, 3), got {self.orientation.shape}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "location": self.location.tolist(),
            "orientation": self.orientation.tolist(),
            "source_id": self.source_id,
            "confidence": self.confidence,
            "features": self.features,
            "event_type": self.event_type,
            "object_id": self.object_id,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialEvent:
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            location=np.array(data["location"]),
            orientation=np.array(data["orientation"]),
            source_id=data["source_id"],
            confidence=data["confidence"],
            features=data.get("features", {}),
            event_type=data.get("event_type", "observation"),
            object_id=data.get("object_id"),
            extra=data.get("extra", {}),
        )


@dataclass
class PlaceCell:
    """Represents a place cell with a preferred location.

    Attributes:
        cell_id: Unique identifier for this cell.
        center: Preferred location (3D).
        radius: Spatial extent of the place field.
        activation: Current activation level [0, 1].
    """

    cell_id: int
    center: np.ndarray  # shape (3,)
    radius: float
    activation: float = 0.0

    def compute_activation(self, location: np.ndarray) -> float:
        """Compute activation given a location.

        Uses Gaussian tuning curve centered on self.center.

        Args:
            location: 3D position to compute activation for.

        Returns:
            Activation level [0, 1].
        """
        distance = np.linalg.norm(location - self.center)
        self.activation = float(np.exp(-(distance**2) / (2 * self.radius**2)))
        return self.activation


@dataclass
class GridCell:
    """Represents a grid cell with periodic spatial tuning.

    Attributes:
        cell_id: Unique identifier for this cell.
        spacing: Distance between grid peaks.
        orientation: Grid orientation angle (radians).
        phase: Phase offset (2D).
        activation: Current activation level.
    """

    cell_id: int
    spacing: float
    orientation: float  # radians
    phase: np.ndarray  # shape (2,)
    activation: float = 0.0

    def compute_activation(self, location: np.ndarray) -> float:
        """Compute grid cell activation for a 2D location.

        Uses sum of three cosines at 60-degree angles.

        Args:
            location: 2D or 3D position (z is ignored if 3D).

        Returns:
            Activation level.
        """
        loc_2d = location[:2] - self.phase

        # Rotate to grid orientation
        cos_theta = np.cos(self.orientation)
        sin_theta = np.sin(self.orientation)
        rot = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        loc_rot = rot @ loc_2d

        # Sum of three cosines at 60-degree intervals
        angles = [0, np.pi / 3, 2 * np.pi / 3]
        activation = 0.0
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            activation += np.cos(2 * np.pi * np.dot(loc_rot, direction) / self.spacing)

        # Normalize to [0, 1]
        self.activation = float((activation / 3 + 1) / 2)
        return self.activation
