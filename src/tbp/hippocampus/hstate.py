# Copyright 2025 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""HState: Latent Hippocampal State Representation.

This module implements HState, a universal latent state representation that
abstracts the hippocampal encoding beyond purely spatial events. HState is
compatible with:
- Successor Representation (SR) theory
- Tolman–Eichenbaum Machine (TEM) principles
- Thousand Brains Theory / Monty architecture
- Abstract reasoning and creativity

Each HState corresponds to a CA3 attractor pattern and can encode ANY
relational context, not just spatial locations.

Key concepts:
- HState is separate from SpatialEvent (input) and raw cortical SDRs (output)
- Each HState has a unique ID derived from its CA3 pattern hash
- HState can optionally reference spatial coordinates OR abstract context
- EC basis codes (grid-like embeddings) are stored for generalization
- Timestamps and sequence indices support temporal ordering

Example:
    >>> # Create an HState from CA3 encoding
    >>> hstate = HState.from_ca3_pattern(
    ...     ca3_pattern=np.array([0, 1, 0, 1, ...]),
    ...     basis_vector=basis_code.encode(location),
    ...     timestamp=time.time(),
    ... )
    >>> # Access properties
    >>> print(hstate.id)  # Unique identifier
    >>> print(hstate.basis_vector)  # EC embedding
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from tbp.hippocampus.types import SpatialEvent


@dataclass
class HState:
    """Latent hippocampal state representation.

    HState represents the abstract state encoded by the hippocampus at a
    particular moment. It corresponds to a CA3 attractor pattern but is
    designed to be domain-agnostic, supporting both spatial and abstract
    relational reasoning.

    Attributes:
        id: Unique identifier (hash of CA3 pattern). Read-only after creation.
        ca3_pattern_hash: Tuple of active indices in CA3 pattern (hashable).
        basis_vector: EC basis code embedding (grid-like, generalizable).
            Shape depends on basis encoding used.
        timestamp: Unix timestamp when this state was created.
        sequence_index: Position in current sequence (0-indexed). None if not
            part of a sequence.
        spatial_location: Optional 3D spatial coordinates [x, y, z] if this
            state has spatial grounding.
        spatial_orientation: Optional 3x3 rotation matrix if spatially grounded.
        context_tag: Optional abstract/task context identifier (e.g., "navigation",
            "object_recognition", "planning").
        source_event: Optional reference to the SpatialEvent that generated this
            state (for provenance tracking).
        confidence: Confidence score [0, 1] for this state encoding.
        extra: Additional metadata dictionary for extensibility.

    Example:
        >>> hstate = HState(
        ...     ca3_pattern_hash=(10, 25, 47, 89, 156),
        ...     basis_vector=np.random.randn(64),
        ...     timestamp=time.time(),
        ...     spatial_location=np.array([1.0, 2.0, 0.5]),
        ...     context_tag="navigation",
        ... )
    """

    ca3_pattern_hash: Tuple[int, ...]
    basis_vector: np.ndarray
    timestamp: float
    sequence_index: Optional[int] = None
    spatial_location: Optional[np.ndarray] = None
    spatial_orientation: Optional[np.ndarray] = None
    context_tag: Optional[str] = None
    source_event: Optional[SpatialEvent] = None
    confidence: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    # Computed field for unique ID
    _id: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Validate and compute derived fields."""
        # Convert basis_vector to numpy array
        self.basis_vector = np.asarray(self.basis_vector, dtype=np.float64)

        # Validate spatial arrays if provided
        if self.spatial_location is not None:
            self.spatial_location = np.asarray(self.spatial_location, dtype=np.float64)
            if self.spatial_location.shape != (3,):
                raise ValueError(
                    f"spatial_location must have shape (3,), got {self.spatial_location.shape}"
                )

        if self.spatial_orientation is not None:
            self.spatial_orientation = np.asarray(self.spatial_orientation, dtype=np.float64)
            if self.spatial_orientation.shape != (3, 3):
                raise ValueError(
                    f"spatial_orientation must have shape (3, 3), "
                    f"got {self.spatial_orientation.shape}"
                )

        # Compute ID from pattern hash
        self._id = self._compute_id()

    def _compute_id(self) -> str:
        """Compute unique ID from CA3 pattern hash.

        Uses SHA-256 hash of pattern indices for collision resistance.
        """
        pattern_bytes = str(self.ca3_pattern_hash).encode("utf-8")
        return hashlib.sha256(pattern_bytes).hexdigest()[:16]

    @property
    def id(self) -> str:
        """Unique identifier for this HState."""
        if self._id is None:
            self._id = self._compute_id()
        return self._id

    @property
    def is_spatial(self) -> bool:
        """Whether this state has spatial grounding."""
        return self.spatial_location is not None

    @property
    def is_abstract(self) -> bool:
        """Whether this state is abstract (no spatial grounding)."""
        return self.spatial_location is None

    @property
    def n_active_cells(self) -> int:
        """Number of active cells in the CA3 pattern."""
        return len(self.ca3_pattern_hash)

    @classmethod
    def from_ca3_pattern(
            cls,
            ca3_pattern: np.ndarray,
            basis_vector: np.ndarray,
            timestamp: float,
            threshold: float = 0.5,
            **kwargs,
    ) -> HState:
        """Create HState from a CA3 pattern array.

        Args:
            ca3_pattern: Full CA3 pattern array (can be binary or continuous).
            basis_vector: EC basis code embedding.
            timestamp: Unix timestamp for this state.
            threshold: Threshold for considering a cell active (default 0.5).
            **kwargs: Additional arguments passed to HState constructor.

        Returns:
            New HState instance.
        """
        active_indices = np.where(ca3_pattern > threshold)[0]
        ca3_pattern_hash = tuple(active_indices.tolist())

        return cls(
            ca3_pattern_hash=ca3_pattern_hash,
            basis_vector=basis_vector,
            timestamp=timestamp,
            **kwargs,
        )

    @classmethod
    def from_spatial_event(
            cls,
            event: SpatialEvent,
            ca3_pattern: np.ndarray,
            basis_vector: np.ndarray,
            threshold: float = 0.5,
            context_tag: Optional[str] = None,
    ) -> HState:
        """Create HState from a SpatialEvent and its CA3 encoding.

        Args:
            event: Source SpatialEvent.
            ca3_pattern: CA3 pattern array encoding this event.
            basis_vector: EC basis code embedding.
            threshold: Threshold for considering a cell active.
            context_tag: Optional context identifier.

        Returns:
            New HState instance with spatial grounding.
        """
        return cls.from_ca3_pattern(
            ca3_pattern=ca3_pattern,
            basis_vector=basis_vector,
            timestamp=event.timestamp,
            threshold=threshold,
            spatial_location=event.location.copy(),
            spatial_orientation=event.orientation.copy(),
            source_event=event,
            confidence=event.confidence,
            context_tag=context_tag,
        )

    def to_pattern(self, n_cells: int) -> np.ndarray:
        """Reconstruct binary CA3 pattern from hash.

        Args:
            n_cells: Total number of CA3 cells.

        Returns:
            Binary pattern array of shape (n_cells,).
        """
        pattern = np.zeros(n_cells, dtype=np.float32)
        for idx in self.ca3_pattern_hash:
            if 0 <= idx < n_cells:
                pattern[idx] = 1.0
        return pattern

    def distance_to(self, other: HState, metric: str = "basis") -> float:
        """Compute distance to another HState.

        Args:
            other: Another HState to compare.
            metric: Distance metric to use:
                - "basis": Euclidean distance in basis space (default)
                - "pattern": Hamming distance of CA3 patterns
                - "spatial": Euclidean distance in spatial coordinates

        Returns:
            Distance value (lower = more similar).

        Raises:
            ValueError: If metric is "spatial" but one state is not spatial.
        """
        if metric == "basis":
            return float(np.linalg.norm(self.basis_vector - other.basis_vector))

        elif metric == "pattern":
            # Hamming distance (symmetric difference of active indices)
            self_set = set(self.ca3_pattern_hash)
            other_set = set(other.ca3_pattern_hash)
            symmetric_diff = self_set.symmetric_difference(other_set)
            return float(len(symmetric_diff))

        elif metric == "spatial":
            if not self.is_spatial or not other.is_spatial:
                raise ValueError("Both states must be spatial for spatial distance")
            # Type narrowing: we've verified both are not None above
            assert self.spatial_location is not None
            assert other.spatial_location is not None
            return float(np.linalg.norm(self.spatial_location - other.spatial_location))

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def overlap_with(self, other: HState) -> float:
        """Compute pattern overlap with another HState.

        Uses Jaccard similarity of active CA3 indices.

        Args:
            other: Another HState to compare.

        Returns:
            Overlap score [0, 1] where 1 = identical patterns.
        """
        self_set = set(self.ca3_pattern_hash)
        other_set = set(other.ca3_pattern_hash)

        if not self_set and not other_set:
            return 1.0  # Both empty = identical

        intersection = len(self_set.intersection(other_set))
        union = len(self_set.union(other_set))

        return intersection / union if union > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize HState to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        result = {
            "id": self.id,
            "ca3_pattern_hash": list(self.ca3_pattern_hash),
            "basis_vector": self.basis_vector.tolist(),
            "timestamp": self.timestamp,
            "sequence_index": self.sequence_index,
            "context_tag": self.context_tag,
            "confidence": self.confidence,
            "extra": self.extra,
        }

        if self.spatial_location is not None:
            result["spatial_location"] = self.spatial_location.tolist()
        if self.spatial_orientation is not None:
            result["spatial_orientation"] = self.spatial_orientation.tolist()

        # Note: source_event is not serialized to avoid circular references

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HState:
        """Deserialize HState from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Reconstructed HState instance.
        """
        spatial_location = None
        spatial_orientation = None

        if "spatial_location" in data:
            spatial_location = np.array(data["spatial_location"])
        if "spatial_orientation" in data:
            spatial_orientation = np.array(data["spatial_orientation"])

        return cls(
            ca3_pattern_hash=tuple(data["ca3_pattern_hash"]),
            basis_vector=np.array(data["basis_vector"]),
            timestamp=data["timestamp"],
            sequence_index=data.get("sequence_index"),
            spatial_location=spatial_location,
            spatial_orientation=spatial_orientation,
            context_tag=data.get("context_tag"),
            confidence=data.get("confidence", 1.0),
            extra=data.get("extra", {}),
        )

    def __hash__(self) -> int:
        """Hash based on CA3 pattern (immutable part of identity)."""
        return hash(self.ca3_pattern_hash)

    def __eq__(self, other: object) -> bool:
        """Equality based on CA3 pattern hash."""
        if not isinstance(other, HState):
            return NotImplemented
        return self.ca3_pattern_hash == other.ca3_pattern_hash

    def __repr__(self) -> str:
        spatial_info = f", loc={self.spatial_location}" if self.is_spatial else ""
        context_info = f", ctx={self.context_tag}" if self.context_tag else ""
        return (
            f"HState(id={self.id[:8]}..., "
            f"n_active={self.n_active_cells}"
            f"{spatial_info}{context_info})"
        )
