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
    - Each HState has a unique ID derived from its DG/CA3 SDR indices
    - HState can optionally reference spatial coordinates OR abstract context
    - EC basis codes (grid-like embeddings) and CA1 SDRs are stored for
      generalization and decoding
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
        id: Unique identifier (hash of DG SDR indices). Read-only after creation.
        sdr_indices: Tuple of active indices in the sparse distributed
            representation from DG (primary code).
        ca1_indices: Optional tuple of active indices in the CA1 SDR used as
            decoded/clean readout for replay and planning.
        basis_vector: Optional EC basis code embedding (grid-like) used for
            interpolation/debugging, not required for core dynamics.
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
        ...     sdr_indices=(10, 25, 47, 89, 156),
        ...     ca1_indices=(5, 12, 89),
        ...     basis_vector=np.random.randn(64),
        ...     timestamp=time.time(),
        ...     spatial_location=np.array([1.0, 2.0, 0.5]),
        ...     context_tag="navigation",
        ... )
    """

    # Primary internal representation: sparse DG SDR indices
    sdr_indices: Tuple[int, ...]
    timestamp: float
    # Optional CA1 SDR indices (decoded / clean state)
    ca1_indices: Optional[Tuple[int, ...]] = None
    # Optional dense basis embedding for debugging/interpolation
    basis_vector: Optional[np.ndarray] = None
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
        # Convert basis_vector to numpy array if provided
        if self.basis_vector is not None:
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
        """Compute unique ID from SDR indices (pattern hash).

        Uses SHA-256 hash of indices for collision resistance.
        """
        pattern_bytes = str(self.sdr_indices).encode("utf-8")
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
        """Number of active cells in the SDR."""
        return len(self.sdr_indices)

    @classmethod
    def from_ca3_pattern(
            cls,
            ca3_pattern: np.ndarray,
            basis_vector: Optional[np.ndarray],
            timestamp: float,
            threshold: float = 0.5,
            **kwargs,
    ) -> HState:
        """Create HState from a CA3 pattern array.

        Args:
            ca3_pattern: Full SDR pattern array (binary or continuous).
            basis_vector: Optional EC basis code embedding.
            timestamp: Unix timestamp for this state.
            threshold: Threshold for considering a cell active (default 0.5).
            **kwargs: Additional arguments passed to HState constructor.

        Returns:
            New HState instance.
        """
        active_indices = np.where(ca3_pattern > threshold)[0]
        sdr_indices = tuple(active_indices.tolist())

        return cls(
            sdr_indices=sdr_indices,
            basis_vector=basis_vector,
            timestamp=timestamp,
            **kwargs,
        )

    @classmethod
    def from_spatial_event(
            cls,
            event: SpatialEvent,
            ca3_pattern: np.ndarray,
            basis_vector: Optional[np.ndarray],
            threshold: float = 0.5,
            context_tag: Optional[str] = None,
    ) -> HState:
        """Create HState from a SpatialEvent and its CA3 encoding.

        Args:
            event: Source SpatialEvent.
            ca3_pattern: SDR pattern array encoding this event (DG or CA3).
            basis_vector: Optional EC basis code embedding.
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

    @classmethod
    def from_dg_ca1(
            cls,
            event: SpatialEvent,
            dg_sdr: np.ndarray,
            ca1_sdr: np.ndarray,
            basis_vector: Optional[np.ndarray],
            threshold: float = 0.5,
            context_tag: Optional[str] = None,
    ) -> HState:
        """Create HState from DG and CA1 SDR patterns plus basis embedding.

        DG SDR provides the primary internal state; CA1 SDR captures the
        decoded/clean readout used for replay and planning.
        """
        dg_active = np.where(dg_sdr > threshold)[0]
        ca1_active = np.where(ca1_sdr > threshold)[0]

        return cls(
            sdr_indices=tuple(dg_active.tolist()),
            ca1_indices=tuple(ca1_active.tolist()),
            basis_vector=basis_vector,
            timestamp=event.timestamp,
            spatial_location=event.location.copy(),
            spatial_orientation=event.orientation.copy(),
            source_event=event,
            confidence=event.confidence,
            context_tag=context_tag,
        )

    def to_pattern(self, n_cells: int) -> np.ndarray:
        """Reconstruct binary DG SDR pattern from stored indices.

        Args:
            n_cells: Total number of CA3 cells.

        Returns:
            Binary pattern array of shape (n_cells,).
        """
        pattern = np.zeros(n_cells, dtype=np.float32)
        for idx in self.sdr_indices:
            if 0 <= idx < n_cells:
                pattern[idx] = 1.0
        return pattern

    def to_ca1_pattern(self, n_cells: int) -> np.ndarray:
        """Reconstruct binary CA1 SDR pattern from stored CA1 indices.

        Args:
            n_cells: Total number of CA1 cells.

        Returns:
            Binary pattern array of shape (n_cells,).
        """
        pattern = np.zeros(n_cells, dtype=np.float32)
        if self.ca1_indices is None:
            return pattern
        for idx in self.ca1_indices:
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
            if self.basis_vector is None or other.basis_vector is None:
                raise ValueError("Both states must have basis_vector for 'basis' metric")
            return float(np.linalg.norm(self.basis_vector - other.basis_vector))

        elif metric == "pattern":
            # Hamming distance (symmetric difference of active indices)
            self_set = set(self.sdr_indices)
            other_set = set(other.sdr_indices)
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

        Uses Jaccard similarity of active SDR indices.

        Args:
            other: Another HState to compare.

        Returns:
            Overlap score [0, 1] where 1 = identical patterns.
        """
        self_set = set(self.sdr_indices)
        other_set = set(other.sdr_indices)

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
            "sdr_indices": list(self.sdr_indices),
            "ca1_indices": list(self.ca1_indices) if self.ca1_indices is not None else None,
            "basis_vector": self.basis_vector.tolist() if self.basis_vector is not None else None,
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

        basis_vec = None
        if "basis_vector" in data and data["basis_vector"] is not None:
            basis_vec = np.array(data["basis_vector"])

        ca1_indices: Optional[Tuple[int, ...]] = None
        if "ca1_indices" in data and data["ca1_indices"] is not None:
            ca1_indices = tuple(data["ca1_indices"])

        return cls(
            sdr_indices=tuple(data["sdr_indices"]),
            ca1_indices=ca1_indices,
            basis_vector=basis_vec,
            timestamp=data["timestamp"],
            sequence_index=data.get("sequence_index"),
            spatial_location=spatial_location,
            spatial_orientation=spatial_orientation,
            context_tag=data.get("context_tag"),
            confidence=data.get("confidence", 1.0),
            extra=data.get("extra", {}),
        )

    def __hash__(self) -> int:
        """Hash based on SDR indices (immutable part of identity)."""
        return hash(self.sdr_indices)

    def __eq__(self, other: object) -> bool:
        """Equality based on SDR indices."""
        if not isinstance(other, HState):
            return NotImplemented
        return self.sdr_indices == other.sdr_indices

    def __repr__(self) -> str:
        spatial_info = f", loc={self.spatial_location}" if self.is_spatial else ""
        context_info = f", ctx={self.context_tag}" if self.context_tag else ""
        return (
            f"HState(id={self.id[:8]}..., "
            f"n_active={self.n_active_cells}"
            f"{spatial_info}{context_info})"
        )
