# Copyright 2025 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Transform registry for converting between location representation types.

This module provides:
- A registry to register and lookup transforms between location types
  (e.g., metric ↔ SDR, SDR ↔ latent, metric ↔ latent).
- Base Transform protocol and built-in transforms (Identity, Linear,
  PlaceFieldSDRTransform).
- Helper to decode a State's location_payload to a target type.

Typical usage:
    from tbp.monty.frameworks.utils.location_transforms import (
        TransformRegistry,
        PlaceFieldSDRTransform,
    )
    registry = TransformRegistry()
    registry.register("metric", "sdr", PlaceFieldSDRTransform(...))
    sdr_payload = registry.apply("metric", "sdr", metric_payload)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tbp.monty.frameworks.utils.sdr import PlaceFieldEncoder, SDR


# ---------------------------------------------------------------------------
# Transform Protocol / Base Class
# ---------------------------------------------------------------------------
class Transform(ABC):
    """Abstract base class for location transforms."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Source location type (e.g., 'metric', 'sdr', 'latent')."""
        ...

    @property
    @abstractmethod
    def target_type(self) -> str:
        """Target location type."""
        ...

    @abstractmethod
    def forward(self, payload: dict) -> dict:
        """Transform payload from source_type to target_type.

        Args:
            payload: A location_payload dict with 'type' == source_type.

        Returns:
            A new location_payload dict with 'type' == target_type.
        """
        ...

    @abstractmethod
    def inverse(self, payload: dict) -> dict:
        """Transform payload from target_type back to source_type (if possible).

        Args:
            payload: A location_payload dict with 'type' == target_type.

        Returns:
            A new location_payload dict with 'type' == source_type.

        Raises:
            NotImplementedError: If inverse is not defined.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in Transforms
# ---------------------------------------------------------------------------
@dataclass
class IdentityTransform(Transform):
    """Identity transform (no-op). Source and target types are the same."""

    _type: str = "metric"

    @property
    def source_type(self) -> str:
        return self._type

    @property
    def target_type(self) -> str:
        return self._type

    def forward(self, payload: dict) -> dict:
        return payload.copy()

    def inverse(self, payload: dict) -> dict:
        return payload.copy()


@dataclass
class LinearTransform(Transform):
    """Linear (affine) transform between metric coordinate frames.

    new_coords = rotation @ coords + translation
    """

    rotation: np.ndarray  # (D, D)
    translation: np.ndarray  # (D,)
    source_frame: str = "body"
    target_frame: str = "world"

    @property
    def source_type(self) -> str:
        return "metric"

    @property
    def target_type(self) -> str:
        return "metric"

    def forward(self, payload: dict) -> dict:
        coords = np.asarray(payload["value"])
        new_coords = self.rotation @ coords + self.translation
        return {
            "type": "metric",
            "value": new_coords,
            "frame_id": self.target_frame,
        }

    def inverse(self, payload: dict) -> dict:
        coords = np.asarray(payload["value"])
        inv_rot = np.linalg.inv(self.rotation)
        orig_coords = inv_rot @ (coords - self.translation)
        return {
            "type": "metric",
            "value": orig_coords,
            "frame_id": self.source_frame,
        }


@dataclass
class PlaceFieldSDRTransform(Transform):
    """Transform between metric coordinates and SDR using PlaceFieldEncoder."""

    encoder: PlaceFieldEncoder
    source_frame: str = "body"

    @property
    def source_type(self) -> str:
        return "metric"

    @property
    def target_type(self) -> str:
        return "sdr"

    def forward(self, payload: dict) -> dict:
        coords = np.asarray(payload["value"])
        sdr = self.encoder.encode(coords)
        return sdr.to_payload(frame_id=payload.get("frame_id", self.source_frame))

    def inverse(self, payload: dict) -> dict:
        sdr = SDR.from_payload(payload)
        coords = self.encoder.decode(sdr)
        return {
            "type": "metric",
            "value": coords,
            "frame_id": payload.get("frame_id", self.source_frame),
        }


# ---------------------------------------------------------------------------
# Transform Registry
# ---------------------------------------------------------------------------
@dataclass
class TransformRegistry:
    """Registry for location transforms.

    Stores transforms keyed by (source_type, target_type) and provides
    lookup and application helpers.
    """

    _transforms: dict[tuple[str, str], Transform] = field(default_factory=dict)

    def register(
        self,
        source_type: str,
        target_type: str,
        transform: Transform,
        bidirectional: bool = True,
    ) -> None:
        """Register a transform.

        Args:
            source_type: Source location type.
            target_type: Target location type.
            transform: Transform instance.
            bidirectional: If True and transform has inverse, also register
                the reverse direction.
        """
        self._transforms[(source_type, target_type)] = transform
        if bidirectional:
            # Register inverse direction (will use transform.inverse)
            self._transforms[(target_type, source_type)] = _InverseWrapper(transform)

    def lookup(self, source_type: str, target_type: str) -> Transform | None:
        """Lookup a transform by (source_type, target_type)."""
        return self._transforms.get((source_type, target_type))

    def apply(
        self, source_type: str, target_type: str, payload: dict
    ) -> dict:
        """Apply a registered transform to a payload.

        Args:
            source_type: Source location type.
            target_type: Target location type.
            payload: Location payload dict.

        Returns:
            Transformed payload.

        Raises:
            KeyError: If no transform is registered for (source_type, target_type).
        """
        transform = self.lookup(source_type, target_type)
        if transform is None:
            raise KeyError(
                f"No transform registered for ({source_type} -> {target_type})"
            )
        return transform.forward(payload)

    def can_transform(self, source_type: str, target_type: str) -> bool:
        """Check if a transform is registered."""
        return (source_type, target_type) in self._transforms


@dataclass
class _InverseWrapper(Transform):
    """Wrapper that swaps forward/inverse for bidirectional registration."""

    _inner: Transform

    @property
    def source_type(self) -> str:
        return self._inner.target_type

    @property
    def target_type(self) -> str:
        return self._inner.source_type

    def forward(self, payload: dict) -> dict:
        return self._inner.inverse(payload)

    def inverse(self, payload: dict) -> dict:
        return self._inner.forward(payload)


# ---------------------------------------------------------------------------
# Global Default Registry (singleton for convenience)
# ---------------------------------------------------------------------------
_default_registry: TransformRegistry | None = None


def get_default_registry() -> TransformRegistry:
    """Get or create the default global transform registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = TransformRegistry()
        # Register identity transforms for common types
        _default_registry.register("metric", "metric", IdentityTransform("metric"), bidirectional=False)
        _default_registry.register("sdr", "sdr", IdentityTransform("sdr"), bidirectional=False)
        _default_registry.register("latent", "latent", IdentityTransform("latent"), bidirectional=False)
    return _default_registry


def register_default_sdr_transform(
    n_cells: int = 4096,
    sparsity: float = 0.02,
    spatial_extent: float = 1.0,
    n_dims: int = 3,
    seed: int = 42,
) -> PlaceFieldSDRTransform:
    """Register a PlaceFieldSDRTransform in the default registry.

    Returns the created transform for reference.
    """
    encoder = PlaceFieldEncoder(
        n_cells=n_cells,
        sparsity=sparsity,
        spatial_extent=spatial_extent,
        n_dims=n_dims,
        seed=seed,
    )
    transform = PlaceFieldSDRTransform(encoder=encoder)
    get_default_registry().register("metric", "sdr", transform)
    return transform
