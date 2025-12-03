# Copyright 2025 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""SDR (Sparse Distributed Representation) utilities for location encoding.

This module provides:
- SDR representation conventions (active indices, dense bool, payload dict).
- Place-field / RBF encoder: maps metric coordinates to SDRs.
- Decoder: recovers approximate metric coordinates from SDRs.
- Distance metrics: Hamming, overlap.
- Binding & union operations for composing/decomposing SDRs.
- Path-integration helpers (transition operators).

Typical usage:
    from tbp.monty.frameworks.utils.sdr import (
        PlaceFieldEncoder,
        hamming_distance,
        sdr_bind,
        sdr_union,
    )
    encoder = PlaceFieldEncoder(n_cells=4096, sparsity=0.02, spatial_extent=1.0)
    sdr = encoder.encode(np.array([0.1, 0.2, 0.0]))
    coords = encoder.decode(sdr)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# SDR Representation
# ---------------------------------------------------------------------------
@dataclass
class SDR:
    """Sparse Distributed Representation.

    Stores a sparse binary vector as a set of active indices.

    Attributes:
        dim: Total dimensionality of the SDR (number of bits).
        active: Sorted array of active bit indices (int32).
        sparsity: Target sparsity (fraction of active bits). Informational.
    """

    dim: int
    active: np.ndarray  # shape (k,), dtype int32, sorted
    sparsity: float = 0.02

    def __post_init__(self):
        self.active = np.asarray(self.active, dtype=np.int32)
        self.active.sort()

    def to_dense(self) -> np.ndarray:
        """Return dense boolean array of shape (dim,)."""
        arr = np.zeros(self.dim, dtype=bool)
        arr[self.active] = True
        return arr

    @classmethod
    def from_dense(cls, arr: np.ndarray, sparsity: float = 0.02) -> "SDR":
        """Create SDR from dense boolean or binary array."""
        active = np.nonzero(arr)[0].astype(np.int32)
        return cls(dim=arr.shape[0], active=active, sparsity=sparsity)

    def to_payload(self, frame_id: str = "body") -> dict:
        """Return a location_payload dict compatible with State."""
        return {
            "type": "sdr",
            "value": self.active.copy(),
            "dim": self.dim,
            "sparsity": self.sparsity,
            "frame_id": frame_id,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "SDR":
        """Create SDR from a location_payload dict."""
        return cls(
            dim=payload["dim"],
            active=np.asarray(payload["value"], dtype=np.int32),
            sparsity=payload.get("sparsity", 0.02),
        )

    def __len__(self) -> int:
        return len(self.active)

    def __repr__(self) -> str:
        return f"SDR(dim={self.dim}, k={len(self.active)}, sparsity={self.sparsity:.3f})"


# ---------------------------------------------------------------------------
# Distance Metrics
# ---------------------------------------------------------------------------
def hamming_distance(a: SDR, b: SDR) -> int:
    """Hamming distance: number of differing bits."""
    assert a.dim == b.dim, "SDRs must have the same dimensionality"
    set_a = set(a.active.tolist())
    set_b = set(b.active.tolist())
    return len(set_a.symmetric_difference(set_b))


def overlap(a: SDR, b: SDR) -> int:
    """Number of shared active bits."""
    set_a = set(a.active.tolist())
    set_b = set(b.active.tolist())
    return len(set_a.intersection(set_b))


def overlap_score(a: SDR, b: SDR) -> float:
    """Jaccard-like overlap: intersection / union."""
    set_a = set(a.active.tolist())
    set_b = set(b.active.tolist())
    union = set_a.union(set_b)
    if len(union) == 0:
        return 1.0
    return len(set_a.intersection(set_b)) / len(union)


# ---------------------------------------------------------------------------
# Binding & Union
# ---------------------------------------------------------------------------
def sdr_bind(a: SDR, b: SDR) -> SDR:
    """Bind two SDRs using symmetric difference (XOR).

    Binding produces a new SDR that can be used to associate feature+location.
    Unbinding: bind(bound, a) ≈ b (approximately, with some noise).
    """
    assert a.dim == b.dim
    set_a = set(a.active.tolist())
    set_b = set(b.active.tolist())
    bound = np.array(sorted(set_a.symmetric_difference(set_b)), dtype=np.int32)
    return SDR(dim=a.dim, active=bound, sparsity=a.sparsity)


def sdr_union(sdrs: Sequence[SDR]) -> SDR:
    """Union of multiple SDRs (bitwise OR).

    Useful for representing multiple hypotheses or uncertainty.
    """
    if len(sdrs) == 0:
        raise ValueError("Cannot union empty sequence")
    dim = sdrs[0].dim
    combined: set[int] = set()
    for s in sdrs:
        assert s.dim == dim
        combined.update(s.active.tolist())
    return SDR(dim=dim, active=np.array(sorted(combined), dtype=np.int32))


def sdr_intersection(sdrs: Sequence[SDR]) -> SDR:
    """Intersection of multiple SDRs (bitwise AND)."""
    if len(sdrs) == 0:
        raise ValueError("Cannot intersect empty sequence")
    dim = sdrs[0].dim
    common = set(sdrs[0].active.tolist())
    for s in sdrs[1:]:
        assert s.dim == dim
        common.intersection_update(s.active.tolist())
    return SDR(dim=dim, active=np.array(sorted(common), dtype=np.int32))


# ---------------------------------------------------------------------------
# Place-Field / RBF Encoder
# ---------------------------------------------------------------------------
@dataclass
class PlaceFieldEncoder:
    """Encode metric coordinates to SDRs using place-field (RBF) populations.

    Each "cell" has a preferred location (center) and a tuning width (sigma).
    The response is a Gaussian of the distance to the center. The top-k
    responding cells become the active bits of the SDR.

    Attributes:
        n_cells: Number of place cells (SDR dimensionality).
        sparsity: Target fraction of active cells.
        spatial_extent: Half-extent of the space (cells uniformly in [-extent, extent]^D).
        sigma: Tuning width (standard deviation) of each cell's Gaussian.
        centers: (n_cells, D) array of preferred locations.
        rng: Random number generator for reproducibility.
    """

    n_cells: int = 4096
    sparsity: float = 0.02
    spatial_extent: float = 1.0
    sigma: float | None = None  # auto-computed if None
    n_dims: int = 3
    seed: int = 42
    centers: np.ndarray = field(init=False, repr=False)
    _k: int = field(init=False, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.centers = rng.uniform(
            -self.spatial_extent, self.spatial_extent, size=(self.n_cells, self.n_dims)
        )
        self._k = max(1, int(self.n_cells * self.sparsity))
        if self.sigma is None:
            # Auto-compute sigma so nearby points have similar SDRs.
            # Heuristic: average distance to nearest neighbor among centers.
            from scipy.spatial import cKDTree

            tree = cKDTree(self.centers)
            dists, _ = tree.query(self.centers, k=2)
            avg_nn_dist = dists[:, 1].mean()
            self.sigma = avg_nn_dist * 1.5

    def encode(self, coords: np.ndarray) -> SDR:
        """Encode a metric coordinate vector to an SDR.

        Args:
            coords: 1D array of shape (D,) representing a point.

        Returns:
            SDR with top-k active cells.
        """
        coords = np.asarray(coords).ravel()
        if coords.shape[0] != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims}D coords, got shape {coords.shape}"
            )
        # Compute squared distances to all centers
        sq_dists = np.sum((self.centers - coords) ** 2, axis=1)
        # Gaussian responses
        sigma: float = self.sigma  # type: ignore[assignment]
        responses = np.exp(-sq_dists / (2 * sigma**2))
        # Top-k indices
        top_k = np.argpartition(responses, -self._k)[-self._k :]
        top_k = top_k[np.argsort(-responses[top_k])]  # sort descending
        return SDR(dim=self.n_cells, active=top_k.astype(np.int32), sparsity=self.sparsity)

    def decode(self, sdr: SDR) -> np.ndarray:
        """Decode an SDR back to approximate metric coordinates.

        Uses weighted average of active cell centers (weights = 1).

        Args:
            sdr: SDR to decode.

        Returns:
            1D array of shape (D,) approximate location.
        """
        if len(sdr.active) == 0:
            return np.zeros(self.n_dims)
        return self.centers[sdr.active].mean(axis=0)


# ---------------------------------------------------------------------------
# Grid-Cell Encoder (multi-scale periodic)
# ---------------------------------------------------------------------------
@dataclass
class GridCellEncoder:
    """Encode metric coordinates using grid-cell-like periodic basis functions.

    Multiple modules with different scales/orientations; each module produces
    a small SDR; concatenated to form the full SDR.

    Attributes:
        n_modules: Number of grid modules (different scales).
        cells_per_module: Number of cells per module.
        spatial_extent: Extent of space.
        base_period: Smallest grid period.
        period_ratio: Ratio between successive module periods.
        seed: Random seed.
    """

    n_modules: int = 8
    cells_per_module: int = 64
    spatial_extent: float = 1.0
    base_period: float = 0.1
    period_ratio: float = 1.4
    n_dims: int = 3
    seed: int = 42
    _phases: np.ndarray = field(init=False, repr=False)
    _periods: np.ndarray = field(init=False, repr=False)
    _orientations: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self._periods = self.base_period * (self.period_ratio ** np.arange(self.n_modules))
        # Random phases for each cell in each module
        self._phases = rng.uniform(0, 2 * np.pi, size=(self.n_modules, self.cells_per_module))
        # Random orientation vectors for each module (unit vectors in n_dims)
        orientations = rng.normal(size=(self.n_modules, self.n_dims))
        self._orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)

    @property
    def dim(self) -> int:
        return self.n_modules * self.cells_per_module

    @property
    def sparsity(self) -> float:
        # Approximately 1 cell active per module
        return self.n_modules / self.dim

    def encode(self, coords: np.ndarray) -> SDR:
        """Encode metric coordinates using grid-cell basis."""
        coords = np.asarray(coords).ravel()
        active = []
        for m in range(self.n_modules):
            # Project coords onto module orientation
            proj = np.dot(coords, self._orientations[m])
            # Periodic response
            responses = np.cos(2 * np.pi * proj / self._periods[m] + self._phases[m])
            # Winner cell
            winner = int(np.argmax(responses))
            active.append(m * self.cells_per_module + winner)
        return SDR(dim=self.dim, active=np.array(active, dtype=np.int32), sparsity=self.sparsity)

    def decode(self, sdr: SDR) -> np.ndarray:
        """Decode is ill-defined for grid cells (periodic ambiguity).

        Returns zeros; use PlaceFieldEncoder if decoding is needed.
        """
        return np.zeros(self.n_dims)


# ---------------------------------------------------------------------------
# Path Integration (Transition Operators)
# ---------------------------------------------------------------------------
@dataclass
class SDRPathIntegrator:
    """Path integration in SDR space using learned/random transition matrices.

    For each discrete movement command, a sparse permutation-like transform
    is applied to shift the SDR pattern.

    Attributes:
        dim: SDR dimensionality.
        n_commands: Number of distinct movement commands.
        seed: Random seed.
    """

    dim: int = 4096
    n_commands: int = 6  # e.g., +x, -x, +y, -y, +z, -z
    seed: int = 42
    _permutations: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        # Each command is a permutation of indices
        self._permutations = np.zeros((self.n_commands, self.dim), dtype=np.int32)
        for c in range(self.n_commands):
            self._permutations[c] = rng.permutation(self.dim)

    def step(self, sdr: SDR, command: int) -> SDR:
        """Apply movement command to SDR, returning new SDR.

        Args:
            sdr: Current location SDR.
            command: Movement command index in [0, n_commands).

        Returns:
            New SDR after applying the transition.
        """
        perm = self._permutations[command]
        new_active = perm[sdr.active]
        return SDR(dim=sdr.dim, active=new_active, sparsity=sdr.sparsity)


# ---------------------------------------------------------------------------
# Convenience: Nearest-Neighbor Search in SDR Space
# ---------------------------------------------------------------------------
def find_nearest_sdrs(
    query: SDR, candidates: Sequence[SDR], k: int = 1
) -> list[tuple[int, int]]:
    """Find k nearest SDRs by Hamming distance.

    Args:
        query: Query SDR.
        candidates: List of candidate SDRs.
        k: Number of nearest neighbors.

    Returns:
        List of (index, hamming_distance) tuples, sorted ascending by distance.
    """
    dists = [(i, hamming_distance(query, c)) for i, c in enumerate(candidates)]
    dists.sort(key=lambda x: x[1])
    return dists[:k]
