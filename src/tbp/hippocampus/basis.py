# Copyright 2025 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Basis Code System for Hippocampal Encoding.

This module implements pluggable basis functions for entorhinal cortex (EC)
encoding. The basis system generalizes grid cell encoding beyond pure spatial
metrics, supporting:
- Spatial grid codes (MetricGridBasis) - classical EC grid cells
- Conceptual grid bases (for abstract spaces)
- Graph Laplacian bases (for graph-structured domains)
- Learned embeddings (via subclassing)

The design follows TEM/SR principles where EC provides a universal coordinate
system that can encode ANY relational structure.

Architecture:
    BasisCode (Abstract)
        ├── MetricGridBasis - Standard spatial grid cells
        ├── PlaceBasis - Place cell population encoding
        ├── CombinedBasis - Multiple bases combined
        └── (User extensions)

Example:
    >>> # Spatial encoding with grid cells
    >>> basis = MetricGridBasis(n_modules=8, n_phases=4)
    >>> location = np.array([1.0, 2.0, 0.5])
    >>> code = basis.encode(location)

    >>> # Combined spatial encoding with grid and place cells
    >>> combined = CombinedBasis([
    ...     MetricGridBasis(n_modules=6),
    ...     PlaceBasis(n_cells=100),
    ... ])
    >>> # Example: you may define your own BasisCode subclass for heading, rotation, task IDs, etc.
    >>> full_code = combined.encode(location)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


@dataclass
class BasisConfig:
    """Base configuration for basis codes.

    Attributes:
        output_dim: Dimensionality of encoded vectors.
        normalize: Whether to L2-normalize output vectors.
        seed: Random seed for reproducibility.
    """

    output_dim: int = 64
    normalize: bool = True
    seed: Optional[int] = None


class BasisCode(ABC):
    """Abstract base class for hippocampal basis encodings.

    BasisCode defines the interface for converting inputs (spatial locations,
    abstract features, graph positions, etc.) into continuous vector embeddings
    suitable for hippocampal processing.

    The design is inspired by:
    - EC grid cells providing metric structure
    - TEM's structural embedding hypothesis
    - SR's learned state representations

    Subclasses must implement:
    - encode(): Convert input to basis vector
    - input_space: Description of expected input format
    - output_dim: Dimensionality of output vectors
    """

    @abstractmethod
    def encode(self, input_data: Any) -> np.ndarray:
        """Encode input data into basis vector.

        Args:
            input_data: Input in format specified by input_space property.

        Returns:
            Numpy array of shape (output_dim,) containing the basis encoding.
        """
        pass

    @property
    @abstractmethod
    def input_space(self) -> str:
        """Description of expected input format."""
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of output vectors."""
        pass

    def encode_batch(self, inputs: Sequence[Any]) -> np.ndarray:
        """Encode multiple inputs.

        Args:
            inputs: Sequence of inputs.

        Returns:
            Array of shape (n_inputs, output_dim).
        """
        return np.array([self.encode(inp) for inp in inputs])

    def distance(self, code1: np.ndarray, code2: np.ndarray) -> float:
        """Compute distance between two basis codes.

        Default implementation uses Euclidean distance.
        Subclasses may override for domain-specific metrics.

        Args:
            code1: First basis code.
            code2: Second basis code.

        Returns:
            Distance value (lower = more similar).
        """
        return float(np.linalg.norm(code1 - code2))

    def similarity(self, code1: np.ndarray, code2: np.ndarray) -> float:
        """Compute similarity between two basis codes.

        Default implementation uses cosine similarity.

        Args:
            code1: First basis code.
            code2: Second basis code.

        Returns:
            Similarity value in [-1, 1] where 1 = identical.
        """
        norm1 = np.linalg.norm(code1)
        norm2 = np.linalg.norm(code2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(np.dot(code1, code2) / (norm1 * norm2))


@dataclass
class MetricGridConfig(BasisConfig):
    """Configuration for MetricGridBasis.

    Attributes:
        n_modules: Number of grid modules (different spacings).
        n_phases: Number of phase offsets per module.
        n_orientations: Number of orientation offsets per module.
        min_spacing: Minimum grid spacing.
        max_spacing: Maximum grid spacing.
        input_dim: Dimensionality of input space (2 or 3).
    """

    n_modules: int = 8
    n_phases: int = 4
    n_orientations: int = 3
    min_spacing: float = 0.1
    max_spacing: float = 2.0
    input_dim: int = 3
    # Override parent's output_dim - will be computed in __post_init__
    output_dim: int = 0

    def __post_init__(self):
        """Compute output dimensionality."""
        # Each module has n_phases * n_orientations * 3 (for hexagonal basis)
        self.output_dim = self.n_modules * self.n_phases * self.n_orientations * 3


class MetricGridBasis(BasisCode):
    """Grid cell basis encoding for metric spaces.

    Implements a population of grid cells with different spacings,
    phases, and orientations. This is the standard EC encoding for
    spatial navigation and provides a multi-scale metric structure.

    Based on the hexagonal grid cell model:
    - Each grid cell has 3 cosine basis functions at 60° angles
    - Multiple modules with geometrically-spaced scales
    - Random phases and orientations for coverage

    Attributes:
        config: MetricGridConfig with hyperparameters.
        spacings: Array of grid spacings per module.
        phases: Array of phase offsets (n_modules, n_phases, 2).
        orientations: Array of orientation angles (n_modules, n_orientations).
    """

    def __init__(self, config: Optional[MetricGridConfig] = None) -> None:
        """Initialize MetricGridBasis.

        Args:
            config: Configuration. Uses defaults if None.
        """
        self.config = config or MetricGridConfig()
        self._rng = np.random.default_rng(self.config.seed)

        # Generate grid parameters
        self._setup_grid_parameters()

    def _setup_grid_parameters(self) -> None:
        """Initialize grid cell parameters."""
        cfg = self.config

        # Geometric series of spacings (like biological grid cells)
        ratio = (cfg.max_spacing / cfg.min_spacing) ** (1 / max(1, cfg.n_modules - 1))
        self.spacings = cfg.min_spacing * (ratio ** np.arange(cfg.n_modules))

        # Random phases for each module
        self.phases = np.zeros((cfg.n_modules, cfg.n_phases, 2))
        for m in range(cfg.n_modules):
            for p in range(cfg.n_phases):
                self.phases[m, p] = self._rng.uniform(
                    -self.spacings[m] / 2,
                    self.spacings[m] / 2,
                    size=2
                )

        # Random orientations (0 to 60 degrees for hexagonal symmetry)
        self.orientations = self._rng.uniform(
            0, np.pi / 3,
            size=(cfg.n_modules, cfg.n_orientations)
        )

    def encode(self, input_data: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Encode a spatial location into grid cell activations.

        Args:
            input_data: 2D or 3D location array [x, y] or [x, y, z].
                Z coordinate is ignored if present.

        Returns:
            Grid cell activation vector of shape (output_dim,).
        """
        location = np.asarray(input_data).flatten()
        loc_2d = location[:2]  # Use only x, y

        cfg = self.config
        activations = []

        for m in range(cfg.n_modules):
            spacing = self.spacings[m]

            for p in range(cfg.n_phases):
                phase = self.phases[m, p]

                for o in range(cfg.n_orientations):
                    orientation = self.orientations[m, o]

                    # Compute grid cell activation
                    activation = self._compute_grid_activation(
                        loc_2d, spacing, phase, orientation
                    )
                    activations.extend(activation)

        result = np.array(activations, dtype=np.float64)

        if self.config.normalize:
            norm = np.linalg.norm(result)
            if norm > 1e-10:
                result = result / norm

        return result

    def _compute_grid_activation(
            self,
            location: np.ndarray,
            spacing: float,
            phase: np.ndarray,
            orientation: float,
    ) -> List[float]:
        """Compute single grid cell activation (3 cosines at 60°).

        Args:
            location: 2D location.
            spacing: Grid spacing.
            phase: 2D phase offset.
            orientation: Grid orientation angle.

        Returns:
            List of 3 activation values.
        """
        loc_shifted = location - phase

        # Rotate to grid orientation
        cos_theta = np.cos(orientation)
        sin_theta = np.sin(orientation)
        rot = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        loc_rot = rot @ loc_shifted

        # Three cosines at 60-degree intervals (hexagonal pattern)
        activations = []
        for angle_offset in [0, np.pi / 3, 2 * np.pi / 3]:
            direction = np.array([np.cos(angle_offset), np.sin(angle_offset)])
            projection = np.dot(loc_rot, direction)
            activation = np.cos(2 * np.pi * projection / spacing)
            activations.append(float(activation))

        return activations

    @property
    def input_space(self) -> str:
        return f"R^{self.config.input_dim} (spatial coordinates)"

    @property
    def output_dim(self) -> int:
        return self.config.output_dim


@dataclass
class PlaceBasisConfig(BasisConfig):
    """Configuration for PlaceBasis.

    Attributes:
        n_cells: Number of place cells.
        field_radius: Radius of place fields.
        spatial_extent: Extent of space for random place cell centers.
        input_dim: Dimensionality of input space (2 or 3).
    """

    n_cells: int = 100
    field_radius: float = 0.2
    spatial_extent: tuple = (2.0, 2.0, 2.0)
    input_dim: int = 3
    # Override parent's output_dim - will be computed in __post_init__
    output_dim: int = 0

    def __post_init__(self):
        self.output_dim = self.n_cells


class PlaceBasis(BasisCode):
    """Place cell basis encoding.

    Implements a population of place cells with Gaussian tuning curves.
    Each place cell has a preferred location and fires when the input
    location is near its center.

    Attributes:
        config: PlaceBasisConfig with hyperparameters.
        centers: Array of place cell centers (n_cells, input_dim).
    """

    def __init__(self, config: Optional[PlaceBasisConfig] = None) -> None:
        """Initialize PlaceBasis.

        Args:
            config: Configuration. Uses defaults if None.
        """
        self.config = config or PlaceBasisConfig()
        self._rng = np.random.default_rng(self.config.seed)

        # Random place cell centers
        extent = np.array(self.config.spatial_extent[:self.config.input_dim])
        self.centers = self._rng.uniform(
            low=-extent / 2,
            high=extent / 2,
            size=(self.config.n_cells, self.config.input_dim)
        )

    def encode(self, input_data: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Encode a location into place cell activations.

        Args:
            input_data: Location array of shape (input_dim,).

        Returns:
            Place cell activation vector of shape (n_cells,).
        """
        location = np.asarray(input_data).flatten()
        if len(location) < self.config.input_dim:
            # Pad with zeros
            padded = np.zeros(self.config.input_dim)
            padded[:len(location)] = location
            location = padded
        else:
            location = location[:self.config.input_dim]

        # Compute distances to all centers
        distances = np.linalg.norm(self.centers - location, axis=1)

        # Gaussian tuning curves
        activations = np.exp(
            -(distances ** 2) / (2 * self.config.field_radius ** 2)
        )

        if self.config.normalize:
            norm = np.linalg.norm(activations)
            if norm > 1e-10:
                activations = activations / norm

        return activations

    @property
    def input_space(self) -> str:
        return f"R^{self.config.input_dim} (spatial coordinates)"

    @property
    def output_dim(self) -> int:
        return self.config.output_dim


class CombinedBasis(BasisCode):
    """Combines multiple basis codes into a single encoding.

    Useful for encoding multiple aspects of state (e.g., location + heading,
    position + velocity, spatial + semantic features).

    Attributes:
        bases: List of component BasisCode instances.
        weights: Optional weights for each basis (default: equal).
        names: Optional semantic names for each basis, enabling dict-based input.
    """

    def __init__(
            self,
            bases: List[BasisCode],
            weights: Optional[List[float]] = None,
            names: Optional[List[str]] = None,
            normalize: bool = True,
    ) -> None:
        """Initialize CombinedBasis.

        Args:
            bases: List of BasisCode instances to combine.
            weights: Optional weights for each basis. If None, uses equal weights.
            names: Optional semantic names for each basis. If provided, enables
                dict-based input where keys match these names. Multiple bases
                can share the same name to receive the same input.
            normalize: Whether to normalize the combined output.

        Raises:
            ValueError: If bases is empty, or if weights/names length doesn't match bases.
        """
        if not bases:
            raise ValueError("Must provide at least one basis")

        self.bases = bases
        self.weights = weights or [1.0] * len(bases)
        self.names = names
        self._normalize = normalize

        if len(self.weights) != len(bases):
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of bases ({len(bases)})")

        if self.names is not None and len(self.names) != len(bases):
            raise ValueError(f"Number of names ({len(self.names)}) must match number of bases ({len(bases)})")

        self._output_dim = sum(b.output_dim for b in bases)

    def encode(self, input_data: Union[Dict[str, Any], Any]) -> np.ndarray:
        """Encode input using all component bases.

        Args:
            input_data: Either:
                - A dict mapping semantic names to their inputs (if names were provided)
                - A single input that will be passed to all bases (if names not provided)

        Returns:
            Concatenated encoding from all bases.

        Raises:
            ValueError: If dict input is provided but required keys are missing.

        Examples:
            >>> # With names - semantic key access
            >>> combined = CombinedBasis(
            ...     bases=[grid_basis, place_basis],
            ...     names=["location", "location"]
            ... )
            >>> code = combined.encode({"location": np.array([1.0, 2.0])})
            >>>
            >>> # Without names - same input to all bases
            >>> combined = CombinedBasis([grid_basis, place_basis])
            >>> code = combined.encode(np.array([1.0, 2.0]))
        """
        if isinstance(input_data, dict):
            encodings = self._encode_dict_input(input_data)
        else:
            encodings = self._encode_single_input(input_data)

        result = np.concatenate(encodings)
        return self._normalize_result(result)

    def _encode_dict_input(self, input_data: Dict[str, Any]) -> List[np.ndarray]:
        """Encode dict input using semantic names or numeric keys.

        Args:
            input_data: Dictionary of inputs.

        Returns:
            List of encoded arrays.

        Raises:
            ValueError: If required keys are missing and names are specified.
        """
        if self.names is not None:
            return self._encode_dict_with_names(input_data)
        return self._encode_dict_fallback(input_data)

    def _encode_dict_with_names(self, input_data: Dict[str, Any]) -> List[np.ndarray]:
        """Encode using semantic names to access dict.

        Args:
            input_data: Dictionary of inputs.

        Returns:
            List of encoded arrays.

        Raises:
            ValueError: If required keys are missing.
        """
        encodings = []
        for i, (basis, weight, name) in enumerate(zip(self.bases, self.weights, self.names)):
            if name not in input_data:
                raise ValueError(
                    f"Input dict missing required key '{name}' for basis {i}. "
                    f"Available keys: {list(input_data.keys())}"
                )
            inp = input_data[name]
            encoded = basis.encode(inp) * weight
            encodings.append(encoded)
        return encodings

    def _encode_dict_fallback(self, input_data: Dict[str, Any]) -> List[np.ndarray]:
        """Encode dict using numeric string keys or entire dict.

        Args:
            input_data: Dictionary of inputs.

        Returns:
            List of encoded arrays.
        """
        encodings = []
        for i, (basis, weight) in enumerate(zip(self.bases, self.weights)):
            key = str(i)
            if key not in input_data:
                raise ValueError(
                    f"Input dict missing required numeric key '{key}' for basis {i}. "
                    f"Available keys: {list(input_data.keys())}"
                )
            inp = input_data[key]
            encoded = basis.encode(inp) * weight
            encodings.append(encoded)
        return encodings

    def _encode_single_input(self, input_data: Any) -> List[np.ndarray]:
        """Encode single input passed to all bases.

        Args:
            input_data: Input to encode.

        Returns:
            List of encoded arrays.
        """
        encodings = []
        for basis, weight in zip(self.bases, self.weights):
            encoded = basis.encode(input_data) * weight
            encodings.append(encoded)
        return encodings

    def _normalize_result(self, result: np.ndarray) -> np.ndarray:
        """Normalize result if configured.

        Args:
            result: Array to normalize.

        Returns:
            Normalized array.
        """
        if self._normalize:
            norm = np.linalg.norm(result)
            if norm > 1e-10:
                result = result / norm
        return result

    @property
    def input_space(self) -> str:
        return " + ".join(b.input_space for b in self.bases)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class IdentityBasis(BasisCode):
    """Identity basis that passes input through unchanged.

    Useful for pre-computed embeddings or as a placeholder.
    """

    def __init__(self, dim: int, normalize: bool = True) -> None:
        """Initialize IdentityBasis.

        Args:
            dim: Expected dimensionality of input/output.
            normalize: Whether to normalize output.
        """
        self._dim = dim
        self._normalize = normalize

    def encode(self, input_data: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Pass through input (with optional normalization).

        Args:
            input_data: Array of shape (dim,).

        Returns:
            Same array (optionally normalized).
        """
        result = np.asarray(input_data, dtype=np.float64).flatten()

        if len(result) != self._dim:
            if len(result) < self._dim:
                padded = np.zeros(self._dim)
                padded[:len(result)] = result
                result = padded
            else:
                result = result[:self._dim]

        if self._normalize:
            norm = np.linalg.norm(result)
            if norm > 1e-10:
                result = result / norm

        return result

    @property
    def input_space(self) -> str:
        return f"R^{self._dim} (pre-computed embedding)"

    @property
    def output_dim(self) -> int:
        return self._dim


# Convenience factory functions


def create_spatial_basis(
        n_modules: int = 8,
        n_phases: int = 4,
        include_place_cells: bool = True,
        n_place_cells: int = 50,
        seed: Optional[int] = None,
) -> BasisCode:
    """Create a standard spatial encoding basis.

    Args:
        n_modules: Number of grid modules.
        n_phases: Number of phases per module.
        include_place_cells: Whether to include place cells.
        n_place_cells: Number of place cells (if included).
        seed: Random seed.

    Returns:
        BasisCode for spatial encoding.
    """
    grid_config = MetricGridConfig(
        n_modules=n_modules,
        n_phases=n_phases,
        seed=seed,
    )
    grid_basis = MetricGridBasis(grid_config)

    if include_place_cells:
        place_config = PlaceBasisConfig(
            n_cells=n_place_cells,
            seed=seed,
        )
        place_basis = PlaceBasis(place_config)
        return CombinedBasis([grid_basis, place_basis])

    return grid_basis
