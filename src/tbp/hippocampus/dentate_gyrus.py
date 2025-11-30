"""Dentate Gyrus (DG) implementation.

The DG performs pattern separation via ultra-sparse encoding. It takes
EC input and produces extremely sparse representations (0.1-1% active)
that map similar inputs to distant memory representations.

Key biological features modeled:
- Extreme sparsification (much sparser than neocortex's 2-5%)
- Massive expansion (EC → larger DG population)
- Strong feedforward inhibition
- Novelty sensitivity (simplified model of adult neurogenesis)

The DG output serves as an "address" or "index" for CA3 memory storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DGConfig:
    """Configuration for Dentate Gyrus.

    Attributes:
        n_input: Number of input dimensions (from EC).
        n_granule_cells: Number of granule cells (typically >> n_input for expansion).
        sparsity: Target sparsity (fraction of active cells). DG uses 0.1-1%,
            much sparser than neocortex (2-5%).
        expansion_factor: How much to expand input dimensionality.
            Actual n_granule_cells = n_input * expansion_factor if not set directly.
        inhibition_strength: Strength of feedforward inhibition (higher = sparser).
        novelty_threshold: Threshold for novelty detection.
        neurogenesis_rate: Rate at which "new" cells become excitable (0-1).
            Models adult neurogenesis effect.
    """

    n_input: int = 100
    n_granule_cells: int = 500  # Default: 5x expansion
    sparsity: float = 0.005  # 0.5% - middle of biological range (0.1-1%)
    expansion_factor: float = 5.0  # DG has ~5x more cells than EC input
    inhibition_strength: float = 1.0
    novelty_threshold: float = 0.3
    neurogenesis_rate: float = 0.01  # 1% of cells are "young" and excitable

    def __post_init__(self):
        # If using default, compute from expansion factor
        if self.n_granule_cells == 500 and self.n_input != 100:
            self.n_granule_cells = int(self.n_input * self.expansion_factor)

        # Validate sparsity is in biological range
        if not 0.001 <= self.sparsity <= 0.01:
            logger.warning(
                f"DG sparsity {self.sparsity} outside biological range [0.001, 0.01]. "
                "Biological DG uses 0.1-1% active cells."
            )


@dataclass
class DGOutput:
    """Output from Dentate Gyrus processing.

    Attributes:
        sparse_code: Binary sparse representation (SDR), shape (n_granule_cells,).
        active_indices: Indices of active granule cells.
        activation_values: Pre-threshold activation values.
        novelty_score: How novel/unfamiliar the input pattern is (0-1).
        n_active: Number of active cells.
        sparsity_achieved: Actual sparsity of output.
    """

    sparse_code: np.ndarray  # Binary SDR
    active_indices: np.ndarray
    activation_values: np.ndarray
    novelty_score: float
    n_active: int = field(init=False)
    sparsity_achieved: float = field(init=False)

    def __post_init__(self):
        self.n_active = len(self.active_indices)
        self.sparsity_achieved = self.n_active / len(self.sparse_code)


class DentateGyrus:
    """Dentate Gyrus: Pattern separation via ultra-sparse encoding.

    The DG transforms EC input into extremely sparse representations suitable
    for indexing memories in CA3. It performs:

    1. **Expansion**: Projects low-dim EC input to high-dim granule cell space
    2. **Sparsification**: Winner-take-all competition produces 0.1-1% active
    3. **Pattern separation**: Similar inputs → distant sparse codes
    4. **Novelty detection**: New/rare patterns activate "young" cells more

    The output is a sparse distributed representation (SDR) that serves as
    the "address" for an episodic memory in CA3.

    Attributes:
        config: DGConfig with hyperparameters.
        weights: Projection weights from EC to granule cells.
        cell_ages: Age of each cell (models neurogenesis; young = more excitable).
    """

    def __init__(self, config: DGConfig | None = None) -> None:
        """Initialize Dentate Gyrus.

        Args:
            config: Configuration. Uses defaults if None.
        """
        self.config = config or DGConfig()
        self._initialize_weights()
        self._initialize_cell_ages()

        # Track recent patterns for novelty detection
        self._recent_patterns: list[np.ndarray] = []
        self._max_recent = 100

    def _initialize_weights(self) -> None:
        """Initialize random projection weights (EC → DG).

        Uses sparse random projection for efficiency and biological plausibility.
        Each granule cell receives input from a random subset of EC cells.
        """
        # Sparse connectivity: each DG cell connects to ~20% of EC inputs
        connectivity = 0.2
        self.weights = np.random.randn(
            self.config.n_granule_cells, self.config.n_input
        )
        # Sparsify connections
        mask = np.random.random(self.weights.shape) > connectivity
        self.weights[mask] = 0
        # Normalize rows so each cell has similar total input
        row_norms = np.linalg.norm(self.weights, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1  # Avoid division by zero
        self.weights /= row_norms

    def _initialize_cell_ages(self) -> None:
        """Initialize cell ages for neurogenesis modeling.

        Young cells (low age) are more excitable and preferentially
        encode novel patterns. This models adult neurogenesis in DG.
        """
        # Most cells are "mature" (high age), some are "young" (low age)
        self.cell_ages = np.random.exponential(scale=10.0, size=self.config.n_granule_cells)
        # Normalize to [0, 1] where 0 = newborn, 1 = fully mature
        self.cell_ages = 1 - np.exp(-self.cell_ages / 10)

    def encode(self, ec_input: np.ndarray) -> DGOutput:
        """Encode EC input into ultra-sparse DG representation.

        This is the main pattern separation function. It:
        1. Projects input to high-dimensional granule cell space
        2. Applies inhibition to enforce sparsity
        3. Selects top-k winners (k based on target sparsity)
        4. Computes novelty score

        Args:
            ec_input: Input from Entorhinal Cortex, shape (n_input,) or (n_features,).
                Will be resized/padded if needed.

        Returns:
            DGOutput with sparse code and metadata.
        """
        # Ensure input is correct shape
        ec_input = self._prepare_input(ec_input)

        # 1. Project to granule cell space (expansion)
        activations = self.weights @ ec_input

        # 2. Add noise for stochasticity
        noise = np.random.randn(self.config.n_granule_cells) * 0.1
        activations += noise

        # 3. Modulate by cell age (young cells more excitable for novel input)
        novelty_score = self._compute_novelty(ec_input)
        if novelty_score > self.config.novelty_threshold:
            # Novel input: boost young cells
            age_modulation = 1.0 + (1.0 - self.cell_ages) * novelty_score
            activations *= age_modulation

        # 4. Apply feedforward inhibition (global normalization)
        activations = activations - np.mean(activations)
        activations = activations / (np.std(activations) + 1e-8)
        activations *= self.config.inhibition_strength

        # 5. Winner-take-all: select top k cells based on target sparsity
        n_active = max(1, int(self.config.n_granule_cells * self.config.sparsity))
        threshold_idx = np.argsort(activations)[-n_active]
        threshold = activations[threshold_idx]

        # 6. Create binary sparse code
        sparse_code = (activations >= threshold).astype(np.float32)
        active_indices = np.where(sparse_code > 0)[0]

        # 7. Update recent patterns for novelty tracking
        self._update_recent_patterns(ec_input)

        return DGOutput(
            sparse_code=sparse_code,
            active_indices=active_indices,
            activation_values=activations,
            novelty_score=novelty_score,
        )

    def _prepare_input(self, ec_input: np.ndarray) -> np.ndarray:
        """Prepare input to match expected dimensionality."""
        ec_input = np.asarray(ec_input).flatten()

        if len(ec_input) == self.config.n_input:
            return ec_input
        elif len(ec_input) < self.config.n_input:
            # Pad with zeros
            padded = np.zeros(self.config.n_input)
            padded[: len(ec_input)] = ec_input
            return padded
        else:
            # Truncate or project down
            return ec_input[: self.config.n_input]

    def _compute_novelty(self, ec_input: np.ndarray) -> float:
        """Compute how novel the input is compared to recent patterns.

        Novel patterns have low similarity to recently seen patterns.

        Args:
            ec_input: Current input pattern.

        Returns:
            Novelty score in [0, 1], where 1 = completely novel.
        """
        if len(self._recent_patterns) == 0:
            return 1.0  # First pattern is maximally novel

        # Compute similarity to recent patterns
        similarities = []
        for recent in self._recent_patterns:
            # Cosine similarity
            dot = np.dot(ec_input, recent)
            norm = np.linalg.norm(ec_input) * np.linalg.norm(recent)
            if norm > 0:
                similarities.append(dot / norm)
            else:
                similarities.append(0.0)

        # Novelty = 1 - max_similarity
        max_similarity = max(similarities) if similarities else 0.0
        return 1.0 - max(0.0, max_similarity)

    def _update_recent_patterns(self, ec_input: np.ndarray) -> None:
        """Update the buffer of recent patterns."""
        self._recent_patterns.append(ec_input.copy())
        if len(self._recent_patterns) > self._max_recent:
            self._recent_patterns.pop(0)

    def simulate_neurogenesis(self) -> None:
        """Simulate neurogenesis by making some cells "younger".

        This should be called periodically (e.g., between episodes) to
        model the continuous generation of new granule cells in DG.
        """
        n_new = max(1, int(self.config.n_granule_cells * self.config.neurogenesis_rate))
        new_cell_indices = np.random.choice(
            self.config.n_granule_cells, size=n_new, replace=False
        )
        # Make selected cells "young" (age = 0)
        self.cell_ages[new_cell_indices] = 0.0

        # Age all cells slightly
        self.cell_ages = np.minimum(1.0, self.cell_ages + 0.01)

        logger.debug(f"Neurogenesis: {n_new} new cells, mean age = {self.cell_ages.mean():.3f}")

    def reset(self) -> None:
        """Reset DG state (not weights)."""
        self._recent_patterns.clear()

    def __repr__(self) -> str:
        return (
            f"DentateGyrus(n_input={self.config.n_input}, "
            f"n_granule={self.config.n_granule_cells}, "
            f"sparsity={self.config.sparsity:.3f})"
        )


def compute_pattern_separation(dg: DentateGyrus, patterns: list[np.ndarray]) -> float:
    """Measure pattern separation quality.

    Computes how well DG separates similar inputs into distinct codes.

    Args:
        dg: DentateGyrus instance.
        patterns: List of input patterns.

    Returns:
        Average Hamming distance between output codes (higher = better separation).
    """
    if len(patterns) < 2:
        return 0.0

    outputs = [dg.encode(p) for p in patterns]
    codes = [o.sparse_code for o in outputs]

    distances = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            # Hamming distance (number of differing bits)
            dist = np.sum(codes[i] != codes[j])
            distances.append(dist)

    return float(np.mean(distances)) if distances else 0.0
