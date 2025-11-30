"""CA1 region implementation with comparator functionality.

CA1 is the primary output of the hippocampus that:
1. Receives predictions from CA3 (Schaffer collaterals)
2. Receives direct input from Entorhinal Cortex (perforant path)
3. Compares these inputs to detect match/mismatch
4. Outputs to subiculum and back to Entorhinal Cortex

Key biological features:
- Comparator function: CA3 prediction vs EC reality
- Match = familiar, mismatch = novelty signal
- Temporal ordering and sequence learning
- Output to neocortex via subiculum
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import SpatialEvent


@dataclass
class CA1Config:
    """Configuration for CA1 comparator network.

    Attributes:
        n_pyramidal_cells: Number of CA1 pyramidal neurons.
        n_active_cells: Target number of active cells per pattern.
        match_threshold: Threshold for considering CA3/EC match (0-1).
        mismatch_learning_rate: Learning rate for mismatch-driven plasticity.
        temporal_window: Time window for sequence detection (seconds).
        ec_weight: Weight of EC input relative to CA3.
        output_sparsity: Sparsity of output to subiculum.
    """

    n_pyramidal_cells: int = 2500  # Similar to CA3
    n_active_cells: int = 50  # ~2% sparsity
    match_threshold: float = 0.6
    mismatch_learning_rate: float = 0.15
    temporal_window: float = 1.0
    ec_weight: float = 0.5  # Balance between EC and CA3
    output_sparsity: float = 0.02


@dataclass
class ComparisonResult:
    """Result of CA1 comparison between CA3 prediction and EC input.

    Attributes:
        match_score: How well CA3 and EC match (0=mismatch, 1=match).
        is_match: Whether this is considered a match.
        novelty_signal: Inverse of match, indicates novelty.
        output_pattern: CA1 output pattern to subiculum.
        ca3_contribution: How much CA3 influenced output.
        ec_contribution: How much EC influenced output.
        sequence_position: Position in detected sequence (if any).
    """

    match_score: float
    is_match: bool
    novelty_signal: float
    output_pattern: np.ndarray
    ca3_contribution: float
    ec_contribution: float
    sequence_position: Optional[int] = None


@dataclass
class SequenceElement:
    """Element in a learned sequence.

    Attributes:
        pattern: The CA1 pattern for this element.
        event: Associated spatial event.
        timestamp: When this element was observed.
        position: Position in the sequence.
    """

    pattern: np.ndarray
    event: SpatialEvent
    timestamp: float
    position: int


class CA1:
    """CA1 comparator network.

    CA1 implements a comparator that:
    - Compares CA3 predictions with direct EC input
    - Detects match (familiar) vs mismatch (novel)
    - Learns temporal sequences
    - Outputs combined representation to neocortex

    The match/mismatch signal is crucial for:
    - Novelty detection (triggers encoding)
    - Expectation violation (prediction error)
    - Memory retrieval confirmation

    Example:
        >>> config = CA1Config(n_pyramidal_cells=1000)
        >>> ca1 = CA1(config)
        >>> # Compare CA3 prediction with EC reality
        >>> result = ca1.compare(ca3_pattern, ec_pattern, event)
        >>> if result.is_match:
        ...     print("Familiar location!")
        >>> else:
        ...     print(f"Novelty detected: {result.novelty_signal:.2f}")
    """

    def __init__(self, config: Optional[CA1Config] = None):
        """Initialize CA1 network.

        Args:
            config: CA1 configuration. Uses defaults if not provided.
        """
        self.config = config or CA1Config()
        self._rng = np.random.default_rng()

        n = self.config.n_pyramidal_cells

        # Projection weights from CA3 (Schaffer collaterals)
        self._ca3_weights = self._rng.normal(
            0, 0.1, (n, n)
        ).astype(np.float32)

        # Projection weights from EC (perforant path)
        self._ec_weights = self._rng.normal(
            0, 0.1, (n, n)
        ).astype(np.float32)

        # Sequence storage
        self._sequences: List[List[SequenceElement]] = []
        self._current_sequence: List[SequenceElement] = []
        self._last_timestamp: Optional[float] = None

        # Statistics
        self._total_comparisons = 0
        self._match_count = 0
        self._mismatch_count = 0
        self._novelty_history: List[float] = []

    def compare(
        self,
        ca3_pattern: np.ndarray,
        ec_pattern: np.ndarray,
        event: SpatialEvent,
    ) -> ComparisonResult:
        """Compare CA3 prediction with EC input.

        This is the core comparator function of CA1.
        Match = the predicted state matches observed state.
        Mismatch = novelty/prediction error.

        Args:
            ca3_pattern: Predicted pattern from CA3.
            ec_pattern: Direct sensory input from EC.
            event: Current spatial event.

        Returns:
            ComparisonResult with match/mismatch information.
        """
        self._total_comparisons += 1

        # Project patterns to CA1 space
        ca3_input = self._project_ca3(ca3_pattern)
        ec_input = self._project_ec(ec_pattern)

        # Compute match score (overlap between CA3 and EC representations)
        ca3_norm = np.linalg.norm(ca3_input) + 1e-10
        ec_norm = np.linalg.norm(ec_input) + 1e-10
        match_score = float(np.dot(ca3_input, ec_input) / (ca3_norm * ec_norm))
        match_score = max(0.0, min(1.0, match_score))  # Clamp to [0, 1]

        is_match = match_score >= self.config.match_threshold
        novelty_signal = 1.0 - match_score

        if is_match:
            self._match_count += 1
        else:
            self._mismatch_count += 1

        self._novelty_history.append(novelty_signal)
        if len(self._novelty_history) > 1000:
            self._novelty_history = self._novelty_history[-500:]

        # Compute output pattern
        # On match: blend CA3 and EC equally
        # On mismatch: weight EC more (reality > prediction)
        if is_match:
            ec_weight = self.config.ec_weight
        else:
            ec_weight = 0.7  # Weight EC more on mismatch

        combined = (1 - ec_weight) * ca3_input + ec_weight * ec_input
        output_pattern = self._apply_sparsity(combined)

        # Mismatch-driven learning: update weights on mismatch
        if not is_match:
            self._mismatch_learning(ca3_input, ec_input)

        # Track sequence
        seq_position = self._update_sequence(output_pattern, event)

        return ComparisonResult(
            match_score=match_score,
            is_match=is_match,
            novelty_signal=novelty_signal,
            output_pattern=output_pattern,
            ca3_contribution=1 - ec_weight,
            ec_contribution=ec_weight,
            sequence_position=seq_position,
        )

    def _project_ca3(self, pattern: np.ndarray) -> np.ndarray:
        """Project CA3 pattern to CA1 space.

        Args:
            pattern: CA3 pattern.

        Returns:
            Projected pattern in CA1 space.
        """
        n = self.config.n_pyramidal_cells
        if len(pattern) != n:
            # Resize pattern
            if len(pattern) > n:
                pattern = pattern[:n]
            else:
                padded = np.zeros(n, dtype=pattern.dtype)
                padded[:len(pattern)] = pattern
                pattern = padded

        return np.tanh(np.dot(self._ca3_weights, pattern))

    def _project_ec(self, pattern: np.ndarray) -> np.ndarray:
        """Project EC pattern to CA1 space.

        Args:
            pattern: EC pattern.

        Returns:
            Projected pattern in CA1 space.
        """
        n = self.config.n_pyramidal_cells
        if len(pattern) != n:
            # Resize pattern
            if len(pattern) > n:
                pattern = pattern[:n]
            else:
                padded = np.zeros(n, dtype=pattern.dtype)
                padded[:len(pattern)] = pattern
                pattern = padded

        return np.tanh(np.dot(self._ec_weights, pattern))

    def _apply_sparsity(self, pattern: np.ndarray) -> np.ndarray:
        """Apply sparsity constraint to output pattern.

        Args:
            pattern: Dense activation pattern.

        Returns:
            Sparse binary output pattern.
        """
        n_active = int(self.config.n_pyramidal_cells * self.config.output_sparsity)

        # Winner-take-all: keep top n_active activations
        if np.any(pattern > 0):
            threshold = np.percentile(pattern[pattern > 0], 100 * (1 - self.config.output_sparsity))
            output = (pattern > threshold).astype(np.float32)
        else:
            output = np.zeros_like(pattern)

        return output

    def _mismatch_learning(
        self, ca3_input: np.ndarray, ec_input: np.ndarray
    ) -> None:
        """Update weights based on mismatch.

        When CA3 prediction doesn't match EC reality, we need to:
        1. Adjust CA3 weights to better predict this EC pattern
        2. Strengthen the EC pathway

        Args:
            ca3_input: Projected CA3 pattern.
            ec_input: Projected EC pattern.
        """
        error = ec_input - ca3_input

        # Hebbian update: strengthen connections that reduce error
        outer_product = np.outer(ec_input, error)
        self._ca3_weights += self.config.mismatch_learning_rate * outer_product

        # Normalize to prevent weight explosion
        max_weight = np.max(np.abs(self._ca3_weights))
        if max_weight > 1.0:
            self._ca3_weights /= max_weight

    def _update_sequence(
        self, pattern: np.ndarray, event: SpatialEvent
    ) -> Optional[int]:
        """Update sequence tracking.

        Detects and stores temporal sequences of patterns.

        Args:
            pattern: Current CA1 output pattern.
            event: Current spatial event.

        Returns:
            Position in current sequence, or None if not in sequence.
        """
        current_time = event.timestamp

        # Check if this continues the current sequence
        if (
            self._last_timestamp is not None
            and current_time - self._last_timestamp < self.config.temporal_window
        ):
            # Continue sequence
            position = len(self._current_sequence)
            element = SequenceElement(
                pattern=pattern.copy(),
                event=event,
                timestamp=current_time,
                position=position,
            )
            self._current_sequence.append(element)
            self._last_timestamp = current_time
            return position
        else:
            # Start new sequence
            if len(self._current_sequence) > 2:
                # Save completed sequence
                self._sequences.append(self._current_sequence)
                if len(self._sequences) > 100:
                    self._sequences = self._sequences[-50:]

            self._current_sequence = [
                SequenceElement(
                    pattern=pattern.copy(),
                    event=event,
                    timestamp=current_time,
                    position=0,
                )
            ]
            self._last_timestamp = current_time
            return 0

    def predict_next(
        self, current_pattern: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:
        """Predict the next pattern in a sequence.

        Uses learned sequences to predict what comes next.

        Args:
            current_pattern: Current CA1 pattern.

        Returns:
            Tuple of (predicted_pattern or None, confidence).
        """
        if not self._sequences:
            return None, 0.0

        best_prediction = None
        best_confidence = 0.0

        for sequence in self._sequences:
            for i, element in enumerate(sequence[:-1]):
                # Check if current matches this sequence element
                overlap = np.sum(current_pattern * element.pattern) / (
                    np.sum(current_pattern) + np.sum(element.pattern) + 1e-10
                )

                if overlap > best_confidence:
                    best_confidence = float(overlap)
                    best_prediction = sequence[i + 1].pattern.copy()

        return best_prediction, best_confidence

    def get_familiarity(self, pattern: np.ndarray) -> float:
        """Compute familiarity of a pattern.

        Based on how well it matches known sequences and patterns.

        Args:
            pattern: Pattern to evaluate.

        Returns:
            Familiarity score (0 = novel, 1 = very familiar).
        """
        if not self._sequences:
            return 0.0

        max_familiarity = 0.0

        for sequence in self._sequences:
            for element in sequence:
                overlap = np.sum(pattern * element.pattern) / (
                    np.sum(pattern) + np.sum(element.pattern) + 1e-10
                )
                max_familiarity = max(max_familiarity, float(overlap))

        return max_familiarity

    @property
    def n_sequences(self) -> int:
        """Number of stored sequences."""
        return len(self._sequences)

    @property
    def statistics(self) -> Dict:
        """Get CA1 statistics.

        Returns:
            Dictionary with comparison statistics.
        """
        return {
            "total_comparisons": self._total_comparisons,
            "match_count": self._match_count,
            "mismatch_count": self._mismatch_count,
            "match_rate": self._match_count / max(1, self._total_comparisons),
            "n_sequences": len(self._sequences),
            "mean_novelty": float(np.mean(self._novelty_history)) if self._novelty_history else 0.0,
            "recent_novelty": float(np.mean(self._novelty_history[-10:])) if self._novelty_history else 0.0,
        }

    def reset(self) -> None:
        """Reset CA1 to initial state."""
        n = self.config.n_pyramidal_cells

        self._ca3_weights = self._rng.normal(0, 0.1, (n, n)).astype(np.float32)
        self._ec_weights = self._rng.normal(0, 0.1, (n, n)).astype(np.float32)

        self._sequences = []
        self._current_sequence = []
        self._last_timestamp = None

        self._total_comparisons = 0
        self._match_count = 0
        self._mismatch_count = 0
        self._novelty_history = []
