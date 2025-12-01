"""CA1 region implementation with comparator functionality and cortical mapping.

CA1 is the primary output of the hippocampus that:
1. Receives predictions from CA3 (Schaffer collaterals)
2. Receives direct input from Entorhinal Cortex (perforant path)
3. Compares these inputs to detect match/mismatch
4. Outputs to subiculum and back to Entorhinal Cortex
5. Maps between HState and cortical SDR representations (TEM extension)

Key biological features:
- Comparator function: CA3 prediction vs EC reality
- Match = familiar, mismatch = novelty signal
- Temporal ordering and sequence learning
- Output to neocortex via subiculum

TEM/Monty Extensions:
- HState ↔ Cortical SDR bidirectional mapping
- Associative memory for pattern binding
- Prediction of cortical patterns from hippocampal states
- Inference of hippocampal states from cortical patterns
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import hashlib

import numpy as np

from .types import SpatialEvent

if TYPE_CHECKING:
    from .hstate import HState


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
        cortical_dim: Dimensionality of cortical SDR patterns.
        cortical_learning_rate: Learning rate for HState-cortical associations.
        cortical_sparsity: Sparsity level for predicted cortical patterns (0-1).
            Default 0.02 means ~2% of cortical neurons are active.
        n_cortical_retrievals: Number of top matches to consider in retrieval.
    """

    n_pyramidal_cells: int = 2500  # Similar to CA3
    n_active_cells: int = 50  # ~2% sparsity
    match_threshold: float = 0.6
    mismatch_learning_rate: float = 0.15
    temporal_window: float = 1.0
    ec_weight: float = 0.5  # Balance between EC and CA3
    output_sparsity: float = 0.02
    # Cortical mapping parameters
    cortical_dim: int = 2048  # Default SDR dimensionality
    cortical_learning_rate: float = 0.1
    cortical_sparsity: float = 0.02  # Sparsity for cortical pattern prediction
    n_cortical_retrievals: int = 5

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.cortical_sparsity < 1:
            raise ValueError(
                f"cortical_sparsity must be in range (0, 1), got {self.cortical_sparsity}"
            )
        if not 0 < self.output_sparsity < 1:
            raise ValueError(
                f"output_sparsity must be in range (0, 1), got {self.output_sparsity}"
            )


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
    """CA1 comparator network with cortical SDR mapping.

    CA1 implements a comparator that:
    - Compares CA3 predictions with direct EC input
    - Detects match (familiar) vs mismatch (novel)
    - Learns temporal sequences
    - Outputs combined representation to neocortex
    - Maps between HState and cortical SDR representations (TEM extension)

    The match/mismatch signal is crucial for:
    - Novelty detection (triggers encoding)
    - Expectation violation (prediction error)
    - Memory retrieval confirmation

    Cortical Mapping (TEM Extension):
    - learn_hstate_cortical_mapping(): Associate HState with cortical SDR
    - predict_cortical_pattern(): Generate cortical SDR from HState
    - infer_hstate_from_cortical(): Find HState matching a cortical SDR

    Example:
        >>> config = CA1Config(n_pyramidal_cells=1000)
        >>> ca1 = CA1(config)
        >>> # Compare CA3 prediction with EC reality
        >>> result = ca1.compare(ca3_pattern, ec_pattern, event)
        >>> if result.is_match:
        ...     print("Familiar location!")
        >>> else:
        ...     print(f"Novelty detected: {result.novelty_signal:.2f}")
        >>> # Map HState to cortical pattern
        >>> cortical_sdr = ca1.predict_cortical_pattern(hstate)
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

        # HState ↔ Cortical SDR mapping (Associative memory)
        # Store as list of (hstate_id, cortical_sdr, ca1_pattern) tuples
        self._cortical_associations: List[Tuple[str, np.ndarray, np.ndarray]] = []
        # Hebbian weights: CA1 pattern -> cortical SDR
        self._ca1_to_cortical_weights: Optional[np.ndarray] = None
        # Inverse mapping: cortical SDR -> CA1 pattern
        self._cortical_to_ca1_weights: Optional[np.ndarray] = None

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
    def n_cortical_associations(self) -> int:
        """Number of stored HState-cortical associations."""
        return len(self._cortical_associations)

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
            "n_cortical_associations": len(self._cortical_associations),
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

        # Reset cortical mapping
        self._cortical_associations = []
        self._ca1_to_cortical_weights = None
        self._cortical_to_ca1_weights = None

        self._total_comparisons = 0
        self._match_count = 0
        self._mismatch_count = 0
        self._novelty_history = []

    # ==================== HState ↔ Cortical SDR Mapping ====================

    def _get_hstate_id(self, hstate: Union["HState", str]) -> str:
        """Extract HState ID from HState object or string.

        Args:
            hstate: HState object or string ID.

        Returns:
            String HState ID.
        """
        if isinstance(hstate, str):
            return hstate
        return hstate.id

    def learn_hstate_cortical_mapping(
        self,
        hstate: Union["HState", str],
        cortical_sdr: np.ndarray,
        ca1_pattern: Optional[np.ndarray] = None,
    ) -> None:
        """Learn association between HState and cortical SDR.

        This creates a bidirectional mapping:
        - HState/CA1 pattern → cortical SDR (for prediction)
        - Cortical SDR → HState (for inference)

        Uses Hebbian learning to bind patterns.

        Args:
            hstate: HState object or HState ID.
            cortical_sdr: Cortical SDR pattern (sparse or dense).
            ca1_pattern: Optional CA1 pattern. If None, generates from HState.
        """
        hstate_id = self._get_hstate_id(hstate)
        cortical_sdr = np.asarray(cortical_sdr, dtype=np.float32).flatten()

        # Generate CA1 pattern if not provided
        if ca1_pattern is None:
            # Use deterministic hash via SHA256 for reproducibility
            seed = int.from_bytes(
                hashlib.sha256(hstate_id.encode()).digest()[:8], "little"
            )
            rng = np.random.default_rng(seed)
            ca1_pattern = np.zeros(self.config.n_pyramidal_cells, dtype=np.float32)
            active = rng.choice(
                self.config.n_pyramidal_cells,
                size=self.config.n_active_cells,
                replace=False
            )
            ca1_pattern[active] = 1.0
        else:
            ca1_pattern = np.asarray(ca1_pattern, dtype=np.float32).flatten()

        # Store association
        self._cortical_associations.append((hstate_id, cortical_sdr.copy(), ca1_pattern.copy()))

        # Update Hebbian weights
        self._update_cortical_weights(ca1_pattern, cortical_sdr)

    def _update_cortical_weights(
        self,
        ca1_pattern: np.ndarray,
        cortical_sdr: np.ndarray,
    ) -> None:
        """Update Hebbian weights for cortical mapping.

        Args:
            ca1_pattern: CA1 activation pattern.
            cortical_sdr: Cortical SDR pattern.
        """
        n_ca1 = len(ca1_pattern)
        n_cortical = len(cortical_sdr)
        lr = self.config.cortical_learning_rate

        # Initialize weights if needed
        if self._ca1_to_cortical_weights is None:
            self._ca1_to_cortical_weights = np.zeros((n_cortical, n_ca1), dtype=np.float32)
        if self._cortical_to_ca1_weights is None:
            self._cortical_to_ca1_weights = np.zeros((n_ca1, n_cortical), dtype=np.float32)

        # Resize if needed (for different cortical dimensions)
        if self._ca1_to_cortical_weights.shape != (n_cortical, n_ca1):
            new_weights = np.zeros((n_cortical, n_ca1), dtype=np.float32)
            min_c = min(self._ca1_to_cortical_weights.shape[0], n_cortical)
            min_n = min(self._ca1_to_cortical_weights.shape[1], n_ca1)
            new_weights[:min_c, :min_n] = self._ca1_to_cortical_weights[:min_c, :min_n]
            self._ca1_to_cortical_weights = new_weights

        if self._cortical_to_ca1_weights.shape != (n_ca1, n_cortical):
            new_weights = np.zeros((n_ca1, n_cortical), dtype=np.float32)
            min_n = min(self._cortical_to_ca1_weights.shape[0], n_ca1)
            min_c = min(self._cortical_to_ca1_weights.shape[1], n_cortical)
            new_weights[:min_n, :min_c] = self._cortical_to_ca1_weights[:min_n, :min_c]
            self._cortical_to_ca1_weights = new_weights

        # Hebbian update: strengthen connections between co-active units
        # CA1 → Cortical: outer product
        self._ca1_to_cortical_weights += lr * np.outer(cortical_sdr, ca1_pattern)

        # Cortical → CA1: outer product
        self._cortical_to_ca1_weights += lr * np.outer(ca1_pattern, cortical_sdr)

        # Normalize to prevent unbounded growth
        max_weight = max(
            np.max(np.abs(self._ca1_to_cortical_weights)),
            np.max(np.abs(self._cortical_to_ca1_weights))
        )
        if max_weight > 1.0:
            self._ca1_to_cortical_weights /= max_weight
            self._cortical_to_ca1_weights /= max_weight

    def predict_cortical_pattern(
        self,
        hstate: Union["HState", str],
        use_hebbian: bool = True,
    ) -> Optional[np.ndarray]:
        """Predict cortical SDR pattern from HState.

        Uses the learned HState → cortical mapping to generate
        the expected cortical activation pattern.

        Args:
            hstate: HState object or HState ID.
            use_hebbian: If True, use Hebbian weights for prediction.
                If False, use direct lookup in associations.

        Returns:
            Predicted cortical SDR pattern, or None if no association found.
        """
        hstate_id = self._get_hstate_id(hstate)

        if use_hebbian and self._ca1_to_cortical_weights is not None:
            # Find CA1 pattern for this HState
            ca1_pattern = None
            for stored_id, _, stored_ca1 in self._cortical_associations:
                if stored_id == hstate_id:
                    ca1_pattern = stored_ca1
                    break

            if ca1_pattern is not None:
                # Use Hebbian weights to predict cortical pattern
                predicted = self._ca1_to_cortical_weights @ ca1_pattern
                # Apply sparsity (winner-take-all)
                n_active = max(1, int(len(predicted) * self.config.cortical_sparsity))
                if n_active > 0:
                    threshold = np.partition(predicted, -n_active)[-n_active]
                    predicted = (predicted >= threshold).astype(np.float32)
                return predicted

        # Fall back to direct lookup
        for stored_id, cortical_sdr, _ in self._cortical_associations:
            if stored_id == hstate_id:
                return cortical_sdr.copy()

        return None

    def infer_hstate_from_cortical(
        self,
        cortical_sdr: np.ndarray,
        use_hebbian: bool = True,
        return_scores: bool = False,
    ) -> Union[Optional[str], Tuple[Optional[str], Dict[str, float]]]:
        """Infer HState ID from cortical SDR pattern.

        Uses the learned cortical → HState mapping to find the
        hippocampal state that best matches the cortical pattern.

        Args:
            cortical_sdr: Cortical SDR pattern to match.
            use_hebbian: If True, use Hebbian weights for inference.
            return_scores: If True, also return match scores for all candidates.

        Returns:
            If return_scores is False: Best matching HState ID, or None.
            If return_scores is True: Tuple of (best HState ID, dict of scores).
        """
        cortical_sdr = np.asarray(cortical_sdr, dtype=np.float32).flatten()
        scores: Dict[str, float] = {}

        if use_hebbian and self._cortical_to_ca1_weights is not None:
            # Use Hebbian weights to infer CA1 pattern
            inferred_ca1 = self._cortical_to_ca1_weights @ cortical_sdr

            # Match against stored associations
            for stored_id, _, stored_ca1 in self._cortical_associations:
                # Cosine similarity
                norm1 = np.linalg.norm(inferred_ca1)
                norm2 = np.linalg.norm(stored_ca1)
                if norm1 > 1e-10 and norm2 > 1e-10:
                    similarity = float(np.dot(inferred_ca1, stored_ca1) / (norm1 * norm2))
                else:
                    similarity = 0.0
                scores[stored_id] = similarity

        else:
            # Direct comparison with stored cortical patterns
            for stored_id, stored_cortical, _ in self._cortical_associations:
                # Cosine similarity
                norm1 = np.linalg.norm(cortical_sdr)
                norm2 = np.linalg.norm(stored_cortical)
                if norm1 > 1e-10 and norm2 > 1e-10:
                    similarity = float(np.dot(cortical_sdr, stored_cortical) / (norm1 * norm2))
                else:
                    similarity = 0.0
                scores[stored_id] = similarity

        if not scores:
            if return_scores:
                return None, {}
            return None

        best_id = max(scores.keys(), key=lambda k: scores[k])

        if return_scores:
            return best_id, scores
        return best_id

    def get_top_hstate_matches(
        self,
        cortical_sdr: np.ndarray,
        n: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Get top N HState matches for a cortical SDR.

        Args:
            cortical_sdr: Cortical SDR pattern to match.
            n: Number of top matches to return. Uses config default if None.

        Returns:
            List of (hstate_id, score) tuples, sorted by score descending.
        """
        if n is None:
            n = self.config.n_cortical_retrievals

        result = self.infer_hstate_from_cortical(
            cortical_sdr, return_scores=True
        )

        # When return_scores=True, we get a tuple
        if isinstance(result, tuple):
            _, scores = result
        else:
            return []

        if not scores:
            return []

        # Sort by score descending
        sorted_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_matches[:n]
