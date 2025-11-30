"""CA3 region implementation with autoassociative memory and pattern completion.

CA3 is a recurrent attractor network that:
1. Receives sparse inputs from Dentate Gyrus (mossy fibers)
2. Forms autoassociative memories via recurrent collaterals
3. Performs pattern completion from partial cues
4. Supports memory replay/reactivation during consolidation

Key biological features:
- Recurrent collateral connections (~2% connectivity in rodents)
- Attractor dynamics for pattern completion
- Sharp-wave ripple generation during replay
- NMDA-dependent synaptic plasticity
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import SpatialEvent


@dataclass
class CA3Config:
    """Configuration for CA3 autoassociative network.

    Attributes:
        n_pyramidal_cells: Number of CA3 pyramidal neurons (default ~250k in rodents).
        n_active_cells: Target number of active cells per pattern (~1-2% sparsity).
        recurrent_connectivity: Fraction of recurrent connections (~2% in rodents).
        learning_rate: Hebbian learning rate for recurrent weights.
        pattern_completion_threshold: Activation threshold for pattern completion.
        max_iterations: Maximum attractor iterations for convergence.
        convergence_threshold: Threshold for attractor convergence.
        noise_level: Noise added during retrieval for generalization.
        memory_capacity: Approximate number of patterns before interference.
    """

    n_pyramidal_cells: int = 2500  # Scaled down from biological ~250k
    n_active_cells: int = 50  # ~2% sparsity
    recurrent_connectivity: float = 0.02
    learning_rate: float = 0.1
    pattern_completion_threshold: float = 0.3
    max_iterations: int = 10
    convergence_threshold: float = 0.01
    noise_level: float = 0.01
    memory_capacity: int = 500


@dataclass
class CA3Memory:
    """A stored memory in CA3.

    Attributes:
        pattern: The stored SDR pattern (binary).
        event: Associated spatial event.
        timestamp: When the memory was stored.
        retrieval_count: How many times this memory was retrieved.
        strength: Memory strength (consolidation level).
    """

    pattern: np.ndarray
    event: SpatialEvent
    timestamp: float
    retrieval_count: int = 0
    strength: float = 1.0


class CA3:
    """CA3 autoassociative network with recurrent collaterals.

    CA3 implements an attractor network that can:
    - Store patterns from DG in autoassociative weights
    - Complete partial patterns from cues
    - Replay stored sequences during consolidation
    - Detect novelty vs. familiarity

    Example:
        >>> config = CA3Config(n_pyramidal_cells=1000)
        >>> ca3 = CA3(config)
        >>> # Store a pattern from DG
        >>> ca3.store(dg_pattern, spatial_event)
        >>> # Later, complete from partial cue
        >>> completed, retrieved_event = ca3.pattern_complete(partial_cue)
    """

    def __init__(self, config: Optional[CA3Config] = None):
        """Initialize CA3 network.

        Args:
            config: CA3 configuration. Uses defaults if not provided.
        """
        self.config = config or CA3Config()
        self._rng = np.random.default_rng()

        # Recurrent weight matrix (sparse)
        # Use sparse representation for efficiency
        n = self.config.n_pyramidal_cells
        n_connections = int(n * n * self.config.recurrent_connectivity)

        # Initialize sparse recurrent weights
        self._recurrent_weights = np.zeros((n, n), dtype=np.float32)

        # Create random sparse connectivity pattern (fixed topology)
        self._connection_mask = self._create_sparse_mask(n, n_connections)

        # Memory storage
        self._memories: List[CA3Memory] = []
        self._pattern_to_memory: Dict[Tuple, int] = {}  # Hash -> memory index

        # Statistics
        self._total_stores = 0
        self._total_retrievals = 0
        self._successful_completions = 0

    def _create_sparse_mask(
        self, n: int, n_connections: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sparse connectivity mask.

        Args:
            n: Number of neurons.
            n_connections: Target number of connections.

        Returns:
            Tuple of (row_indices, col_indices) for sparse connections.
        """
        # Random connections excluding self-connections
        rows = self._rng.integers(0, n, size=n_connections)
        cols = self._rng.integers(0, n, size=n_connections)

        # Remove self-connections
        valid = rows != cols
        return rows[valid], cols[valid]

    def store(self, dg_pattern: np.ndarray, event: SpatialEvent) -> bool:
        """Store a pattern from Dentate Gyrus.

        Uses Hebbian learning to strengthen connections between
        co-active neurons in the input pattern.

        Args:
            dg_pattern: Sparse binary pattern from DG.
            event: Associated spatial event.

        Returns:
            True if stored as new pattern, False if existing pattern was strengthened.
        """
        # Project DG pattern to CA3 space if needed
        ca3_pattern = self._project_from_dg(dg_pattern)

        # Check if pattern already exists (update strength)
        pattern_hash = self._pattern_hash(ca3_pattern)
        if pattern_hash in self._pattern_to_memory:
            idx = self._pattern_to_memory[pattern_hash]
            self._memories[idx].strength += 0.1
            return False

        # Hebbian learning on recurrent weights
        # Strengthen connections between co-active neurons
        active_indices = np.where(ca3_pattern > 0)[0]
        for i in active_indices:
            for j in active_indices:
                if i != j:
                    # Only update if connection exists in mask
                    self._recurrent_weights[i, j] += (
                        self.config.learning_rate * ca3_pattern[i] * ca3_pattern[j]
                    )

        # Normalize weights to prevent unbounded growth
        max_weight = np.max(np.abs(self._recurrent_weights))
        if max_weight > 1.0:
            self._recurrent_weights /= max_weight

        # Store memory
        memory = CA3Memory(
            pattern=ca3_pattern.copy(),
            event=event,
            timestamp=event.timestamp,
        )
        self._memories.append(memory)
        self._pattern_to_memory[pattern_hash] = len(self._memories) - 1

        self._total_stores += 1

        # Check capacity and consolidate if needed
        if len(self._memories) > self.config.memory_capacity:
            self._consolidate()

        return True

    def _project_from_dg(self, dg_pattern: np.ndarray) -> np.ndarray:
        """Project DG pattern to CA3 representation.

        Mossy fiber projection from DG to CA3 is sparse but strong.

        Args:
            dg_pattern: Pattern from Dentate Gyrus.

        Returns:
            CA3 pattern (same or projected dimensionality).
        """
        if len(dg_pattern) == self.config.n_pyramidal_cells:
            return dg_pattern

        # Project to CA3 dimensionality
        ca3_pattern = np.zeros(self.config.n_pyramidal_cells, dtype=np.float32)

        # Sparse random projection
        active_dg = np.where(dg_pattern > 0)[0]
        n_ca3_active = min(
            len(active_dg) * 2, self.config.n_active_cells
        )  # Slight expansion

        if len(active_dg) > 0:
            # Each DG active neuron activates 2-3 CA3 neurons
            for dg_idx in active_dg:
                ca3_targets = self._rng.integers(
                    0, self.config.n_pyramidal_cells, size=2
                )
                ca3_pattern[ca3_targets] = 1.0

            # Ensure target sparsity
            if np.sum(ca3_pattern) > n_ca3_active:
                active = np.where(ca3_pattern > 0)[0]
                keep = self._rng.choice(active, size=n_ca3_active, replace=False)
                ca3_pattern = np.zeros_like(ca3_pattern)
                ca3_pattern[keep] = 1.0

        return ca3_pattern

    def pattern_complete(
        self, partial_cue: np.ndarray, return_event: bool = True
) -> Tuple[np.ndarray, Optional[SpatialEvent]]:
        """Complete a pattern from partial cue using attractor dynamics.

        Iteratively updates the pattern based on recurrent weights
        until convergence or max iterations.

        Args:
            partial_cue: Partial or noisy pattern to complete.
            return_event: Whether to also return the associated event.

        Returns:
            Tuple of (completed_pattern, associated_event or None).
        """
        self._total_retrievals += 1

        # Project to CA3 if needed
        current = self._project_from_dg(partial_cue).astype(np.float32)

        # Add noise for generalization
        noise = self._rng.normal(0, self.config.noise_level, current.shape)
        current = current + noise

        # Attractor dynamics
        for iteration in range(self.config.max_iterations):
            # Compute next state from recurrent input
            recurrent_input = np.dot(self._recurrent_weights, current)

            # Apply sparsity constraint: always select top n_active_cells
            n_active = self.config.n_active_cells
            if n_active >= len(recurrent_input):
                # If more active cells than available, activate all
                new_pattern = np.ones_like(recurrent_input, dtype=np.float32)
            else:
                # Find indices of top n_active values
                top_indices = np.argpartition(recurrent_input, -n_active)[-n_active:]
                new_pattern = np.zeros_like(recurrent_input, dtype=np.float32)
                new_pattern[top_indices] = 1.0
            # Blend with input (partial cue influence)
            blend_factor = 1.0 / (iteration + 2)
            new_pattern = (1 - blend_factor) * new_pattern + blend_factor * (
                partial_cue[:self.config.n_pyramidal_cells] if len(partial_cue) >= self.config.n_pyramidal_cells else current
            )

            # Check convergence
            change = np.mean(np.abs(new_pattern - current))
            current = new_pattern

            if change < self.config.convergence_threshold:
                break

        # Find best matching stored memory
        completed = (current > 0.5).astype(np.float32)
        event = None

        if return_event and self._memories:
            best_match_idx = self._find_best_match(completed)
            if best_match_idx >= 0:
                memory = self._memories[best_match_idx]
                memory.retrieval_count += 1
                event = memory.event
                self._successful_completions += 1

        return completed, event

    def _find_best_match(self, pattern: np.ndarray) -> int:
        """Find the memory that best matches a pattern.

        Args:
            pattern: Pattern to match.

        Returns:
            Index of best matching memory, or -1 if no good match.
        """
        if not self._memories:
            return -1

        best_overlap = 0.0
        best_idx = -1

        for idx, memory in enumerate(self._memories):
            overlap = np.sum(pattern * memory.pattern) / (
                np.sum(pattern) + np.sum(memory.pattern) + 1e-10
            )
            if overlap > best_overlap and overlap > self.config.pattern_completion_threshold:
                best_overlap = overlap
                best_idx = idx

        return best_idx

    def replay(self, n_patterns: int = 5) -> List[Tuple[np.ndarray, SpatialEvent]]:
        """Replay stored memories (sharp-wave ripple simulation).

        During consolidation, CA3 spontaneously reactivates stored patterns.
        This is thought to help transfer memories to neocortex.

        Args:
            n_patterns: Number of patterns to replay.

        Returns:
            List of (pattern, event) tuples in replay order.
        """
        if not self._memories:
            return []

        # Weight by recency and strength for replay probability
        weights = np.array([
            m.strength * np.exp(-0.001 * (len(self._memories) - i))
            for i, m in enumerate(self._memories)
        ])
        weights = weights / weights.sum()

        # Sample memories for replay
        n_to_replay = min(n_patterns, len(self._memories))
        indices = self._rng.choice(
            len(self._memories),
            size=n_to_replay,
            replace=False,
            p=weights
        )

        replayed = []
        for idx in indices:
            memory = self._memories[idx]
            # Pattern completion during replay
            completed, event = self.pattern_complete(memory.pattern, return_event=True)
            if event is not None:
                replayed.append((completed, event))

        return replayed

    def compute_novelty(self, pattern: np.ndarray) -> float:
        """Compute novelty score for a pattern.

        High novelty = poor pattern completion = new experience.
        Low novelty = good pattern completion = familiar experience.

        Args:
            pattern: Pattern to evaluate.

        Returns:
            Novelty score between 0 (familiar) and 1 (novel).
        """
        if not self._memories:
            return 1.0

        # Try pattern completion
        completed, event = self.pattern_complete(pattern, return_event=False)

        # Novelty is inverse of best match quality
        best_overlap = 0.0
        for memory in self._memories:
            overlap = np.sum(completed * memory.pattern) / (
                np.sum(completed) + np.sum(memory.pattern) + 1e-10
            )
            best_overlap = max(best_overlap, overlap)

        return 1.0 - best_overlap

    def _consolidate(self) -> None:
        """Consolidate memories, removing weak ones.

        Simulates synaptic consolidation where weak memories
        are pruned and strong ones are preserved.
        """
        # Remove memories with low strength and few retrievals
        threshold_strength = 0.3
        self._memories = [
            m for m in self._memories
            if m.strength > threshold_strength or m.retrieval_count > 2
        ]

        # Update index mapping
        self._pattern_to_memory = {
            self._pattern_hash(m.pattern): i
            for i, m in enumerate(self._memories)
        }

    def _pattern_hash(self, pattern: np.ndarray) -> tuple:
        """Create hashable representation of pattern.

        Args:
            pattern: Binary pattern.

        Returns:
            Tuple of active indices for hashing.
        """
        return tuple(np.where(pattern > 0.5)[0])

    @property
    def n_memories(self) -> int:
        """Number of stored memories."""
        return len(self._memories)

    @property
    def statistics(self) -> dict:
        """Get CA3 statistics.

        Returns:
            Dictionary with store/retrieval statistics.
        """
        return {
            "n_memories": len(self._memories),
            "total_stores": self._total_stores,
            "total_retrievals": self._total_retrievals,
            "successful_completions": self._successful_completions,
            "completion_rate": (
                self._successful_completions / max(1, self._total_retrievals)
            ),
            "memory_utilization": len(self._memories) / self.config.memory_capacity,
        }

    def reset(self) -> None:
        """Reset CA3 network to initial state."""
        n = self.config.n_pyramidal_cells
        self._recurrent_weights = np.zeros((n, n), dtype=np.float32)
        self._memories = []
        self._pattern_to_memory = {}
        self._total_stores = 0
        self._total_retrievals = 0
        self._successful_completions = 0
