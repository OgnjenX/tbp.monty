"""CA3 region implementation with autoassociative memory and pattern completion.

CA3 is a recurrent attractor network that:
1. Receives sparse inputs from Dentate Gyrus (mossy fibers)
2. Forms autoassociative memories via recurrent collaterals
3. Performs pattern completion from partial cues
4. Supports memory replay/reactivation during consolidation
5. Maintains transition graph for Successor Representation (SR)

Key biological features:
- Recurrent collateral connections (~2% connectivity in rodents)
- Attractor dynamics for pattern completion
- Sharp-wave ripple generation during replay
- NMDA-dependent synaptic plasticity
- Transition learning between sequential states (SR foundation)

TEM/SR Extensions:
- HState: Abstract latent states that can represent spatial OR abstract context
- Transition Graph: Directed graph tracking state transitions for prediction
- Multi-step prediction: predict_future() for SR-style lookahead
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from .types import SpatialEvent

if TYPE_CHECKING:
    from .hstate import HState


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
        transition_learning_rate: Learning rate for transition graph updates.
        transition_decay: Decay factor for old transitions (0=no decay, 1=full decay).
        sr_gamma: Discount factor for multi-step SR predictions (0-1).
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
    # SR/Transition graph parameters
    transition_learning_rate: float = 0.1
    transition_decay: float = 0.01
    sr_gamma: float = 0.9


@dataclass
class CA3Memory:
    """A stored memory in CA3.

    Attributes:
        pattern: The stored SDR pattern (binary).
        event: Associated spatial event.
        timestamp: When the memory was stored.
        retrieval_count: How many times this memory was retrieved.
        strength: Memory strength (consolidation level).
        hstate_id: Optional ID of associated HState.
        basis_vector: Optional EC basis vector for this memory.
    """

    pattern: np.ndarray
    event: SpatialEvent
    timestamp: float
    retrieval_count: int = 0
    strength: float = 1.0
    hstate_id: Optional[str] = None
    basis_vector: Optional[np.ndarray] = None


@dataclass
class TransitionEntry:
    """Entry in the transition graph.

    Attributes:
        count: Number of times this transition was observed.
        recency: Timestamp of most recent occurrence.
        strength: Learned transition strength (decays over time).
    """

    count: int = 0
    recency: float = 0.0
    strength: float = 0.0


class CA3:
    """CA3 autoassociative network with recurrent collaterals and transition graph.

    CA3 implements an attractor network that can:
    - Store patterns from DG in autoassociative weights
    - Complete partial patterns from cues
    - Replay stored sequences during consolidation
    - Detect novelty vs. familiarity
    - Learn and predict state transitions (SR foundation)
    - Generate HState latent representations

    The transition graph enables:
    - Successor Representation (SR) style predictions
    - Multi-step trajectory forecasting
    - Novel path discovery through recombination

    Example:
        >>> config = CA3Config(n_pyramidal_cells=1000)
        >>> ca3 = CA3(config)
        >>> # Store a pattern from DG
        >>> ca3.store(dg_pattern, spatial_event)
        >>> # Later, complete from partial cue
        >>> completed, retrieved_event = ca3.pattern_complete(partial_cue)
        >>> # Predict future states
        >>> future = ca3.predict_future(current_hstate, n_steps=5)
    """

    def __init__(self, config: Optional[CA3Config] = None, seed: Optional[int] = None):
        """Initialize CA3 network.

        Args:
            config: CA3 configuration. Uses defaults if not provided.
            seed: Random seed for reproducibility. If None, uses system entropy.
        """
        self.config = config or CA3Config()
        self._rng = np.random.default_rng(seed)

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

        # HState storage (id -> HState)
        self._hstates: Dict[str, HState] = {}
        self._pattern_to_hstate: Dict[Tuple, str] = {}  # Pattern hash -> HState id

        # Transition graph: sparse dict-of-dicts
        # transitions[from_id][to_id] = TransitionEntry
        self._transitions: Dict[str, Dict[str, TransitionEntry]] = {}

        # Current state tracking for transition learning
        self._previous_hstate_id: Optional[str] = None
        self._current_hstate_id: Optional[str] = None

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

    def store(
            self,
            dg_pattern: np.ndarray,
            event: SpatialEvent,
            ca3_pattern: Optional[np.ndarray] = None,
    ) -> bool:
        """Store a pattern from Dentate Gyrus.

        Uses Hebbian learning to strengthen connections between
        co-active neurons in the input pattern.

        Args:
            dg_pattern: Sparse binary pattern from DG.
            event: Associated spatial event.
            ca3_pattern: Optional precomputed CA3 pattern. If provided, bypasses
                projection to avoid redundant computation.

        Returns:
            True if stored as new pattern, False if existing pattern was strengthened.
        """
        # Project DG pattern to CA3 space if needed
        if ca3_pattern is None:
            ca3_pattern = self._project_from_dg(dg_pattern)
        else:
            ca3_pattern = np.asarray(ca3_pattern, dtype=np.float32)

        # Check if pattern already exists (update strength)
        pattern_hash = self._pattern_hash(ca3_pattern)
        if pattern_hash in self._pattern_to_memory:
            idx = self._pattern_to_memory[pattern_hash]
            self._memories[idx].strength += 0.1
            return False

        # Hebbian learning on recurrent weights
        # Strengthen connections between co-active neurons
        active_indices = np.nonzero(ca3_pattern > 0)[0]
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
        active_dg = np.nonzero(dg_pattern > 0)[0]
        n_ca3_active = min(
            len(active_dg) * 2, self.config.n_active_cells
        )  # Slight expansion

        if len(active_dg) > 0:
            # Each DG active neuron activates 2-3 CA3 neurons
            for _ in range(len(active_dg)):
                ca3_targets = self._rng.integers(
                    0, self.config.n_pyramidal_cells, size=2
                )
                ca3_pattern[ca3_targets] = 1.0

            # Ensure target sparsity
            if np.sum(ca3_pattern) > n_ca3_active:
                active = np.nonzero(ca3_pattern > 0)[0]
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
                partial_cue[:self.config.n_pyramidal_cells] if len(partial_cue)
                                                               >= self.config.n_pyramidal_cells else current
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

    # ==================== Extended Replay (TEM/SR-consistent) ====================

    def replay_forward(
            self,
            start_hstate: Union["HState", str],
            depth: int = 5,
            update_transitions: bool = True,
    ) -> List["HState"]:
        """Replay forward trajectory from starting HState.

        Uses the transition graph to generate a forward sequence,
        simulating prospective planning or prediction.

        Args:
            start_hstate: Starting HState or HState ID.
            depth: Maximum number of steps to replay.
            update_transitions: Whether to strengthen replayed transitions.

        Returns:
            List of HStates in forward replay order.
        """
        trajectory = self.predict_future(start_hstate, n_steps=depth, stochastic=False)

        if update_transitions and len(trajectory) > 0:
            # Strengthen replayed transitions
            prev_id = self._get_hstate_id(start_hstate)
            for hstate in trajectory:
                self.register_transition(prev_id, hstate.id, weight=0.5)  # Lower weight for replay
                prev_id = hstate.id

        return trajectory

    def replay_backward(
            self,
            end_hstate: Union["HState", str],
            depth: int = 5,
            update_transitions: bool = True,
    ) -> List["HState"]:
        """Replay backward trajectory to an ending HState.

        Finds predecessors in the transition graph and generates
        a reverse sequence, simulating retrospective recall.

        Args:
            end_hstate: Ending HState or HState ID.
            depth: Maximum number of steps to replay backward.
            update_transitions: Whether to strengthen replayed transitions.

        Returns:
            List of HStates in backward replay order (most recent predecessor first).
        """
        end_id = self._get_hstate_id(end_hstate)
        trajectory: List[HState] = []
        current_id = end_id

        for _ in range(depth):
            # Find predecessors (states that transition TO current)
            predecessors: List[Tuple[str, float]] = []
            for from_id, targets in self._transitions.items():
                if current_id in targets:
                    entry = targets[current_id]
                    predecessors.append((from_id, entry.strength))

            if not predecessors:
                break

            # Select best predecessor
            predecessors.sort(key=lambda x: x[1], reverse=True)
            best_pred_id = predecessors[0][0]

            if best_pred_id in self._hstates:
                trajectory.append(self._hstates[best_pred_id])

                if update_transitions:
                    # Strengthen the backward-traced transition
                    self.register_transition(best_pred_id, current_id, weight=0.5)

                current_id = best_pred_id
            else:
                break

        return trajectory

    def replay_recombine(
            self,
            hstate_a: Union["HState", str],
            hstate_b: Union["HState", str],
            max_path_length: int = 10,
    ) -> List[List["HState"]]:
        """Explore novel paths between two HStates (creativity/planning).

        Uses transition graph to find or imagine paths between states,
        enabling recombination of experiences for novel scenarios.

        This implements a key TEM principle: using relational structure
        to infer never-experienced trajectories.

        Args:
            hstate_a: First HState or HState ID.
            hstate_b: Second HState or HState ID.
            max_path_length: Maximum path length to explore.

        Returns:
            List of possible paths (each path is a list of HStates).
            Returns empty list if no paths found.
        """
        id_a = self._get_hstate_id(hstate_a)
        id_b = self._get_hstate_id(hstate_b)

        if id_a not in self._hstates or id_b not in self._hstates:
            return []

        paths: List[List[HState]] = []
        queue: List[Tuple[str, List[str]]] = [(id_a, [id_a])]
        visited_paths: set = set()

        while queue and len(paths) < 5:
            current_id, path = queue.pop(0)

            if self._try_add_path(current_id, path, id_b, paths):
                continue

            if len(path) >= max_path_length:
                continue

            self._explore_successors(current_id, path, queue, visited_paths)

        return paths

    def _try_add_path(
            self,
            current_id: str,
            path: List[str],
            target_id: str,
            paths: List[List["HState"]],
    ) -> bool:
        """Try to add a path if target is reached.

        Args:
            current_id: Current node ID in path.
            path: Current path as list of IDs.
            target_id: Target node ID.
            paths: List to append found paths to.

        Returns:
            True if target was reached and path was processed.
        """
        if current_id != target_id:
            return False

        hstate_path = [self._hstates[pid] for pid in path if pid in self._hstates]
        if len(hstate_path) == len(path):
            paths.append(hstate_path)
        return True

    def _explore_successors(
            self,
            current_id: str,
            path: List[str],
            queue: List[Tuple[str, List[str]]],
            visited_paths: set,
    ) -> None:
        """Explore successor nodes and add valid ones to queue.

        Args:
            current_id: Current node ID.
            path: Current path as list of IDs.
            queue: BFS queue to append to.
            visited_paths: Set of visited path tuples.
        """
        if current_id not in self._transitions:
            return

        successors = sorted(
            self._transitions[current_id].items(),
            key=lambda x: x[1].strength,
            reverse=True
        )[:3]

        for next_id, _ in successors:
            if next_id not in path:
                new_path = path + [next_id]
                path_key = tuple(new_path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    queue.append((next_id, new_path))

    def replay_sequence(
            self,
            hstate_sequence: List[Union["HState", str]],
            update_transitions: bool = True,
    ) -> List["HState"]:
        """Replay a specific sequence of HStates.

        Used for consolidation or teaching specific trajectories.

        Args:
            hstate_sequence: Sequence of HStates or HState IDs to replay.
            update_transitions: Whether to strengthen sequence transitions.

        Returns:
            List of successfully replayed HStates.
        """
        replayed: List[HState] = []

        for i, hstate in enumerate(hstate_sequence):
            hstate_id = self._get_hstate_id(hstate)

            if hstate_id in self._hstates:
                replayed.append(self._hstates[hstate_id])

                # Update transitions between consecutive states
                if update_transitions and i > 0:
                    prev_id = self._get_hstate_id(hstate_sequence[i - 1])
                    self.register_transition(prev_id, hstate_id, weight=0.3)

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
        completed, _ = self.pattern_complete(pattern, return_event=False)

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
        return tuple(np.nonzero(pattern > 0.5)[0])

    @property
    def n_memories(self) -> int:
        """Number of stored memories."""
        return len(self._memories)

    @property
    def n_hstates(self) -> int:
        """Number of stored HStates."""
        return len(self._hstates)

    @property
    def n_transitions(self) -> int:
        """Total number of learned transitions."""
        return sum(len(targets) for targets in self._transitions.values())

    @property
    def current_hstate(self) -> Optional["HState"]:
        """Current HState (most recently encoded/retrieved)."""
        if self._current_hstate_id is None:
            return None
        return self._hstates.get(self._current_hstate_id)

    @property
    def statistics(self) -> dict:
        """Get CA3 statistics.

        Returns:
            Dictionary with store/retrieval statistics.
        """
        return {
            "n_memories": len(self._memories),
            "n_hstates": len(self._hstates),
            "n_transitions": self.n_transitions,
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
        self._hstates = {}
        self._pattern_to_hstate = {}
        self._transitions = {}
        self._previous_hstate_id = None
        self._current_hstate_id = None
        self._total_stores = 0
        self._total_retrievals = 0
        self._successful_completions = 0

    # ==================== HState Encoding ====================

    def encode_to_hstate(
            self,
            dg_pattern: np.ndarray,
            event: SpatialEvent,
            basis_vector: np.ndarray,
            context_tag: Optional[str] = None,
            update_transitions: bool = True,
    ) -> "HState":
        """Encode a DG pattern and event into an HState.

        This is the primary encoding method that creates an HState from
        incoming sensory information. It:
        1. Projects DG pattern to CA3 space (once)
        2. Stores in autoassociative memory (using precomputed pattern)
        3. Creates/retrieves corresponding HState
        4. Updates transition graph if enabled

        Args:
            dg_pattern: Sparse binary pattern from Dentate Gyrus.
            event: Associated spatial event.
            basis_vector: EC basis code embedding for this state.
            context_tag: Optional context identifier.
            update_transitions: Whether to update transition graph.

        Returns:
            HState representing this encoded experience.
        """
        from .hstate import HState

        # Project DG pattern to CA3 space ONCE
        ca3_pattern = self._project_from_dg(dg_pattern)
        pattern_hash = self._pattern_hash(ca3_pattern)

        # Check if HState already exists for this pattern
        if pattern_hash in self._pattern_to_hstate:
            hstate_id = self._pattern_to_hstate[pattern_hash]
            hstate = self._hstates[hstate_id]

            # Update memory strength
            if pattern_hash in self._pattern_to_memory:
                idx = self._pattern_to_memory[pattern_hash]
                self._memories[idx].strength += 0.1

        else:
            # Create new HState
            hstate = HState.from_spatial_event(
                event=event,
                ca3_pattern=ca3_pattern,
                basis_vector=basis_vector,
                context_tag=context_tag,
            )

            # Store HState
            self._hstates[hstate.id] = hstate
            self._pattern_to_hstate[pattern_hash] = hstate.id

            # Store in autoassociative memory with precomputed CA3 pattern
            self.store(dg_pattern, event, ca3_pattern=ca3_pattern)

            # Update memory with HState reference
            if pattern_hash in self._pattern_to_memory:
                idx = self._pattern_to_memory[pattern_hash]
                self._memories[idx].hstate_id = hstate.id
                self._memories[idx].basis_vector = basis_vector.copy()

        # Update transition graph
        if update_transitions and self._current_hstate_id is not None:
            self.register_transition(self._current_hstate_id, hstate.id)

        # Update current state tracking
        self._previous_hstate_id = self._current_hstate_id
        self._current_hstate_id = hstate.id

        return hstate

    def retrieve_hstate(
            self,
            partial_cue: np.ndarray,
            update_current: bool = True,
    ) -> Optional["HState"]:
        """Retrieve HState from partial pattern cue.

        Uses pattern completion to find best matching HState.

        Args:
            partial_cue: Partial or noisy pattern for retrieval.
            update_current: Whether to update current state tracking.

        Returns:
            Retrieved HState, or None if no good match found.
        """
        completed, event = self.pattern_complete(partial_cue, return_event=True)

        if event is None:
            return None

        pattern_hash = self._pattern_hash(completed)

        if pattern_hash in self._pattern_to_hstate:
            hstate_id = self._pattern_to_hstate[pattern_hash]
            hstate = self._hstates.get(hstate_id)

            if hstate is not None and update_current:
                self._previous_hstate_id = self._current_hstate_id
                self._current_hstate_id = hstate.id

            return hstate

        return None

    def get_hstate_by_id(self, hstate_id: str) -> Optional["HState"]:
        """Get HState by its unique ID.

        Args:
            hstate_id: Unique identifier of HState.

        Returns:
            HState if found, None otherwise.
        """
        return self._hstates.get(hstate_id)

    # ==================== Transition Graph (SR Foundation) ====================

    def register_transition(
            self,
            from_hstate_id: str,
            to_hstate_id: str,
            weight: float = 1.0,
    ) -> None:
        """Register a transition between two HStates.

        Updates the transition graph with the observed transition.
        Uses incremental learning with decay.

        Args:
            from_hstate_id: Source HState ID.
            to_hstate_id: Target HState ID.
            weight: Weight of this transition observation.
        """
        if from_hstate_id not in self._transitions:
            self._transitions[from_hstate_id] = {}

        if to_hstate_id not in self._transitions[from_hstate_id]:
            self._transitions[from_hstate_id][to_hstate_id] = TransitionEntry()

        entry = self._transitions[from_hstate_id][to_hstate_id]
        entry.count += 1

        # Update recency with fallback logic
        # Recency represents the timestamp of the most recent occurrence (larger = more recent)
        if to_hstate_id in self._hstates:
            hstate = self._hstates[to_hstate_id]
            # Use HState timestamp if available, otherwise use current time
            if hasattr(hstate, "timestamp") and hstate.timestamp is not None:
                entry.recency = hstate.timestamp
            else:
                # Fallback to current time if timestamp not available
                import time
                entry.recency = time.time()
        else:
            # If HState not found, use current time
            import time
            entry.recency = time.time()

        # Incremental strength update with decay
        entry.strength = (
                (1 - self.config.transition_learning_rate) * entry.strength
                + self.config.transition_learning_rate * weight
        )

        # Apply decay to other transitions from this state
        for other_id, other_entry in self._transitions[from_hstate_id].items():
            if other_id != to_hstate_id:
                other_entry.strength *= (1 - self.config.transition_decay)

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

    def successors(
            self,
            hstate: Union["HState", str],
            min_strength: float = 0.0,
    ) -> List["HState"]:
        """Get successor HStates from transition graph.

        Args:
            hstate: HState or HState ID to get successors for.
            min_strength: Minimum transition strength to include.

        Returns:
            List of successor HStates, sorted by transition strength.
        """
        hstate_id = self._get_hstate_id(hstate)

        if hstate_id not in self._transitions:
            return []

        successors_list: List[Tuple[HState, float]] = []
        for to_id, entry in self._transitions[hstate_id].items():
            if entry.strength >= min_strength and to_id in self._hstates:
                successors_list.append((self._hstates[to_id], entry.strength))

        # Sort by strength descending
        successors_list.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in successors_list]

    def transition_probability(
            self,
            hstate: Union["HState", str],
    ) -> Dict[str, float]:
        """Get normalized transition probabilities from an HState.

        Args:
            hstate: HState or HState ID.

        Returns:
            Dictionary mapping successor HState IDs to probabilities.
        """
        hstate_id = self._get_hstate_id(hstate)

        if hstate_id not in self._transitions:
            return {}

        strengths = {
            to_id: entry.strength
            for to_id, entry in self._transitions[hstate_id].items()
            if to_id in self._hstates
        }

        total = sum(strengths.values())
        if total < 1e-10:
            return {}

        return {to_id: s / total for to_id, s in strengths.items()}

    def sample_next(
            self,
            hstate: Union["HState", str],
            stochastic: bool = True,
    ) -> Optional["HState"]:
        """Sample next HState from transition distribution.

        Args:
            hstate: Current HState or HState ID.
            stochastic: If True, sample probabilistically.
                If False, return most likely successor.

        Returns:
            Sampled successor HState, or None if no successors.
        """
        probs = self.transition_probability(hstate)

        if not probs:
            return None

        if stochastic:
            ids = list(probs.keys())
            ps = list(probs.values())
            chosen_id = self._rng.choice(ids, p=ps)
        else:
            chosen_id = max(probs.keys(), key=lambda k: probs[k])

        return self._hstates.get(chosen_id)

    def predict_future(
            self,
            hstate: Union["HState", str],
            n_steps: int = 5,
            stochastic: bool = False,
    ) -> List["HState"]:
        """Predict future HStates using SR-style multi-step prediction.

        Rolls out the transition graph for n_steps to predict
        likely future states.

        Args:
            hstate: Starting HState or HState ID.
            n_steps: Number of steps to predict.
            stochastic: Whether to sample stochastically.

        Returns:
            List of predicted future HStates (length <= n_steps).
        """
        current = hstate if isinstance(hstate, str) else hstate.id
        predictions = []

        for _ in range(n_steps):
            next_hstate = self.sample_next(current, stochastic=stochastic)
            if next_hstate is None:
                break
            predictions.append(next_hstate)
            current = next_hstate.id

        return predictions

    def compute_sr_vector(
            self,
            hstate: Union["HState", str],
            n_steps: int = 10,
            gamma: Optional[float] = None,
    ) -> np.ndarray:
        """Compute Successor Representation vector for an HState.

        The SR vector represents the expected future occupancy of each
        state when starting from the given state.

        Args:
            hstate: Starting HState or HState ID.
            n_steps: Number of steps for SR computation.
            gamma: Discount factor (uses config default if None).

        Returns:
            SR vector indexed by HState position in _hstates.
        """
        if gamma is None:
            gamma = self.config.sr_gamma

        hstate_id = self._get_hstate_id(hstate)
        hstate_ids = list(self._hstates.keys())
        n_states = len(hstate_ids)
        id_to_idx = {hid: i for i, hid in enumerate(hstate_ids)}

        if hstate_id not in id_to_idx:
            return np.zeros(n_states)

        # Initialize SR vector with identity (count self)
        sr = np.zeros(n_states)
        sr[id_to_idx[hstate_id]] = 1.0

        # Compute expected future occupancy
        current_dist = np.zeros(n_states)
        current_dist[id_to_idx[hstate_id]] = 1.0

        for step in range(1, n_steps + 1):
            discount = gamma ** step
            next_dist = np.zeros(n_states)

            for from_idx, from_id in enumerate(hstate_ids):
                if current_dist[from_idx] < 1e-10:
                    continue

                probs = self.transition_probability(from_id)
                for to_id, prob in probs.items():
                    if to_id in id_to_idx:
                        to_idx = id_to_idx[to_id]
                        next_dist[to_idx] += current_dist[from_idx] * prob

            sr += discount * next_dist
            current_dist = next_dist

        return sr
