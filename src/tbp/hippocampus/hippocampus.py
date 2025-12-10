# Copyright 2025 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""High-level Hippocampus API for Universal Relational Map System.

This module provides a unified interface to the hippocampus that integrates:
- Entorhinal Cortex (EC): Basis code encoding (spatial and abstract)
- Dentate Gyrus (DG): Pattern separation
- CA3: Autoassociative memory with transition graph (SR foundation)
- CA1: Comparator and cortical mapping

The Hippocampus class implements a complete TEM/SR-inspired system that can:
- Encode spatial AND abstract events into HStates
- Learn and predict state transitions
- Map between hippocampal and cortical representations
- Support replay for consolidation and imagination

This is the primary entry point for integrating hippocampal computation
with Monty cortical columns.

Example:
    >>> from tbp.hippocampus import Hippocampus, SpatialEvent
    >>> import numpy as np
    >>>
    >>> # Create hippocampus
    >>> hipp = Hippocampus()
    >>>
    >>> # Encode a spatial event
    >>> event = SpatialEvent(
    ...     timestamp=time.time(),
    ...     location=np.array([1.0, 2.0, 0.5]),
    ...     orientation=np.eye(3),
    ...     source_id="lm_0",
    ...     confidence=0.9,
    ... )
    >>> hstate = hipp.encode_event(event)
    >>>
    >>> # Predict future states
    >>> future = hipp.predict_next_hstates(n=5)
    >>>
    >>> # Get cortical prediction
    >>> cortical_sdr = hipp.predict_cortical_future(steps=3)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .basis import BasisCode, create_spatial_basis
from .ca1 import CA1, CA1Config
from .ca3 import CA3, CA3Config
from .dentate_gyrus import DentateGyrus, DGConfig
from .entorhinal import EntorhinalCortex
from .hstate import HState
from .memory import EpisodicMemory
from .types import SpatialEvent

logger = logging.getLogger(__name__)


@dataclass
class HippocampusConfig:
    """Configuration for the unified Hippocampus system.

    Attributes:
        n_pyramidal_cells: Number of pyramidal cells in CA3/CA1.
        n_granule_cells: Number of granule cells in DG.
        n_place_cells: Number of place cells in EC.
        n_grid_cells: Number of grid cells in EC (per module).
        n_grid_modules: Number of grid modules with different scales.
        dg_sparsity: Target sparsity for DG output.
        memory_capacity: Maximum number of patterns in CA3.
        sr_gamma: Discount factor for SR predictions.
        cortical_dim: Dimensionality of cortical SDRs.
        enable_replay: Whether to enable automatic replay.
        replay_interval: Steps between automatic replay.
        context_tag: Default context tag for new states.
    """

    n_pyramidal_cells: int = 2500
    n_granule_cells: int = 500
    n_place_cells: int = 100
    n_grid_cells: int = 50
    n_grid_modules: int = 8
    dg_sparsity: float = 0.005
    memory_capacity: int = 500
    sr_gamma: float = 0.9
    cortical_dim: int = 2048
    enable_replay: bool = True
    replay_interval: int = 100
    context_tag: Optional[str] = None


class Hippocampus:
    """Unified Hippocampus System for Universal Relational Mapping.

    This class provides a high-level API that integrates all hippocampal
    components (EC, DG, CA3, CA1) into a coherent system for:

    1. **Encoding**: Convert spatial/abstract events into HStates
    2. **Transition Learning**: Build SR-style transition graphs
    3. **Prediction**: Forecast future states and cortical patterns
    4. **Retrieval**: Pattern completion and association recall
    5. **Replay**: Consolidation and creative recombination

    The design is compatible with:
    - Hippocampal anatomy (DG, CA3, CA1)
    - Successor Representation (SR) theory
    - Tolman–Eichenbaum Machine (TEM) principles
    - Thousand Brains Theory / Monty architecture

    Attributes:
        config: HippocampusConfig with system parameters.
        ec: EntorhinalCortex for spatial encoding.
        dg: DentateGyrus for pattern separation.
        ca3: CA3 autoassociative memory with recurrent dynamics.
        ca1: CA1 comparator with cortical mapping.
        basis: BasisCode for encoding coordinates/features.
    """

    def __init__(
            self,
            config: Optional[HippocampusConfig] = None,
            basis: Optional[BasisCode] = None,
    ) -> None:
        """Initialize Hippocampus system.

        Args:
            config: System configuration. Uses defaults if None.
            basis: BasisCode for encoding. Creates MetricGridBasis if None.
        """
        self.config = config or HippocampusConfig()

        # Initialize basis code first so DG can use its dimensionality
        if basis is not None:
            self.basis = basis
        else:
            self.basis = create_spatial_basis(
                n_modules=self.config.n_grid_modules,
                n_place_cells=self.config.n_place_cells,
            )

        # Initialize components
        self._init_ec()
        self._init_dg()
        self._init_ca3()
        self._init_ca1()

        # Context tracking
        self._current_context: Optional[str] = self.config.context_tag
        self._context_history: List[str] = []

        # Step counter for replay scheduling
        self._step_count = 0

        # HState storage and current state tracking (CA1-decoded)
        self._hstates: Dict[str, HState] = {}
        self._current_hstate_id: Optional[str] = None
        # Optional per-HState scalar values for goal/value modulation
        self._hstate_values: Dict[str, float] = {}
        # Last CA3 pattern for simple transition learning
        self._last_ca3_pattern: Optional[np.ndarray] = None

    def _init_ec(self) -> None:
        """Initialize Entorhinal Cortex."""
        self.memory = EpisodicMemory()
        self.ec = EntorhinalCortex(
            memory=self.memory,
            n_place_cells=self.config.n_place_cells,
            n_grid_cells=self.config.n_grid_cells,
        )

    def _init_dg(self) -> None:
        """Initialize Dentate Gyrus."""
        # Use basis encoder dimensionality as DG input space
        dg_config = DGConfig(
            n_input=self.basis.output_dim,
            n_granule_cells=self.config.n_granule_cells,
            sparsity=self.config.dg_sparsity,
        )
        self.dg = DentateGyrus(dg_config)

    def _init_ca3(self) -> None:
        """Initialize CA3."""
        ca3_config = CA3Config(
            n_pyramidal_cells=self.config.n_pyramidal_cells,
            memory_capacity=self.config.memory_capacity,
            sr_gamma=self.config.sr_gamma,
        )
        self.ca3 = CA3(ca3_config)

    def _init_ca1(self) -> None:
        """Initialize CA1."""
        ca1_config = CA1Config(
            n_pyramidal_cells=self.config.n_pyramidal_cells,
            cortical_dim=self.config.cortical_dim,
        )
        self.ca1 = CA1(ca1_config)

    # ==================== State Access ====================

    def encode_event(
            self,
            event: SpatialEvent,
            cortical_sdr: Optional[np.ndarray] = None,
            context_tag: Optional[str] = None,
    ) -> HState:
        """Encode a spatial event into an HState.

        This is the primary entry point for encoding new experiences.
        The event flows through: EC → DG → CA3 → HState

        If a cortical SDR is provided, it will also be associated
        with the HState via CA1.

        Args:
            event: SpatialEvent to encode.
            cortical_sdr: Optional cortical SDR to associate.
            context_tag: Optional context tag (uses default if None).

        Returns:
            Encoded HState representing this experience.
        """
        self._step_count += 1
        ctx = context_tag or self._current_context

        # 1. Basis / EC encoding (universal coordinate embedding)
        basis_vector = self.basis.encode(event.location)

        # 2. DG pattern separation from basis space (DG SDR)
        dg_output = self.dg.encode(basis_vector)

        # 3. CA3 encoding/learning from DG SDR
        dg_sdr = dg_output.sparse_code
        ca3_pattern = self.ca3._project_from_dg(dg_sdr)
        # Autoassociative storage
        self.ca3.store(dg_sdr, event, ca3_pattern=ca3_pattern)

        # Simple Hebbian transition learning between successive CA3 patterns
        if self._last_ca3_pattern is not None:
            try:
                self.ca3.learn_transition(self._last_ca3_pattern, ca3_pattern)
            except ValueError:
                pass
        self._last_ca3_pattern = ca3_pattern.copy()

        # 4. CA1 comparator/decoder: integrate CA3 prediction with EC basis
        ca1_sdr = self.ca1.step(ca3_pattern, ec_input=basis_vector)

        # 5. Create/update HState using DG SDR (primary) and CA1 SDR (decoded)
        hstate = HState.from_dg_ca1(
            event=event,
            dg_sdr=dg_sdr,
            ca1_sdr=ca1_sdr,
            basis_vector=basis_vector,
            context_tag=ctx,
        )
        self._hstates[hstate.id] = hstate
        self._current_hstate_id = hstate.id

        # 6. Optional cortical association via CA1 (cortical SDR mapping)
        if cortical_sdr is not None:
            self.ca1.learn_hstate_cortical_mapping(
                hstate=hstate,
                cortical_sdr=cortical_sdr,
            )

        # 7. Periodic replay for consolidation
        if self.config.enable_replay and self._step_count % self.config.replay_interval == 0:
            self._automatic_replay()

        return hstate

    def current_hstate(self) -> Optional[HState]:
        """Get the current HState (most recently encoded).

        Returns:
            Current HState, or None if no state has been encoded.
        """
        if self._current_hstate_id is None:
            return None
        return self._hstates.get(self._current_hstate_id)

    def get_hstate(self, hstate_id: str) -> Optional[HState]:
        """Get an HState by its ID.

        Args:
            hstate_id: Unique identifier of the HState.

        Returns:
            HState if found, None otherwise.
        """
        return self._hstates.get(hstate_id)

    def update_hstate_value(self, hstate_id: str, value: float) -> None:
        """Update the estimated value of an HState.

        Stores a scalar value used later for local value modulation
        during replay/planning. Safe to call even if the HState does
        not exist (no-op in that case).
        """
        if hstate_id in self._hstates:
            self._hstate_values[hstate_id] = float(value)

    # ==================== Relational Queries ====================

    def predict_next_hstates(
            self,
            n: int = 5,
            from_hstate: Optional[Union[HState, str]] = None,
    ) -> List[HState]:
        """Predict next likely HStates from current or specified state.

        Uses a single CA3 dynamic step from the starting state and reads
        out the resulting attractor as a small set of candidate HStates.

        Args:
            n: Maximum number of predictions.
            from_hstate: Starting state. Uses current if None.

        Returns:
            List of predicted next HStates, sorted by probability.
        """
        if from_hstate is None:
            from_hstate = self.current_hstate()
            if from_hstate is None:
                return []

        # One dynamic CA3 step followed by CA1 decoding.
        trajectory = self.replay_forward(start_hstate=from_hstate, depth=1)
        return trajectory[:n]

    def predict_future_hstates(
            self,
            steps: int = 5,
            from_hstate: Optional[Union[HState, str]] = None,
            stochastic: bool = False,
    ) -> List[HState]:
        """Predict future HStates over multiple steps using CA3 dynamics.

        Args:
            steps: Number of future steps to predict.
            from_hstate: Starting state. Uses current if None.
            stochastic: Whether to sample stochastically.

        Returns:
            List of predicted future HStates.
        """
        if from_hstate is None:
            from_hstate = self.current_hstate()
            if from_hstate is None:
                return []

        # Use forward replay as a multi-step dynamic rollout
        return self.replay_forward(start_hstate=from_hstate, depth=steps)

    def predict_cortical_future(
            self,
            steps: int = 3,
            from_hstate: Optional[Union[HState, str]] = None,
    ) -> List[np.ndarray]:
        """Predict future cortical SDR patterns.

        Combines future state prediction with CA1 cortical mapping.

        Args:
            steps: Number of future steps.
            from_hstate: Starting state. Uses current if None.

        Returns:
            List of predicted cortical SDR patterns.
        """
        future_hstates = self.predict_future_hstates(steps, from_hstate)

        cortical_patterns = []
        for hstate in future_hstates:
            pattern = self.ca1.predict_cortical_pattern(hstate)
            if pattern is not None:
                cortical_patterns.append(pattern)

        return cortical_patterns

    # Transition probabilities and SR computation have been removed from the
    # mechanistic API; temporal structure now lives entirely in CA3 dynamics.

    # ==================== Abstract Task Support ====================

    def register_context(self, tag: str) -> None:
        """Register a new context tag for subsequent encodings.

        Different contexts can produce different basis codes and
        transition patterns.

        Args:
            tag: Context identifier (e.g., "navigation", "planning").
        """
        if self._current_context is not None:
            self._context_history.append(self._current_context)
        self._current_context = tag
        logger.debug(f"Registered context: {tag}")

    def get_context(self) -> Optional[str]:
        """Get current context tag."""
        return self._current_context

    def get_states_by_context(self, context_tag: str) -> List[HState]:
        """Get all HStates with a specific context tag.

        Args:
            context_tag: Context to filter by.

        Returns:
            List of matching HStates.
        """
        return [
            hstate for hstate in self._hstates.values()
            if hstate.context_tag == context_tag
        ]

    # ==================== Retrieval & Pattern Completion ====================

    def retrieve_from_pattern(
            self,
            partial_pattern: np.ndarray,
    ) -> Optional[HState]:
        """Retrieve HState from partial DG pattern cue (legacy helper).

        Uses CA3 pattern completion followed by nearest HState lookup
        in DG SDR space. This does not participate in replay/planning
        dynamics and is kept as a convenience utility.

        Args:
            partial_pattern: Partial or noisy pattern.

        Returns:
            Retrieved HState, or None if no match found.
        """
        completed, _ = self.ca3.pattern_complete(partial_pattern, return_event=False)
        if completed is None:
            return None

        # Match to stored HState by DG SDR overlap
        best_h: Optional[HState] = None
        best_overlap = 0.0
        active = set(np.where(completed > 0.5)[0])

        for h in self._hstates.values():
            h_set = set(h.sdr_indices)
            if not h_set:
                continue
            inter = len(active & h_set)
            union = len(active | h_set)
            if union == 0:
                continue
            overlap = inter / union
            if overlap > best_overlap:
                best_overlap = overlap
                best_h = h

        return best_h

    def retrieve_from_cortical(
            self,
            cortical_sdr: np.ndarray,
    ) -> Optional[HState]:
        """Retrieve HState from cortical SDR pattern.

        Uses CA1's cortical-to-hippocampal mapping.

        Args:
            cortical_sdr: Cortical SDR pattern.

        Returns:
            Best matching HState, or None if no match.
        """
        hstate_id = self.ca1.infer_hstate_from_cortical(cortical_sdr)
        if isinstance(hstate_id, tuple):
            hstate_id = hstate_id[0]
        if hstate_id is not None:
            return self._hstates.get(hstate_id)
        return None

    def retrieve_from_location(
            self,
            location: np.ndarray,
            threshold: float = 0.5,
    ) -> List[HState]:
        """Retrieve HStates near a spatial location.

        Args:
            location: 3D spatial coordinates.
            threshold: Maximum distance for matching.

        Returns:
            List of HStates near the location, sorted by distance.
        """
        matches: List[Tuple[HState, float]] = []

        for hstate in self._hstates.values():
            if hstate.is_spatial:
                assert hstate.spatial_location is not None
                dist = float(np.linalg.norm(hstate.spatial_location - location))
                if dist <= threshold:
                    matches.append((hstate, dist))

        matches.sort(key=lambda x: x[1])
        return [m[0] for m in matches]

    # ==================== Replay ====================

    def replay_forward(
            self,
            start_hstate: Optional[Union[HState, str]] = None,
            depth: int = 5,
    ) -> List[HState]:
        """Replay forward from current or specified state.

        Args:
            start_hstate: Starting state. Uses current if None.
            depth: Number of steps to replay.

        Returns:
            List of replayed HStates.
        """
        if start_hstate is None:
            start = self.current_hstate()
        elif isinstance(start_hstate, HState):
            start = start_hstate
        else:
            start = self.get_hstate(start_hstate)

        if start is None:
            return []

        # Derive initial CA3 pattern from stored DG SDR
        dg_pattern = start.to_pattern(self.config.n_granule_cells)
        x_ca3 = self.ca3._project_from_dg(dg_pattern)

        trajectory: List[HState] = []

        for _ in range(depth):
            x_ca3 = self.ca3.step(x_ca3)
            # CA1 comparator/decoder as exclusive readout
            x_ca1 = self.ca1.step(x_ca3, ec_input=None)
            h_next = self._decode_ca1_to_hstate(x_ca1)
            if h_next is None:
                break
            # Avoid trivial immediate self-loops
            if trajectory and h_next.id == trajectory[-1].id:
                break
            trajectory.append(h_next)

        return trajectory

    def replay_backward(
            self,
            end_hstate: Optional[Union[HState, str]] = None,
            depth: int = 5,
    ) -> List[HState]:
        """Replay backward to current or specified state.

        Args:
            end_hstate: Ending state. Uses current if None.
            depth: Number of steps to replay backward.

        Returns:
            List of predecessor HStates.
        """
        if end_hstate is None:
            end = self.current_hstate()
        elif isinstance(end_hstate, HState):
            end = end_hstate
        else:
            end = self.get_hstate(end_hstate)

        if end is None:
            return []

        dg_pattern = end.to_pattern(self.config.n_granule_cells)
        x_ca3 = self.ca3._project_from_dg(dg_pattern)

        trajectory: List[HState] = []

        for _ in range(depth):
            x_ca3 = self.ca3.step(x_ca3, reverse=True)
            x_ca1 = self.ca1.step(x_ca3, ec_input=None)
            h_prev = self._decode_ca1_to_hstate(x_ca1)
            if h_prev is None:
                break
            if trajectory and h_prev.id == trajectory[-1].id:
                break
            trajectory.append(h_prev)

        return trajectory

    def replay_recombine(
            self,
            hstate_a: Union[HState, str],
            hstate_b: Union[HState, str],
            max_path_length: int = 10,
    ) -> List[List[HState]]:
        """Explore novel paths between two states via mixed CA3 dynamics.

        Uses a noisy superposition of the DG→CA3 encodings for the two
        starting HStates and decodes each step through CA1.
        """
        if isinstance(hstate_a, HState):
            a = hstate_a
        else:
            a = self.get_hstate(hstate_a)

        if isinstance(hstate_b, HState):
            b = hstate_b
        else:
            b = self.get_hstate(hstate_b)

        if a is None or b is None:
            return []

        dg_a = a.to_pattern(self.config.n_granule_cells)
        dg_b = b.to_pattern(self.config.n_granule_cells)
        ca3_a = self.ca3._project_from_dg(dg_a)
        ca3_b = self.ca3._project_from_dg(dg_b)

        paths: List[List[HState]] = []
        n_rollouts = 3

        for _ in range(n_rollouts):
            noise = np.random.standard_normal(ca3_a.shape).astype(np.float32) * self.ca3.config.noise_level
            x_ca3 = np.clip(0.5 * (ca3_a + ca3_b) + noise, 0.0, 1.0)
            x_ca3 = self.ca3._k_wta(x_ca3, self.ca3.config.n_active_cells)

            path: List[HState] = []
            for _ in range(max_path_length):
                x_ca1 = self.ca1.step(x_ca3, ec_input=None)
                h = self._decode_ca1_to_hstate(x_ca1)
                if h is None:
                    break
                if path and h.id == path[-1].id:
                    break
                path.append(h)
                x_ca3 = self.ca3.step(x_ca3)

            if path and all(len(path) != len(p) or any(h1.id != h2.id for h1, h2 in zip(path, p))
                           for p in paths):
                paths.append(path)

        return paths

    def plan(
            self,
            start_hstate_id: str,
            goal_descriptor: Union[str, HState, np.ndarray, Dict[str, Any]],
            n_candidates: int = 5,
            max_len: int = 12,
    ) -> List[List[HState]]:
        """High-level planning API using hippocampal replay.

        Args:
            start_hstate_id: ID of the starting HState.
            goal_descriptor: Goal specification, which may be:
                - HState ID (string)
                - HState instance
                - Basis vector target (np.ndarray with basis.output_dim)
                - Spatial coordinates (np.ndarray of length 3)
                - Dict with keys describing goal ('hstate_id', 'basis_vector',
                  or 'location').
            n_candidates: Number of candidate trajectories to return.
            max_len: Maximum length of each trajectory.

        Returns:
            Ranked list of trajectories (best first), each a list of HStates.
        """
        start_hstate = self.get_hstate(start_hstate_id)
        if start_hstate is None:
            return []

        goal_hstate = self._resolve_goal_descriptor(goal_descriptor)
        if goal_hstate is None:
            return []

        # Goal-directed replay via CA3 dynamics + CA1 decoding
        paths: List[List[HState]] = []

        # Precompute goal modulation in CA3 space
        goal_mod = self._compute_goal_modulation(goal_hstate)
        value_mod = self._compute_value_modulation()

        dg_start = start_hstate.to_pattern(self.config.n_granule_cells)
        x0 = self.ca3._project_from_dg(dg_start)

        for _ in range(n_candidates):
            x_ca3 = x0.copy()
            path: List[HState] = []
            for _ in range(max_len):
                x_ca3 = self.ca3.step(x_ca3, goal_mod=goal_mod, value_mod=value_mod)
                x_ca1 = self.ca1.step(x_ca3, ec_input=None)
                h = self._decode_ca1_to_hstate(x_ca1)
                if h is None:
                    break
                if path and h.id == path[-1].id:
                    break
                path.append(h)
                if h.id == goal_hstate.id:
                    break
            if path:
                paths.append(path)

        return paths

    def _resolve_goal_descriptor(
            self,
            goal_descriptor: Union[str, HState, np.ndarray, Dict[str, Any]],
    ) -> Optional[HState]:
        """Resolve various goal descriptor formats to a concrete HState.

        The resolution strategy:
        - If string: treat as HState ID.
        - If HState: return as-is.
        - If dict: check keys 'hstate_id', 'basis_vector', 'location'.
        - If np.ndarray: treat as basis_vector if dimension matches,
          otherwise treat as spatial coordinates (length 3).
        """
        # Direct HState or ID
        if isinstance(goal_descriptor, HState):
            return goal_descriptor
        if isinstance(goal_descriptor, str):
            return self.get_hstate(goal_descriptor)

        # Dict-based descriptor
        if isinstance(goal_descriptor, dict):
            if "hstate_id" in goal_descriptor:
                return self.get_hstate(str(goal_descriptor["hstate_id"]))
            if "basis_vector" in goal_descriptor:
                vec = np.asarray(goal_descriptor["basis_vector"], dtype=np.float64)
                return self._nearest_hstate_in_basis(vec)
            if "location" in goal_descriptor:
                loc = np.asarray(goal_descriptor["location"], dtype=np.float64)
                basis_vec = self.basis.encode(loc)
                return self._nearest_hstate_in_basis(basis_vec)

        # Numpy array or sequence: basis vector or coordinates
        arr = np.asarray(goal_descriptor, dtype=np.float64)
        if arr.ndim == 1:
            if not hasattr(self.basis, "output_dim"):
                raise ValueError("Cannot resolve goal descriptor: self.basis is missing 'output_dim' attribute, so cannot determine if input is a basis vector or coordinates.")
            if arr.shape[0] == self.basis.output_dim:
                # Interpret as basis vector
                return self._nearest_hstate_in_basis(arr)
            if arr.shape[0] == 3:
                # Interpret as spatial coordinates
                basis_vec = self.basis.encode(arr)
                return self._nearest_hstate_in_basis(basis_vec)

        return None

    def _nearest_hstate_in_basis(self, target_basis: np.ndarray) -> Optional[HState]:
        """Find the stored HState whose basis_vector is closest to target."""
        target = np.asarray(target_basis, dtype=np.float64)
        best_hstate: Optional[HState] = None
        best_dist = float("inf")

        for hstate in self._hstates.values():
            if hstate.basis_vector is None:
                continue
            dist = float(np.linalg.norm(hstate.basis_vector - target))
            if dist < best_dist:
                best_dist = dist
                best_hstate = hstate

        return best_hstate

    def _decode_ca1_to_hstate(self, x_ca1: np.ndarray) -> Optional[HState]:
        """Decode a CA1 SDR back to the closest stored HState.

        Uses Jaccard overlap between active CA1 indices and each HState's
        stored ca1_indices. If no HState carries CA1 indices (e.g., legacy
        encodings), fall back to DG SDR overlap to avoid empty replays.
        """
        x = np.asarray(x_ca1, dtype=np.float32).ravel()
        active = set(np.where(x > 0.5)[0])
        if not active:
            return None

        best_h: Optional[HState] = None
        best_overlap = 0.0

        # First try CA1-index-based match
        for h in self._hstates.values():
            if h.ca1_indices is None:
                continue
            h_set = set(h.ca1_indices)
            if not h_set:
                continue
            inter = len(active & h_set)
            union = len(active | h_set)
            if union == 0:
                continue
            overlap = inter / union
            if overlap > best_overlap:
                best_overlap = overlap
                best_h = h

        if best_h is not None:
            return best_h

        # Fallback: DG-overlap match for legacy states without CA1 indices
        dg_size = self.config.n_granule_cells
        best_h = None
        best_overlap = 0.0
        for h in self._hstates.values():
            h_set = set(h.sdr_indices)
            if not h_set:
                continue
            inter = len(active & h_set)
            union = len(active | h_set)
            if union == 0:
                continue
            overlap = inter / union
            if overlap > best_overlap:
                best_overlap = overlap
                best_h = h

        return best_h

    def _compute_goal_modulation(self, goal: HState) -> np.ndarray:
        """Compute CA3 unit goal modulation from basis-space proximity."""
        goal_mod = np.zeros(self.ca3.config.n_pyramidal_cells, dtype=np.float32)
        if goal.basis_vector is None:
            return goal_mod

        for h in self._hstates.values():
            if h.basis_vector is None:
                continue
            dist = float(np.linalg.norm(h.basis_vector - goal.basis_vector))
            sim = 1.0 / (1.0 + dist)
            if sim <= 0.0:
                continue
            dg_pattern = h.to_pattern(self.config.n_granule_cells)
            ca3_pattern = self.ca3._project_from_dg(dg_pattern)
            goal_mod += sim * ca3_pattern

        if np.max(goal_mod) > 0:
            goal_mod /= float(np.max(goal_mod))
        return goal_mod

    def _compute_value_modulation(self) -> np.ndarray:
        """Compute CA3 unit value modulation from stored HState values."""
        value_mod = np.zeros(self.ca3.config.n_pyramidal_cells, dtype=np.float32)

        for hid, val in self._hstate_values.items():
            h = self._hstates.get(hid)
            if h is None or abs(val) < 1e-8:
                continue
            dg_pattern = h.to_pattern(self.config.n_granule_cells)
            ca3_pattern = self.ca3._project_from_dg(dg_pattern)
            value_mod += float(val) * ca3_pattern

        if np.max(np.abs(value_mod)) > 0:
            value_mod /= float(np.max(np.abs(value_mod)))

        return value_mod

    def _automatic_replay(self) -> None:
        """Perform automatic replay for consolidation."""
        # Replay recent trajectories
        if self.current_hstate() is not None:
            self.ca3.replay(n_patterns=3)
            logger.debug(f"Automatic replay at step {self._step_count}")

    # ==================== Comparison (CA1) ====================

    def compare(
            self,
            predicted_pattern: np.ndarray,
            observed_pattern: np.ndarray,
            event: SpatialEvent,
    ):
        """Compare predicted and observed patterns via CA1.

        Args:
            predicted_pattern: Expected pattern (from CA3).
            observed_pattern: Actual observed pattern (from EC).
            event: Associated spatial event.

        Returns:
            ComparisonResult with match/mismatch information.
        """
        return self.ca1.compare(predicted_pattern, observed_pattern, event)

    # ==================== Statistics & Diagnostics ====================

    @property
    def n_hstates(self) -> int:
        """Number of stored HStates."""
        return len(self._hstates)

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics.

        Returns:
            Dictionary with statistics from all components.
        """
        return {
            "n_hstates": len(self._hstates),
            "n_memories": self.ca3.n_memories,
            "n_episodes": len(self.memory),
            "n_cortical_associations": self.ca1.n_cortical_associations,
            "step_count": self._step_count,
            "current_context": self._current_context,
            "ca3": self.ca3.statistics,
            "ca1": self.ca1.statistics,
        }

    def reset(self) -> None:
        """Reset hippocampus to initial state.

        Clears all memories, HStates, and transitions.
        """
        self.ec.reset()
        self.dg.reset()
        self.ca3.reset()
        self.ca1.reset()
        self.memory.clear()
        self._current_context = self.config.context_tag
        self._context_history = []
        self._step_count = 0
        logger.info("Hippocampus reset")

    def __repr__(self) -> str:
        return (
            f"Hippocampus("
            f"hstates={self.n_hstates}, "
            f"context={self._current_context})"
        )
