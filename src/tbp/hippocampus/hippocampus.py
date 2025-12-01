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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .basis import BasisCode, MetricGridBasis, MetricGridConfig, create_spatial_basis
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
        ca3: CA3 autoassociative memory with transition graph.
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

        # Initialize components
        self._init_ec()
        self._init_dg()
        self._init_ca3()
        self._init_ca1()

        # Initialize basis code
        if basis is not None:
            self.basis = basis
        else:
            self.basis = create_spatial_basis(
                n_modules=self.config.n_grid_modules,
                n_place_cells=self.config.n_place_cells,
            )

        # Context tracking
        self._current_context: Optional[str] = self.config.context_tag
        self._context_history: List[str] = []

        # Step counter for replay scheduling
        self._step_count = 0

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
        dg_config = DGConfig(
            n_input=self.config.n_place_cells + self.config.n_grid_cells,
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

        # 1. EC encoding
        ec_activations = self.ec.receive_event(event)
        ec_pattern = np.concatenate([
            ec_activations["place_activations"],
            ec_activations["grid_activations"],
        ])

        # 2. DG pattern separation
        dg_output = self.dg.encode(ec_pattern)

        # 3. Basis code encoding
        basis_vector = self.basis.encode(event.location)

        # 4. CA3 encoding to HState
        hstate = self.ca3.encode_to_hstate(
            dg_pattern=dg_output.sparse_code,
            event=event,
            basis_vector=basis_vector,
            context_tag=ctx,
        )

        # 5. Optional cortical association via CA1
        if cortical_sdr is not None:
            self.ca1.learn_hstate_cortical_mapping(
                hstate=hstate,
                cortical_sdr=cortical_sdr,
            )

        # 6. Periodic replay for consolidation
        if self.config.enable_replay and self._step_count % self.config.replay_interval == 0:
            self._automatic_replay()

        return hstate

    def current_hstate(self) -> Optional[HState]:
        """Get the current HState (most recently encoded).

        Returns:
            Current HState, or None if no state has been encoded.
        """
        return self.ca3.current_hstate

    def get_hstate(self, hstate_id: str) -> Optional[HState]:
        """Get an HState by its ID.

        Args:
            hstate_id: Unique identifier of the HState.

        Returns:
            HState if found, None otherwise.
        """
        return self.ca3.get_hstate_by_id(hstate_id)

    # ==================== Relational Queries ====================

    def predict_next_hstates(
        self,
        n: int = 5,
        from_hstate: Optional[Union[HState, str]] = None,
    ) -> List[HState]:
        """Predict next likely HStates from current or specified state.

        Uses the transition graph to predict successors.

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

        return self.ca3.successors(from_hstate)[:n]

    def predict_future_hstates(
        self,
        steps: int = 5,
        from_hstate: Optional[Union[HState, str]] = None,
        stochastic: bool = False,
    ) -> List[HState]:
        """Predict future HStates over multiple steps.

        Uses SR-style multi-step prediction.

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

        return self.ca3.predict_future(from_hstate, n_steps=steps, stochastic=stochastic)

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

    def transition_probabilities(
        self,
        from_hstate: Optional[Union[HState, str]] = None,
    ) -> Dict[str, float]:
        """Get transition probabilities from current or specified state.

        Args:
            from_hstate: Starting state. Uses current if None.

        Returns:
            Dictionary mapping HState IDs to transition probabilities.
        """
        if from_hstate is None:
            from_hstate = self.current_hstate()
            if from_hstate is None:
                return {}

        return self.ca3.transition_probability(from_hstate)

    def compute_sr(
        self,
        from_hstate: Optional[Union[HState, str]] = None,
        n_steps: int = 10,
    ) -> np.ndarray:
        """Compute Successor Representation vector.

        Args:
            from_hstate: Starting state. Uses current if None.
            n_steps: Number of steps for SR computation.

        Returns:
            SR vector representing expected future state occupancy.
        """
        if from_hstate is None:
            from_hstate = self.current_hstate()
            if from_hstate is None:
                return np.array([])

        return self.ca3.compute_sr_vector(from_hstate, n_steps=n_steps)

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
            hstate for hstate in self.ca3._hstates.values()
            if hstate.context_tag == context_tag
        ]

    # ==================== Retrieval & Pattern Completion ====================

    def retrieve_from_pattern(
        self,
        partial_pattern: np.ndarray,
    ) -> Optional[HState]:
        """Retrieve HState from partial pattern cue.

        Uses CA3 pattern completion.

        Args:
            partial_pattern: Partial or noisy pattern.

        Returns:
            Retrieved HState, or None if no match found.
        """
        return self.ca3.retrieve_hstate(partial_pattern)

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
            return self.ca3.get_hstate_by_id(hstate_id)
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

        for hstate in self.ca3._hstates.values():
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
            start_hstate = self.current_hstate()
            if start_hstate is None:
                return []

        return self.ca3.replay_forward(start_hstate, depth)

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
            end_hstate = self.current_hstate()
            if end_hstate is None:
                return []

        return self.ca3.replay_backward(end_hstate, depth)

    def replay_recombine(
        self,
        hstate_a: Union[HState, str],
        hstate_b: Union[HState, str],
        max_path_length: int = 10,
    ) -> List[List[HState]]:
        """Explore novel paths between two states (creativity).

        Args:
            hstate_a: First HState.
            hstate_b: Second HState.
            max_path_length: Maximum path length.

        Returns:
            List of possible paths between the states.
        """
        return self.ca3.replay_recombine(hstate_a, hstate_b, max_path_length)

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
        return self.ca3.n_hstates

    @property
    def n_transitions(self) -> int:
        """Number of learned transitions."""
        return self.ca3.n_transitions

    @property
    def statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics.

        Returns:
            Dictionary with statistics from all components.
        """
        return {
            "n_hstates": self.ca3.n_hstates,
            "n_transitions": self.ca3.n_transitions,
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
            f"transitions={self.n_transitions}, "
            f"context={self._current_context})"
        )
