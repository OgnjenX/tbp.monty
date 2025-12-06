# Copyright 2025 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Hippocampus module - Universal Relational Map System.

This package implements hippocampal computations for spatial and abstract
relational learning, inspired by the Tolman-Eichenbaum Machine (TEM) and
Successor Representation (SR) theories. The system maintains a latent state
space (HState) and learns transition dynamics for prediction and navigation.

Key Concepts:
    HState: The core latent representation maintained by the hippocampus.
        Each HState captures the system's belief about "where it is" in a
        relational space (physical or abstract). HStates are identified by
        CA3 pattern hashes and contain spatial/abstract context embeddings.

    Transition Graph: CA3 maintains a directed graph of transitions between
        HStates. This enables computing successor representations (SR) and
        predicting future states given actions.

    Basis Codes: Generalized spatial encoding via pluggable basis systems.
        MetricGridBasis (default) uses grid cells for Euclidean space.
        PlaceBasis uses place cell population encoding.
        CombinedBasis combines multiple bases for hierarchical encoding.
        You may implement your own BasisCode subclass for graph-based relational tasks.

    Observation Model: CA1 learns bidirectional mappings between HStates
        and Monty cortical column SDRs, enabling perception-based localization
        and imagination of sensory consequences.

Components:
    Hippocampus: High-level API integrating all components. Provides:
        - encode_event(): Convert observations to HState
        - predict_next_hstates(): One-step prediction
        - predict_future_hstates(): Multi-step rollout
        - predict_cortical_future(): Predicted sensory SDRs
        - replay_*(): Replay for consolidation and planning

    Entorhinal Cortex (EC): Grid cells, place cells, spatial encoding.
        Encodes continuous coordinates into SDR format for DG/CA3.

    Dentate Gyrus (DG): Pattern separation with ultra-sparse encoding.
        Transforms EC input into orthogonalized patterns for CA3.

    CA3: Autoassociative memory with transition learning. Stores HStates,
        learns transition probabilities, supports pattern completion and
        successor representation computation.

    CA1: Comparator between CA3 predictions and EC input. Also learns
        cortical SDR associations for observation-based inference.

    EpisodicMemory: Timestamped event buffer for temporal context.

The core modules have NO dependencies on tbp.monty. Integration with Monty
is handled via adapters in tbp.hippocampus.adapters.

Example (spatial navigation):
    >>> from tbp.hippocampus import Hippocampus, HippocampusConfig, SpatialEvent
    >>> hippocampus = Hippocampus(HippocampusConfig())
    >>> 
    >>> # Encode current observation
    >>> event = SpatialEvent(x=1.0, y=2.0, z=0.0, object_id="table")
    >>> hstate = hippocampus.encode_event(event)
    >>> 
    >>> # Predict future states
    >>> futures = hippocampus.predict_future_hstates(steps=5)
    >>> for step, predictions in enumerate(futures):
    ...     print(f"Step {step}: {len(predictions)} possible states")

Example (combining multiple bases):
    >>> from tbp.hippocampus import Hippocampus, HippocampusConfig
    >>> from tbp.hippocampus.basis import MetricGridBasis, PlaceBasis, CombinedBasis
    >>>
    >>> # Create combined basis with grid and place cells
    >>> grid_basis = MetricGridBasis()
    >>> place_basis = PlaceBasis()
    >>> combined = CombinedBasis([grid_basis, place_basis])
    >>>
    >>> # Use in hippocampus configuration
    >>> config = HippocampusConfig()
    >>> hippocampus = Hippocampus(config)

Structure:
    tbp.hippocampus.types         - Core data types (SpatialEvent, PlaceCell, etc.)
    tbp.hippocampus.hstate        - HState latent representation
    tbp.hippocampus.basis         - Pluggable basis code system (EC generalization)
    tbp.hippocampus.entorhinal    - Entorhinal cortex implementation
    tbp.hippocampus.dentate_gyrus - Pattern separation (ultra-sparse coding)
    tbp.hippocampus.ca3           - Autoassociative memory + transition graph
    tbp.hippocampus.ca1           - Comparator network + cortical mapping
    tbp.hippocampus.memory        - Episodic memory buffer
    tbp.hippocampus.hippocampus   - High-level integrated API
    tbp.hippocampus.adapters      - Integration adapters (e.g., MontyAdapter)
"""

# Basis code system (new)
from tbp.hippocampus.basis import (
    BasisCode,
    BasisConfig,
    MetricGridBasis,
    PlaceBasis,
    PlaceBasisConfig,
    CombinedBasis,
    IdentityBasis,
    create_spatial_basis,
)
# CA1 (extended with cortical mapping)
from tbp.hippocampus.ca1 import CA1, CA1Config, ComparisonResult
# CA3 (extended with transition graph)
from tbp.hippocampus.ca3 import CA3, CA3Config, CA3Memory, TransitionEntry
# Dentate Gyrus
from tbp.hippocampus.dentate_gyrus import DentateGyrus, DGConfig
# Entorhinal Cortex
from tbp.hippocampus.entorhinal import EntorhinalCortex
# High-level API (new)
from tbp.hippocampus.hippocampus import Hippocampus, HippocampusConfig
# HState latent representation (new)
from tbp.hippocampus.hstate import HState
# Episodic memory
from tbp.hippocampus.memory import EpisodicMemory
# Core types (backwards compatible)
from tbp.hippocampus.types import SpatialEvent

__all__ = [
    # === Core types (backwards compatible) ===
    "SpatialEvent",

    # === HState latent representation (new) ===
    "HState",

    # === Basis code system (new) ===
    "BasisCode",
    "BasisConfig",
    "MetricGridBasis",
    "PlaceBasis",
    "PlaceBasisConfig",
    "CombinedBasis",
    "IdentityBasis",
    "create_spatial_basis",

    # === Entorhinal Cortex ===
    "EntorhinalCortex",

    # === Dentate Gyrus ===
    "DentateGyrus",
    "DGConfig",

    # === CA3 (extended) ===
    "CA3",
    "CA3Config",
    "CA3Memory",
    "TransitionEntry",

    # === CA1 (extended) ===
    "CA1",
    "CA1Config",
    "ComparisonResult",

    # === Memory ===
    "EpisodicMemory",

    # === High-level API (new) ===
    "Hippocampus",
    "HippocampusConfig",
]
