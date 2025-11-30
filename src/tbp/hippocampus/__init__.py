# Copyright 2025 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Hippocampus module - independent of Monty (neocortex).

This package implements hippocampal computations including:
- Entorhinal Cortex (EC): grid cells, place cells, spatial encoding
- Dentate Gyrus (DG): pattern separation with ultra-sparse encoding
- CA3: autoassociative memory with pattern completion
- CA1: comparator between CA3 predictions and EC input
- Episodic memory: event storage and retrieval

The core modules have NO dependencies on tbp.monty. Integration with Monty
is handled via adapters in tbp.hippocampus.adapters.

Structure:
    tbp.hippocampus.types        - Core data types (SpatialEvent, etc.)
    tbp.hippocampus.entorhinal   - Entorhinal cortex implementation
    tbp.hippocampus.dentate_gyrus - Pattern separation (ultra-sparse coding)
    tbp.hippocampus.ca3          - Autoassociative memory
    tbp.hippocampus.ca1          - Comparator network
    tbp.hippocampus.memory       - Episodic memory buffer
    tbp.hippocampus.adapters     - Integration adapters (e.g., MontyAdapter)
"""

from tbp.hippocampus.ca1 import CA1, CA1Config, ComparisonResult
from tbp.hippocampus.ca3 import CA3, CA3Config, CA3Memory
from tbp.hippocampus.dentate_gyrus import DentateGyrus, DGConfig
from tbp.hippocampus.entorhinal import EntorhinalCortex
from tbp.hippocampus.memory import EpisodicMemory
from tbp.hippocampus.types import SpatialEvent

__all__ = [
    # Core types
    "SpatialEvent",
    # Entorhinal Cortex
    "EntorhinalCortex",
    # Dentate Gyrus
    "DentateGyrus",
    "DGConfig",
    # CA3
    "CA3",
    "CA3Config",
    "CA3Memory",
    # CA1
    "CA1",
    "CA1Config",
    "ComparisonResult",
    # Memory
    "EpisodicMemory",
]
