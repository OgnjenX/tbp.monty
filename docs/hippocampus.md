---
title: Hippocampus
---

The hippocampus module (`tbp.hippocampus`) implements a biologically-inspired model of hippocampal computations. It is designed as an **independent package** with no dependencies on Monty's neocortical components, making it easy to use standalone or integrate with Monty via adapters.

## Overview

The hippocampus is critical for episodic memory, spatial navigation, and novelty detection. Our implementation models the canonical trisynaptic loop:

```
                    ┌─────────────────────────────────────────────────┐
                    │              HIPPOCAMPUS                        │
                    │                                                 │
    Monty LMs       │    ┌──────────┐     ┌───────┐     ┌─────┐      │
    (Neocortex)  ───┼───►│    EC    │────►│  DG   │────►│ CA3 │──┐   │
                    │    │ (grid +  │     │(0.1-1%│     │(auto│  │   │
                    │    │  place)  │     │sparse)│     │assoc│  │   │
                    │    └────┬─────┘     └───────┘     └─────┘  │   │
                    │         │                                   │   │
                    │         │     Direct path                   │   │
                    │         ▼                                   ▼   │
                    │    ┌─────────────────────────────────────┐     │
                    │    │                CA1                   │     │
                    │    │    (Comparator: prediction vs input) │     │
                    │    │    Match = familiar, Mismatch = novel│     │
                    │    └──────────────────┬──────────────────┘     │
                    │                       │                         │
                    │                       ▼                         │
                    │              ┌─────────────────┐                │
                    │              │ EpisodicMemory  │                │
                    │              │ (consolidated)  │                │
                    │              └─────────────────┘                │
                    └─────────────────────────────────────────────────┘
```

## Architecture

The hippocampus package is structured as follows:

| Module | Class | Description |
|--------|-------|-------------|
| `types` | `SpatialEvent` | Core data type representing a spatial observation (location, orientation, features) |
| `entorhinal` | `EntorhinalCortex` | Grid cells and place cells for spatial encoding |
| `dentate_gyrus` | `DentateGyrus` | Pattern separation via ultra-sparse encoding (0.1-1% active) |
| `ca3` | `CA3` | Autoassociative memory with pattern completion and replay |
| `ca1` | `CA1` | Comparator between CA3 predictions and EC input |
| `memory` | `EpisodicMemory` | Thread-safe circular buffer for event storage |
| `adapters` | `MontyAdapter` | Bridge between Monty LMs and hippocampus (only Monty-dependent code) |

## Components

### SpatialEvent

The fundamental data type representing a spatial observation:

```python
from tbp.hippocampus import SpatialEvent
import numpy as np

event = SpatialEvent(
    location=np.array([1.0, 2.0, 3.0]),      # 3D position
    orientation=np.eye(3),                    # 3x3 rotation matrix
    features={"object_id": "mug"},            # Arbitrary feature dict
    timestamp=0.0,                            # When observed
    source_id="sensor_0",                     # Which sensor/LM
    confidence=0.9,                           # Observation confidence
)
```

### Entorhinal Cortex (EC)

The gateway to the hippocampus. Provides grid cell and place cell representations of spatial locations.

```python
from tbp.hippocampus import EntorhinalCortex, SpatialEvent

ec = EntorhinalCortex(n_place_cells=100, n_grid_cells=50)

# Process an event
activations = ec.receive_event(event)

# Get representations
place_code = ec.get_place_representation(event.location)  # Gaussian bumps
grid_code = ec.get_grid_representation(event.location)    # Periodic pattern
```

**Key features:**
- Place cells: Gaussian tuning curves centered at learned locations
- Grid cells: Periodic spatial firing patterns (simplified hex grid model)
- Listener pattern: downstream modules can subscribe to new events
- Built-in episodic memory buffer for recent events

### Dentate Gyrus (DG)

Performs **pattern separation** by converting EC input into ultra-sparse codes. This is the "address generator" for CA3 memory storage.

```python
from tbp.hippocampus import DentateGyrus, DGConfig

config = DGConfig(
    n_input=100,              # EC input dimensionality
    n_granule_cells=1000,     # Number of granule cells (expansion)
    sparsity=0.005,           # Target sparsity (0.5% active)
    neurogenesis_rate=0.01,   # Fraction of "young" cells
)
dg = DentateGyrus(config)

# Encode EC input to sparse DG pattern
ec_input = np.random.randn(100)  # Continuous EC vector
output = dg.encode(ec_input)

print(output.sparse_code.shape)      # (1000,) binary vector
print(output.sparsity_achieved)      # ~0.005
print(output.novelty_score)          # 0.0-1.0
print(output.active_indices)         # Which cells fired
```

**Key features:**
- **Random expansion**: Each granule cell connects to ~20% of EC inputs (sparse random projection)
- **Winner-take-all**: Only top 0.1-1% of cells activate (ultra-sparse SDR)
- **Pattern separation**: Similar EC inputs → distant DG codes
- **Novelty detection**: Compares current input to recent patterns
- **Neurogenesis simulation**: Young cells are more excitable for novel inputs

**Why ultra-sparse?**
- DG is much sparser than neocortex (0.5% vs 2-5%)
- This ensures CA3 stores distinct, non-overlapping memory traces
- Small input changes can produce very different DG codes

### CA3 (Autoassociative Memory)

Stores episodic memories and performs **pattern completion** from partial cues.

```python
from tbp.hippocampus import CA3, CA3Config

config = CA3Config(
    n_pyramidal_cells=2500,       # CA3 population size
    n_active_cells=50,            # ~2% sparsity
    recurrent_connectivity=0.02,  # Sparse recurrent connections
    learning_rate=0.1,            # Hebbian learning rate
    pattern_completion_threshold=0.3,
    memory_capacity=500,
)
ca3 = CA3(config)

# Store a pattern from DG
ca3.store(dg_pattern, spatial_event)

# Complete from partial cue
completed_pattern, retrieved_event = ca3.pattern_complete(partial_cue)

# Replay memories (sharp-wave ripple simulation)
replayed = ca3.replay(n_patterns=5)

# Check novelty
novelty = ca3.compute_novelty(new_pattern)  # 1.0 = novel, 0.0 = familiar
```

**Key features:**
- **Hebbian learning**: Strengthens connections between co-active neurons
- **Attractor dynamics**: Iteratively settles to stored pattern from partial cue
- **Memory replay**: Reactivates stored patterns weighted by strength/recency
- **Consolidation**: Prunes weak memories when capacity exceeded
- **Novelty detection**: Poor pattern completion = high novelty

### CA1 (Comparator)

Compares CA3 predictions with direct EC input to detect **match vs mismatch**.

```python
from tbp.hippocampus import CA1, CA1Config

config = CA1Config(
    n_pyramidal_cells=2500,
    match_threshold=0.6,      # Threshold for "match"
    ec_weight=0.5,            # Balance between EC and CA3
    temporal_window=1.0,      # For sequence detection
)
ca1 = CA1(config)

# Compare CA3 prediction with EC reality
result = ca1.compare(ca3_pattern, ec_pattern, event)

print(result.match_score)      # 0.0-1.0
print(result.is_match)         # True/False
print(result.novelty_signal)   # 1 - match_score
print(result.sequence_position)  # Position in detected sequence
```

**Key features:**
- **Match/mismatch detection**: Core novelty signal
- **Weighted blending**: On mismatch, weights EC more (reality > prediction)
- **Sequence learning**: Tracks temporal sequences of patterns
- **Prediction**: Can predict next pattern in learned sequences
- **Mismatch-driven learning**: Updates weights when prediction fails

### Episodic Memory

Thread-safe circular buffer for storing spatial events.

```python
from tbp.hippocampus import EpisodicMemory

memory = EpisodicMemory(capacity=1000)

# Store events
memory.store(event)
memory.store_batch([event1, event2, event3])

# Retrieve
recent = memory.get_recent(n=10)
by_source = memory.get_by_source("sensor_0")
filtered = memory.query(lambda e: e.confidence > 0.8)
```

## Integration with Monty

The `MontyAdapter` bridges Monty's Learning Modules to the hippocampus:

```python
from tbp.hippocampus import EntorhinalCortex
from tbp.hippocampus.adapters import MontyAdapter

# Create hippocampus (independent of Monty)
ec = EntorhinalCortex(n_place_cells=100, n_grid_cells=50)

# Create adapter
adapter = MontyAdapter(ec)

# In experiment loop, after monty._vote():
adapter.process_step(monty)
```

The adapter:
1. Polls each LM's `send_out_vote()` and `get_output()`
2. Converts CMP `State` objects to `SpatialEvent`s
3. Feeds events into the hippocampal pipeline

**Key design principle**: Only `adapters/monty_adapter.py` imports from `tbp.monty`. All other hippocampus code is Monty-independent, making future extraction to a separate package trivial.

## Full Pipeline Example

```python
import numpy as np
from tbp.hippocampus import (
    EntorhinalCortex, DentateGyrus, CA3, CA1,
    SpatialEvent, DGConfig, CA3Config, CA1Config,
)

# Initialize components
ec = EntorhinalCortex(n_place_cells=100, n_grid_cells=50)
dg = DentateGyrus(DGConfig(n_input=100, n_granule_cells=1000))
ca3 = CA3(CA3Config(n_pyramidal_cells=1000))
ca1 = CA1(CA1Config(n_pyramidal_cells=1000))

# Create a spatial event
event = SpatialEvent(
    location=np.array([1.0, 2.0, 3.0]),
    orientation=np.eye(3),
    features={"object_id": "mug"},
    timestamp=0.0,
    source_id="sensor_0",
    confidence=0.9,
)

# Process through the trisynaptic loop
# 1. EC: get spatial representation
ec_activations = ec.receive_event(event)

# 2. Build EC input vector for DG
ec_input = np.concatenate([
    event.location,
    event.orientation.flatten(),
    [event.confidence]
])

# 3. DG: pattern separation
dg_output = dg.encode(ec_input)
print(f"DG sparsity: {dg_output.sparsity_achieved:.3f}")
print(f"Novelty: {dg_output.novelty_score:.2f}")

# 4. CA3: store memory
ca3.store(dg_output.sparse_code, event)

# 5. Later: pattern completion from partial cue
completed, retrieved = ca3.pattern_complete(dg_output.sparse_code)

# 6. CA1: compare prediction with reality
result = ca1.compare(completed, dg_output.sparse_code, event)
if result.is_match:
    print("Familiar location!")
else:
    print(f"Novel! Novelty signal: {result.novelty_signal:.2f}")
```

## Biological Alignment

| Region | Implementation | Biological Feature |
|--------|----------------|-------------------|
| EC | Grid + place cells | ✓ Spatial encoding via periodic and localized firing |
| DG | 0.1-1% sparsity, random expansion | ✓ Ultra-sparse, pattern separation |
| DG | Novelty-boosted young cells | ✓ Adult neurogenesis effect |
| CA3 | Hebbian autoassociator | ✓ Recurrent collaterals, attractor dynamics |
| CA3 | Memory replay | ✓ Sharp-wave ripple analogue |
| CA1 | CA3 vs EC comparison | ✓ Match/mismatch detection |
| CA1 | Sequence learning | ✓ Temporal coding |

**Simplifications:**
- Grid cells use simplified periodic patterns, not full hexagonal lattices
- Neurogenesis modulates excitability but doesn't add/remove cells
- CA3 connectivity mask not fully enforced in weight updates
- No subiculum or deep-layer feedback yet

## Running Tests

```bash
# All hippocampus tests (64 tests)
conda run -n tbp.monty python -m pytest tests/unit/hippocampus/ -v -o addopts=""

# Individual components
conda run -n tbp.monty python -m pytest tests/unit/hippocampus/test_core.py -v -o addopts=""
conda run -n tbp.monty python -m pytest tests/unit/hippocampus/test_dentate_gyrus.py -v -o addopts=""
conda run -n tbp.monty python -m pytest tests/unit/hippocampus/test_ca3.py -v -o addopts=""
conda run -n tbp.monty python -m pytest tests/unit/hippocampus/test_ca1.py -v -o addopts=""
conda run -n tbp.monty python -m pytest tests/unit/hippocampus/test_monty_adapter.py -v -o addopts=""
```

## Future Work

- [ ] Add EC → CA3 direct "content" path (DG provides address, EC provides content)
- [ ] Implement true hexagonal grid cell patterns
- [ ] Add actual neurogenesis (add/remove cells over time)
- [ ] Enforce sparse connectivity mask in CA3 weight updates
- [ ] Add subiculum output layer
- [ ] Implement oscillatory timing (theta/gamma rhythms)
- [ ] Add consolidation to neocortex (hippocampal-cortical dialogue)
