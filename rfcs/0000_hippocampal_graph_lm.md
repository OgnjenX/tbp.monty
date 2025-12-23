- Start Date: 2025-12-23
- RFC PR: (leave empty)

# RFC 0000: HippocampalGraphLM — Fast Episodic Relational Learning

## Summary
Introduce a hippocampus-inspired, fast-learning graph learning module (HippocampalGraphLM) that captures object-to-object relations within and across episodes. The module stores co-occurrence, spatial displacement, temporal offset, and context for pairs of objects, enabling rapid associative recall, next-object prediction, and replay for consolidation. This RFC proposes how it fits alongside existing Monty LMs, what behaviors it should support, and how it maps (and intentionally diverges) from biological hippocampus function.

## Motivation
- Findings: Existing Monty LMs model intra-object structure but lack explicit object–object relational memory needed for episodic recall and compositional scenes.
- Gap: No fast, one-shot relational binding; no way to query "what tends to appear with X (and where/when)?"
- Goal: Provide an explicit relational graph that learns in a few shots and can replay or bias neocortical LMs, improving scene understanding and planning.

## Goals
- One/few-shot learning of relations between observed objects within an episode.
- Store spatial displacement, temporal ordering, co-occurrence counts, and optional context labels per pair.
- Provide APIs for associative recall (related objects + stats), next-object prediction, and replay of episodes or relation-centric batches.
- Integrate cleanly with Monty episode loops and GraphLM-based pipelines.
- Support consolidation via replay to downstream neocortical LMs.

## Non-Goals
- Full biological fidelity of DG→CA3→CA1 circuitry or SDR representations.
- Long-term consolidation pipeline fully implemented; replay API provided but integration into neocortical training is user-defined.
- Probabilistic generative modeling of full scenes; focus is fast relational memory.

## Proposed Design
### Components
- `HippocampalGraphLM`: orchestrates episode buffering, consolidation, querying, replay hooks.
- `HippocampalGraphMemory`: persistent relational graph storing pairwise stats (co-occurrence counts, spatial displacement list, temporal offsets, contexts, timestamps).
- `ObjectObservation`: typed observation (id, location, pose optional, timestamp, confidence).
- `ObjectRelation`: stats aggregator for a directed pair (A→B) with symmetric insertion for convenience.
- `ReplayBatch`: a sequence of observations for replay to downstream LMs (episode-level or relation-centric).

### Learning Flow
1. During an episode: `observe_object(object_id, location, timestamp, context?)` buffers observations.
2. Consolidation: `end_episode()` forms all unordered pairs within the episode, updates relation stats with learning rate (fast default 0.8–1.0), increments episode count.
3. Auto-consolidation: optional `max_observations_per_episode` triggers `end_episode()` automatically.
4. Context: optional string tag per episode stored on each relation update.
5. Episode storage: if `enable_replay=True`, episodes are stored (with pruning to `max_episode_history`).

### Querying & Inference
- `get_related_objects(object_id)` → mapping of neighbors to relation stats.
- `get_most_common_neighbors(object_id, top_k)` → ranked co-occurrence counts.
- `predict_next_object(object_id)` → argmax neighbor by co-occurrence (optionally weighted by recency/temporal offset in future work).
- Relation stats expose average spatial displacement and average temporal offset for placement/ordering hints.

### Replay (core feature)
Two replay strategies for consolidation and reinforcement:

#### Episode Replay
- Sample past episodes from `episode_history` and emit their full observation sequences.
- Use case: strengthen representations of frequent scene configurations.
- Control: `replay_episodes(num_replays, context_filter)` with optional context filtering.

#### Relation-Centric Replay
- Sample high-frequency object pairs and create mini-episodes from learned spatial/temporal statistics.
- Observations are synthetic: pair (A, B) with A at origin, B at learned average displacement and temporal offset.
- Use case: reinforce associative bindings without full episode replay; efficient for high-value pairs.
- Control: `replay_relations(num_replays, top_k_relations, temperature, context_filter)`.
  - `temperature`: 0 = greedy (always sample highest frequency), higher = more uniform sampling.
  - `top_k_relations`: how many relations to consider (top by frequency).

#### Callback Integration
- `register_replay_callback(callback)` — register handlers to receive ReplayBatch objects.
- Downstream LMs (e.g., neocortical GraphLMs) subscribe and update on replay data, achieving consolidation.

### Forgetting Strategy
**Key design decision: explicit decay and history pruning (not pure one-shot)**

Why not pure one-shot (infinite memory)?
1. **Biological precedent**: hippocampus does consolidate to neocortex and gradually forgets "unique" episodes while preserving "gist."
2. **Computational efficiency**: storing infinite episodes and relation samples is memory-inefficient; capacity limits prevent pathological overfitting to outliers.
3. **Noise resilience**: relations from very old or noisy episodes may mislead predictions; decay prioritizes recent and frequent patterns.

Implementation:
- `decay_rate` parameter (0–1): exponential decay for relation weight based on `last_update_time`.
  - 0 = no decay (infinite memory).
  - 0.01–0.1 = gradual decay over hundreds/thousands of episodes.
  - Decay is *optionally applied* during prediction or replay; today stored as a parameter but not actively computed (future work).
- `max_episode_history` parameter: prune oldest episodes when history exceeds this size (default 100).
  - This is *active pruning*: oldest episodes are discarded to bound memory.
  - Relation stats are retained (aggregated from all episodes).

Rationale:
- Storing full episodes allows replay; pruning prevents unbounded memory growth.
- Relation aggregates (counts, averages) persist even after episode deletion.
- Decay rate can later be applied to co-occurrence counts as a multiplicative weight in future work.

### Integration Points in Monty
- Hook inside episode loop after each object hypothesis: call `observe_object(...)` with current object id and pose/location.
- After episode termination: call `end_episode()`; optionally trigger replay batches to neocortical LMs.
- Downstream LMs subscribe via `register_replay_callback(...)` to receive and train on replayed data.
- Storage: module lives alongside other LMs; no changes required to existing object models.

### Biological Inspiration vs. Implementation
- Parallels: fast binding of episodes; associative recall; replay for consolidation and system memory transfer.
- Differences: no DG→CA3→CA1 separation, no SDRs; explicit graph edges instead of distributed codes; replay is schematic (average spatial/temporal) rather than ripple-accurate; forgetting via active history pruning rather than sparsity-driven decay in CA1.

## Data Model
- Nodes: object ids (string labels consistent with Monty object hypotheses).
- Edges: directed records with stats: co_occurrence_count (int), spatial_displacements (list[np.ndarray]), temporal_offsets (list[float]), contexts (set[str]), last_seen timestamp.
- Symmetry: insert both directions for convenience; aggregation remains per direction.

## Configuration
- `learning_rate` (float, default 0.8–1.0): blend factor for displacement/temporal running means.
- `max_observations_per_episode` (int|None): auto-consolidation threshold.
- `decay_rate` (float, 0–1): exponential decay strength (future: active application).
- `enable_replay` (bool, default True): whether to store episode history.
- `max_episode_history` (int, default 100): maximum episodes before pruning.
- `temperature` (float, default 1.0 in relation replay): sampling temperature for relation selection.

## Testing Strategy
- Unit coverage for observation creation, relation accumulation, stats aggregation, context handling, prediction, and reset/clear paths.
- Replay tests for episode and relation-centric generation, callback invocation, history pruning, context filtering, and temperature control.
- Integration scenarios for multi-object scenes, compositional objects, sequence prediction, and replay with multiple callbacks.

## Migration / Backwards Compatibility
- Additive module: no breaking changes to existing LMs or configs. Optional adoption in pipelines.

## Open Questions
- **Decay active application**: when and how to apply `decay_rate` to co-occurrence counts? On prediction? During replay sampling? Per-episode?
- **Replay scheduling**: in a live training loop, how often to trigger relation vs. episode replay? Mix ratio policy?
- **Temporal modeling**: should prediction weight temporal offsets vs. raw co-occurrence? Explore future.
- **Context gating**: how strong should context filtering be for retrieval and replay in multi-scene learning?
- **Capacity vs. forgetting trade-off**: is `max_episode_history` the right knob, or should we use relation-level decay instead?

## Appendix (Visualization Narrative)
A scene becomes a graph: nodes are objects, edges store how often they co-occur, where B sits relative to A, and the typical order (Δt). **Replay** samples paths like `plate → fork → knife` and feeds them to slower LMs, nudging them toward stable scene priors. **Forgetting** via history pruning ensures the hippocampus doesn't become a perfect archive; old, low-frequency relations fade, while gist (high-count averages) remains. Together, this mirrors biological consolidation: hippocampus learns fast, replays during sleep, and lets cortex absorb the summary.
