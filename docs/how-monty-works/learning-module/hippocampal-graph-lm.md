# HippocampalGraphLM

Fast episodic relational learning for Monty: a graph memory that records object-to-object relations (co-occurrence, spatial displacement, temporal offset, and context) in one/few shots, then supports associative recall, next-object prediction, and replay for consolidation.

## Why
- Existing LMs model intra-object structure; scenes need inter-object bindings for episodic recall and compositional reasoning.
- HippocampalGraphLM provides rapid binding so downstream (slower) models can be guided or replay-trained.
- Replay enables consolidation: hippocampal patterns strengthen neocortical models without requiring neocortex to re-learn from scratch.

## Mental model
- **Nodes**: object ids seen in an episode.
- **Edges**: directed relation stats `(A → B)` with:
  - `co_occurrence_count`
  - `spatial_displacements`: list of vectors `B - A`
  - `temporal_offsets`: list of `t(B) - t(A)`
  - `contexts`: set of episode labels
- **Learning**: every episode forms all unordered pairs; stats aggregate with a fast learning rate.

## Data model and aggregation
For a displacement stream `d_i` the running mean uses learning rate `α`:
$$\mu_{t} = (1-\alpha)\,\mu_{t-1} + \alpha\,d_t$$
(implemented via explicit list + numpy mean today; hook for EMA later).

Stored per relation:
- `co_occurrence_count`: integer
- `spatial_displacements`: list[np.ndarray]
- `temporal_offsets`: list[float]
- `contexts`: set[str]
- `last_seen`: float | None (timestamp)

## Core API (current)
- `observe_object(object_id, location, timestamp, pose=None, confidence=1.0)` — buffer an observation for the active episode.
- `set_episode_context(context)` — tag all subsequent observations until `end_episode`.
- `end_episode()` — consolidate buffered observations into relations; clears episode buffer.
- `get_relation(a, b)` — fetch relation stats (directed).
- `get_related_objects(a)` — mapping of neighbors to relations.
- `get_most_common_neighbors(a, top_k=5)` — ranked co-occurrence counts.
- `predict_next_object(a)` — argmax neighbor by co-occurrence (future: weight by temporal offset/recency).
- `clear_memory()` / `reset()` — wipe long-term graph or just the current episode buffer.

## Episode flow
1. Start episode: optionally `set_episode_context("kitchen")`.
2. For each detected object hypothesis: `observe_object(obj_id, location, timestamp, pose?)`.
3. On termination or auto-threshold: `end_episode()` builds all pairs from the buffered observations and updates stats.

## Replay (planned extension)
- **Episode replay**: sample past episodes and emit their observation sequences to downstream LMs.
- **Relation replay**: sample high-value pairs and emit synthetic mini-episodes containing those pairs.
- **Scheduling knobs**: mix ratio (live vs. replay), sampling temperature, context filter.

## Integration into Monty
- Wire calls to `observe_object` inside the episode loop right after object inference/pose estimation.
- Call `end_episode` when the environment signals episode end (or let `max_observations_per_episode` auto-consolidate).
- Optional: feed replay batches to neocortical LMs during off-policy training.

## Biological parallels and differences
- Parallels: fast binding of episodes; associative recall; replay for consolidation.
- Differences: no DG→CA3→CA1 separation, no SDRs; explicit graph edges instead of distributed codes; replay is schematic rather than ripple-accurate.

## Practical tips
- Use contexts to separate scenes (e.g., `kitchen`, `office`).
- For compositional parts (logo on object), treat parts as objects; spatial displacement encodes attachment.
- For sequence biasing, `predict_next_object` provides a quick heuristic; extend with temporal weighting if needed.

## References in repo
- Implementation: `src/tbp/monty/frameworks/models/hippocampal_graph_lm.py`
- Tests: `tests/unit/test_hippocampal_graph_lm.py`
- Demo: `examples/hippocampal_graph_lm_demo.py`
