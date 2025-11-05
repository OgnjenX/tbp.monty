RFC: Creativity Meta-Layer (CML) for Perspective-Shifting in Monty (TBT)

Status: Draft
Author: TBD

Motivation
- Enable transformational creativity by allowing controlled perspective shifts:
  dynamic remapping of inter-LM reference frames, temporary feature reweighting, and
  optional topology modulation under conflict/novelty signals.

Background
- Columns are EvidenceGraphLMs; voting combines hypotheses via inter-sensor transforms.
- CMP State passes messages; motor/GSG drive exploration.

Design Overview
- Add a Creativity Meta-Layer (CML) that observes per-step metrics and, when
  appropriate, installs alternative transform profiles (schemas), reweights features,
  or modulates voting topology. Non-breaking, opt-in via new hooks.

API Changes (Minimal Hooks)
- MontyBase: optional callbacks `before_vote_cb(sensor_outputs, lm_outputs)` and
  `after_step_cb(step_metrics)` (no-op by default).
- MontyForGraphMatching & Evidence subclass: optional `transform_manager` consulted in
  `_combine_votes` for pairwise transform override and postprocessing.
- EvidenceGraphLM: helper methods `get_entropy()`, `get_mlh_slope(window)` and
  temporary feature weight override/restore.

Control Policy (External to core)
- Trigger perspective shift when entropy high, conflicts high, and MLH progress low.
- Actions include alternative transform families, feature reweighting, topology tweaks,
  and shadow-track evaluation before consolidation.

Metrics
- Coherence: evidence concentration (1 - normalized entropy), transform residuals.
- Novelty: SDR/feature overlap vs. memory, bin coverage in graph_memory.
- Efficiency: time-to-coherence, compute budget used.

Validation Plan
- Compare baseline vs. CML on symmetric/ambiguous objects and distribution shifts.
- Ablations for shadow tracks, reweighting, topology. Use existing loggers to track
  new CML metrics.

Rollout
- PR1 (this RFC): hook interfaces and identity defaults.
- PR2: minimal CML and TransformManager with alt-transform blending.
- PR3: shadow tracks + schema cache and persistence.
- PR4: topology modulation, extended configs, docs.

Backward Compatibility
- All additions are optional and default to identity behavior; no change in results
  unless hooks are populated.

