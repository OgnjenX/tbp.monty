# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from collections import deque
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)
from tbp.monty.frameworks.models.abstract_monty_classes import LearningModule
from tbp.monty.frameworks.models.entorhinal_interface import (
    EntorhinalLocationIntegrator,
    EntorhinalLocationIntegratorConfig,
    observation_coordinate_frame,
)
from tbp.monty.frameworks.models.hippocampal_graph_lm import (
    HippocampalGraphLM,
    ObjectObservation,
    ReplayBatch,
)
from tbp.monty.frameworks.models.transition_consolidation_lm import (
    SequenceReplayMemory,
    TransitionConsolidationMemory,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HippocampalGraphSidecarConfig:
    enabled: bool = True
    learning_rate: float = 0.8
    decay_rate: float = 0.0
    max_observations_per_episode: int = 10_000
    enable_replay: bool = True
    max_episode_history: int = 200
    context_from_primary_target: bool = False
    dedupe_consecutive_object_ids: bool = True
    output_confidence_threshold: float = 0.0
    # Completion suggestions behavior (pattern completion vs temporal "next")
    completion_use_temporal: bool = False
    # Optional replay-to-consolidation target (simple cortical transition memory)
    enable_consolidation: bool = False
    consolidation_dedupe_consecutive_object_ids: bool = True
    replay_episode_batches_per_episode: int = 0
    replay_relation_batches_per_episode: int = 0
    replay_sequence_batches_per_episode: int = 0
    replay_top_k_relations: int = 10
    replay_temperature: float = 1.0
    # Optional belief encoding (top-k hypotheses from votes, gated commit)
    enable_belief_encoding: bool = False
    belief_window_size: int = 20
    belief_top_k: int = 5
    belief_p_min: float = 0.15
    belief_entropy_max: float = 1.2
    belief_novelty_min: float = 0.15
    belief_min_objects: int = 2
    enable_entorhinal_location_integration: bool = True
    entorhinal_sensor_key_hint: Optional[str] = None
    write_csv: bool = True


class MontyObjectRecognitionHippocampalGraphSidecarExperiment(
    MontyObjectRecognitionExperiment
):
    """Object recognition experiment with HippocampalGraphLM as a sidecar.

    This does not register HippocampalGraphLM as a Monty learning module.
    Instead, it listens to learning module outputs each step and streams
    object-level events into HippocampalGraphLM via `observe_object`.

    This is the lowest-friction way to evaluate hippocampal relational memory
    on existing Monty experiments (especially multi-object world-image scenes).
    """

    def __init__(self, config):
        super().__init__(config=config)
        self._hipp_cfg = HippocampalGraphSidecarConfig(
            **dict(config.get("hippocampal_graph", {}))
        )
        self._hipp: Optional[HippocampalGraphLM] = None
        self._consolidation: Optional[TransitionConsolidationMemory] = None
        self._sequence_replay: Optional[SequenceReplayMemory] = None
        self._sequence_episode: list[ObjectObservation] = []
        self._hipp_csv_path: Optional[Path] = None
        self._hipp_csv_file = None
        self._hipp_csv_writer: Optional[csv.DictWriter] = None
        self._last_object_id: Optional[str] = None
        self._belief_buffer: deque[Tuple[float, list[Tuple[str, float, np.ndarray]]]] = deque()
        self._prev_dist: Optional[Dict[str, float]] = None
        self._episode_observation_count: int = 0
        self._last_committed_cue: Optional[str] = None
        self._semantic_id_to_label: Optional[Dict[Any, str]] = None
        self._last_env_observation: Any = None
        self._last_agent_state: Any = None
        self._last_observation_frame: Optional[str] = None
        self._entorhinal: Optional[EntorhinalLocationIntegrator] = None

    def setup_experiment(self, config: Dict[str, Any]) -> None:
        super().setup_experiment(config)
        if not self._hipp_cfg.enabled:
            return

        self._hipp = HippocampalGraphLM(
            learning_rate=self._hipp_cfg.learning_rate,
            decay_rate=self._hipp_cfg.decay_rate,
            max_observations_per_episode=self._hipp_cfg.max_observations_per_episode,
            enable_replay=self._hipp_cfg.enable_replay,
            max_episode_history=self._hipp_cfg.max_episode_history,
        )

        self._entorhinal = EntorhinalLocationIntegrator(
            EntorhinalLocationIntegratorConfig(
                enabled=self._hipp_cfg.enable_entorhinal_location_integration,
                sensor_key_hint=self._hipp_cfg.entorhinal_sensor_key_hint,
            )
        )

        if self._hipp_cfg.enable_consolidation:
            self._consolidation = TransitionConsolidationMemory(
                dedupe_consecutive_object_ids=self._hipp_cfg.consolidation_dedupe_consecutive_object_ids
            )
            self._hipp.register_replay_callback(self._on_hippocampal_replay)
            self._sequence_replay = SequenceReplayMemory(
                max_episode_history=self._hipp_cfg.max_episode_history
            )
            for lm in getattr(self.model, "learning_modules", []):
                if hasattr(lm, "on_replay_batch"):
                    self._hipp.register_replay_callback(lm.on_replay_batch)

        if self._hipp_cfg.write_csv:
            self._hipp_csv_path = self.output_dir / "hippocampal_graph_sidecar.csv"
            self._hipp_csv_file = self._hipp_csv_path.open("w", newline="")
            self._hipp_csv_writer = csv.DictWriter(
                self._hipp_csv_file,
                fieldnames=[
                    "phase",
                    "episode_index",
                    "primary_target",
                    "context",
                    "n_observations_in_episode",
                    "episode_count",
                    "stored_episodes",
                    "n_objects",
                    "n_relations",
                    "total_observations",
                    "completion_cue",
                    "completion_suggestions",
                    "consolidation_n_objects",
                    "consolidation_n_transitions",
                    "consolidation_total_transitions",
                    "sequence_n_observations",
                ],
            )
            self._hipp_csv_writer.writeheader()

    def close(self):
        try:
            if self._hipp_csv_file is not None:
                self._hipp_csv_file.close()
        finally:
            super().close()

    def pre_episode(self):
        super().pre_episode()
        if self._hipp is None:
            return

        self._hipp.reset()
        self._last_object_id = None
        self._belief_buffer.clear()
        self._prev_dist = None
        self._episode_observation_count = 0
        self._last_committed_cue = None
        self._semantic_id_to_label = getattr(self.env_interface, "semantic_id_to_label", None)
        self._sequence_episode = []

        context = None
        if self._hipp_cfg.context_from_primary_target:
            context = getattr(self.env_interface, "primary_target", {}).get("object")
        if context:
            self._hipp.set_episode_context(str(context))

    def post_episode(self, steps):
        if self._hipp is not None:
            context = self._hipp.current_episode_context

            if self._hipp_cfg.enable_belief_encoding:
                self._flush_belief_buffer(force=True)
                self._hipp.current_episode_observations = []
                self._hipp.current_episode_context = None
            else:
                self._hipp.end_episode()

            stats = self._hipp.get_statistics()
            n_obs = self._episode_observation_count
            cue = self._last_committed_cue

            suggestions = self._get_completion_suggestions(cue, context)

            self._run_replay_to_consolidation(context=context)
            consolidation_stats = self._consolidation.get_statistics() if self._consolidation is not None else {}

            if self._hipp_csv_writer is not None:
                self._write_hipp_csv(context, stats, n_obs, cue, suggestions, consolidation_stats)

        super().post_episode(steps)

    def _get_completion_suggestions(self, cue: Optional[str], context: Optional[str]) -> list[Tuple[str, float]]:
        """Return completion suggestions for a cue or an empty list."""
        if cue is None or self._hipp is None:
            return []
        if self._hipp_cfg.completion_use_temporal:
            return self._hipp.graph_memory.get_temporal_next_candidates(
                current_object=cue,
                context=context,
                top_k=5,
            )
        return self._hipp.complete_from_cue([cue], context_filter=context, top_k_objects=5)

    def _write_hipp_csv(
        self,
        context: Optional[str],
        stats: Dict[str, Any],
        n_obs: int,
        cue: Optional[str],
        suggestions: list[Tuple[str, float]],
        consolidation_stats: Dict[str, Any],
    ) -> None:
        """Write a single CSV row for hippocampal sidecar stats."""
        if self._hipp_csv_writer is None:
            return

        phase = "train" if self.experiment_mode.value == "train" else "eval"
        episode_index = self.train_episodes if phase == "train" else self.eval_episodes
        primary_target = getattr(self.env_interface, "primary_target", {}).get("object")
        writer = self._hipp_csv_writer
        writer.writerow(
            {
                "phase": phase,
                "episode_index": episode_index,
                "primary_target": primary_target,
                "context": context,
                "n_observations_in_episode": n_obs,
                "episode_count": stats.get("episode_count"),
                "stored_episodes": stats.get("stored_episodes"),
                "n_objects": stats.get("n_objects"),
                "n_relations": stats.get("n_relations"),
                "total_observations": stats.get("total_observations"),
                "completion_cue": cue,
                "completion_suggestions": ";".join(
                    f"{obj}:{score:.3f}" for obj, score in suggestions
                ),
                "consolidation_n_objects": consolidation_stats.get("n_objects"),
                "consolidation_n_transitions": consolidation_stats.get("n_transitions"),
                "consolidation_total_transitions": consolidation_stats.get("total_transitions"),
                "sequence_n_observations": len(self._sequence_episode),
            }
        )

    def _on_hippocampal_replay(self, batch) -> None:
        if self._consolidation is None:
            return
        self._consolidation.observe_replay_batch(batch)

    def _run_replay_to_consolidation(self, context: Optional[str]) -> None:
        if self._hipp is None or self._consolidation is None:
            return

        n_ep = int(self._hipp_cfg.replay_episode_batches_per_episode)
        n_rel = int(self._hipp_cfg.replay_relation_batches_per_episode)
        n_seq = int(self._hipp_cfg.replay_sequence_batches_per_episode)
        if n_ep <= 0 and n_rel <= 0 and n_seq <= 0:
            return

        if n_ep > 0:
            self._hipp.replay_episodes(
                num_replays=n_ep,
                context_filter=context,
                invoke_callbacks=True,
            )

        if n_rel > 0:
            self._hipp.replay_relations(
                num_replays=n_rel,
                top_k_relations=int(self._hipp_cfg.replay_top_k_relations),
                temperature=float(self._hipp_cfg.replay_temperature),
                context_filter=context,
                invoke_callbacks=True,
            )

        if n_seq > 0 and self._sequence_replay is not None:
            if len(self._sequence_episode) >= 2:
                self._sequence_replay.store_episode(
                    ReplayBatch(
                        observations=self._sequence_episode.copy(),
                        source_type="sequence_episode",
                        batch_id=f"sequence_episode_{self._hipp.graph_memory.episode_count}",
                        context=context,
                    )
                )
            for batch in self._sequence_replay.generate_episode_replay(
                num_replays=n_seq,
                context_filter=context,
            ):
                self._consolidation.observe_replay_batch(batch)

    def run_episode_steps(self):
        env_interface = self.env_interface
        if env_interface is None:
            raise RuntimeError("env_interface is not set; did you call pre_epoch()?")

        loader_step = -1
        for loader_step, observation in enumerate(env_interface):
            self._last_env_observation = observation
            if self._should_terminate_episode(loader_step):
                return loader_step

            self._step_model(observation)
            self._step_hippocampus(loader_step)

            if self.model.is_done:
                return loader_step

        self.model.set_is_done()
        return loader_step

    def _should_terminate_episode(self, step: int) -> bool:
        if self.model.check_reached_max_matching_steps(self.max_steps):
            logger.info(f"Terminated due to maximum matching steps : {self.max_steps}")
            return True

        if step >= self.max_total_steps:
            logger.info(f"Terminated due to maximum episode steps : {step}")
            self.model.deal_with_time_out()
            return True

        return False

    def _step_model(self, observation: Any) -> None:
        if self.model.is_motor_only_step:
            self.model.pass_features_directly_to_motor_system(observation)
        else:
            self.model.step(observation)

    def _step_hippocampus(self, step: int) -> None:
        if self._hipp is None:
            return
        self._last_agent_state = getattr(self.model, "get_agent_state", lambda: None)()
        self._last_observation_frame = observation_coordinate_frame(self._last_env_observation)
        if self._hipp_cfg.enable_belief_encoding:
            self._observe_beliefs_from_lms(step_timestamp=float(step))
        else:
            self._observe_from_lms(step_timestamp=float(step))

    def _observe_from_lms(self, step_timestamp: float) -> None:
        if self._hipp is None:
            return

        for lm in getattr(self.model, "learning_modules", []):
            observation_data = self._extract_observation_data(lm)
            if observation_data is None:
                continue

            self._hipp.observe_object(
                object_id=observation_data["object_id"],
                location=observation_data["location"],
                timestamp=step_timestamp,
                confidence=observation_data["confidence"],
                source_lm=observation_data["source_lm"],
            )
            self._append_to_sequence_episode(
                object_id=observation_data["object_id"],
                location=observation_data["location"],
                timestamp=step_timestamp,
                confidence=observation_data["confidence"],
                source_lm=observation_data["source_lm"],
            )
            self._last_object_id = observation_data["object_id"]
            self._episode_observation_count += 1
            self._last_committed_cue = observation_data["object_id"]

    def _extract_observation_data(self, lm) -> Optional[Dict[str, Any]]:
        """Extract and validate observation data from a learning module.

        Returns a dict with observation data if valid, None otherwise.
        """
        try:
            output = lm.get_output()
        except Exception:
            return None

        if output is None or not getattr(output, "use_state", False):
            return None

        non_morph_features = getattr(output, "non_morphological_features", {}) or {}
        object_id = non_morph_features.get("object_id")
        if not object_id:
            return None

        confidence = float(getattr(output, "confidence", 0.0))
        if confidence < self._hipp_cfg.output_confidence_threshold:
            return None

        object_id = str(object_id)
        if (
            self._hipp_cfg.dedupe_consecutive_object_ids
            and object_id == self._last_object_id
        ):
            return None

        location_raw = getattr(output, "location", None)
        if location_raw is None:
            return None
        location = np.asarray(location_raw, dtype=float)
        if np.size(location) == 0:
            return None
        location = self._maybe_integrate_location(location)

        return {
            "object_id": object_id,
            "location": location,
            "confidence": confidence,
            "source_lm": str(getattr(lm, "learning_module_id", "lm")),
        }

    def _maybe_integrate_location(self, location: np.ndarray) -> np.ndarray:
        """Optionally convert sensor-frame locations into a shared world frame.

        This is a no-op when observations declare world coordinates (typical when
        `world_coord=True` in the env observation processor). For safety, this also
        no-ops when the reference frame is unknown/ambiguous.
        """
        if self._last_observation_frame != "sensor":
            return location
        if not self._hipp_cfg.enable_entorhinal_location_integration:
            return location
        if self._entorhinal is None or self._last_agent_state is None:
            return location
        return self._entorhinal.sensor_to_world(
            location_sensor=location,
            agent_state=self._last_agent_state,
        )

    def _observe_beliefs_from_lms(self, step_timestamp: float) -> None:
        """Collect top-k hypotheses from votes and commit weighted co-activations."""
        candidates = self._collect_vote_candidates(top_k=self._hipp_cfg.belief_top_k)
        if not candidates:
            return

        dist = {obj_id: prob for obj_id, prob, _ in candidates}
        entropy = self._entropy([prob for _, prob, _ in candidates])
        max_p = max(prob for _, prob, _ in candidates)
        stable = (max_p >= self._hipp_cfg.belief_p_min) or (entropy <= self._hipp_cfg.belief_entropy_max)

        novelty = 1.0
        if self._prev_dist is not None:
            novelty = 1.0 - self._cosine_similarity_dict(dist, self._prev_dist)
        self._prev_dist = dist

        if stable:
            self._belief_buffer.append((step_timestamp, candidates))
            maxlen = max(1, int(self._hipp_cfg.belief_window_size))
            while len(self._belief_buffer) > maxlen:
                self._belief_buffer.popleft()
            # Separate from co-occurrence encoding: commit a single best-guess
            # observation per step for true temporal transition learning.
            best_obj_id, best_prob, best_location = max(
                candidates, key=lambda t: t[1]
            )
            self._append_to_sequence_episode(
                object_id=best_obj_id,
                location=best_location,
                timestamp=step_timestamp,
                confidence=best_prob,
                source_lm="votes",
            )

        if not stable or novelty < self._hipp_cfg.belief_novelty_min:
            return

        if self._count_distinct_objects_in_buffer() < max(2, int(self._hipp_cfg.belief_min_objects)):
            return

        self._flush_belief_buffer(force=False)

    def _flush_belief_buffer(self, force: bool) -> None:
        if self._hipp is None or not self._belief_buffer:
            return

        context = self._hipp.current_episode_context

        evidence: Dict[str, float] = {}
        loc_sums: Dict[str, np.ndarray] = {}
        time_sums: Dict[str, float] = {}

        for t, candidates in self._belief_buffer:
            for obj_id, prob, location in candidates:
                evidence[obj_id] = evidence.get(obj_id, 0.0) + prob
                loc_sums[obj_id] = loc_sums.get(obj_id, np.zeros(3, dtype=float)) + (prob * location)
                time_sums[obj_id] = time_sums.get(obj_id, 0.0) + (prob * t)

        objects = [o for o, e in evidence.items() if e > 0.0]
        if len(objects) < max(2, int(self._hipp_cfg.belief_min_objects)):
            if force:
                self._belief_buffer.clear()
            return

        gate_strength = self._compute_gate_strength()

        episode_obs = self._create_episode_observations(
            objects, evidence, loc_sums, time_sums
        )

        if not episode_obs:
            self._belief_buffer.clear()
            return

        self._hipp.graph_memory.store_episode(episode_obs, context=context)
        self._hipp.graph_memory.episode_count += 1
        while len(self._hipp.graph_memory.episode_history) > self._hipp_cfg.max_episode_history:
            self._hipp.graph_memory.episode_history.pop(0)
        self._episode_observation_count += len(episode_obs)
        self._last_committed_cue = episode_obs[0].object_id

        self._update_weighted_coactivations(
            objects, evidence, loc_sums, time_sums, gate_strength, context
        )

        self._hipp.graph_memory.total_observations += len(episode_obs)
        self._hipp.graph_memory.current_time += 1.0
        self._belief_buffer.clear()

    def _collect_vote_candidates(self, top_k: int) -> list[Tuple[str, float, np.ndarray]]:
        """Collect and normalize top-k vote candidates across all LMs."""
        if self._hipp is None:
            return []

        raw = self._extract_raw_vote_candidates()
        if not raw:
            return []

        return self._normalize_vote_candidates(raw, top_k)

    def _append_to_sequence_episode(
        self,
        object_id: str,
        location: np.ndarray,
        timestamp: float,
        confidence: float,
        source_lm: str,
    ) -> None:
        if self._consolidation is None:
            return

        object_id = str(object_id)
        if (
            self._hipp_cfg.consolidation_dedupe_consecutive_object_ids
            and self._sequence_episode
            and self._sequence_episode[-1].object_id == object_id
        ):
            return

        self._sequence_episode.append(
            ObjectObservation(
                object_id=object_id,
                location=np.asarray(location, dtype=float),
                timestamp=float(timestamp),
                confidence=float(confidence),
                source_lm=str(source_lm),
            )
        )

    def _compute_gate_strength(self) -> float:
        """Compute gate strength from last belief buffer entry."""
        if not self._belief_buffer:
            return 0.0
        last_candidates = self._belief_buffer[-1][1]
        last_max_p = max(prob for _, prob, _ in last_candidates)
        return float(np.clip(last_max_p, 0.0, 1.0))

    def _create_episode_observations(
        self,
        objects: list[str],
        evidence: Dict[str, float],
        loc_sums: Dict[str, np.ndarray],
        time_sums: Dict[str, float],
    ) -> list[ObjectObservation]:
        """Create condensed episode observations from belief buffer."""
        episode_obs: list[ObjectObservation] = []
        for obj_id in sorted(objects, key=lambda o: evidence[o], reverse=True)[:10]:
            e = evidence[obj_id]
            loc = loc_sums[obj_id] / max(1e-9, e)
            ts = time_sums[obj_id] / max(1e-9, e)
            episode_obs.append(
                ObjectObservation(
                    object_id=obj_id,
                    location=loc,
                    timestamp=float(ts),
                    confidence=float(np.clip(e, 0.0, 1.0)),
                    source_lm="belief_buffer",
                )
            )
        return episode_obs

    def _update_weighted_coactivations(
        self,
        objects: list[str],
        evidence: Dict[str, float],
        loc_sums: Dict[str, np.ndarray],
        time_sums: Dict[str, float],
        gate_strength: float,
        context: Optional[str],
    ) -> None:
        """Update weighted co-activation relations between objects."""
        if self._hipp is None:
            return
        for i, obj_a in enumerate(objects):
            for obj_b in objects[i + 1:]:
                e_a = evidence[obj_a]
                e_b = evidence[obj_b]
                if e_a <= 0.0 or e_b <= 0.0:
                    continue
                loc_a = loc_sums[obj_a] / max(1e-9, e_a)
                loc_b = loc_sums[obj_b] / max(1e-9, e_b)
                t_a = time_sums[obj_a] / max(1e-9, e_a)
                t_b = time_sums[obj_b] / max(1e-9, e_b)
                weight = gate_strength * e_a * e_b
                self._hipp.graph_memory.add_relation(
                    object_a=obj_a,
                    object_b=obj_b,
                    spatial_displacement=loc_b - loc_a,
                    temporal_offset=float(t_b - t_a),
                    context=context,
                    timestamp=self._hipp.graph_memory.current_time,
                    weight=weight,
                )

    def _extract_raw_vote_candidates(self) -> list[Tuple[str, float, np.ndarray]]:
        """Extract raw vote candidates from all learning modules."""
        raw: list[Tuple[str, float, np.ndarray]] = []
        for lm in getattr(self.model, "learning_modules", []):
            raw.extend(self._process_lm_vote(lm))
        return raw

    def _process_lm_vote(self, lm: LearningModule) -> list[Tuple[str, float, np.ndarray]]:
        """Process a single learning module's vote.

        A learning module may emit multiple possible states (e.g., multiple graph ids);
        each state contributes a raw candidate that can affect normalization and gating.
        """
        try:
            vote = lm.send_out_vote()
        except Exception:
            return []
        
        if not isinstance(vote, dict):
            return []
        
        possible_states: Dict[str, Any] = vote.get("possible_states", {})
        if not isinstance(possible_states, dict):
            return []

        candidates: list[Tuple[str, float, np.ndarray]] = []
        for graph_id, states in possible_states.items():
            candidate = self._extract_best_state(graph_id, states)
            if candidate:
                candidates.append(candidate)
        return candidates

    def _extract_best_state(self, graph_id: Any, states) -> Optional[Tuple[str, float, np.ndarray]]:
        """Extract best state from possible states."""
        if not isinstance(states, Sequence) or not states:
            return None
        
        best_state = max(states, key=lambda s: float(getattr(s, "confidence", 0.0)))
        conf = float(getattr(best_state, "confidence", 0.0))
        loc = np.asarray(getattr(best_state, "location", None), dtype=float)

        if np.size(loc) == 0:
            return None
        loc = self._maybe_integrate_location(loc)

        obj_id = self._normalize_object_id(graph_id)
        return (obj_id, conf, loc)

    def _normalize_vote_candidates(
        self, raw: list[Tuple[str, float, np.ndarray]], top_k: int
    ) -> list[Tuple[str, float, np.ndarray]]:
        """Normalize and return top-k vote candidates."""
        raw.sort(key=lambda t: t[1], reverse=True)
        candidates = raw[: max(1, int(top_k))]
        total = sum(max(0.0, c) for _, c, _ in candidates)
        
        if total <= 0:
            prob = 1.0 / len(candidates)
            return [(obj_id, prob, loc) for obj_id, _, loc in candidates]
        
        return [(obj_id, conf / total, loc) for obj_id, conf, loc in candidates]

    def _count_distinct_objects_in_buffer(self) -> int:
        seen: set[str] = set()
        for _, candidates in self._belief_buffer:
            for obj_id, prob, _ in candidates:
                if prob > 0:
                    seen.add(obj_id)
        return len(seen)

    def _normalize_object_id(self, object_id: Any) -> str:
        """Map semantic ids to labels when possible, otherwise stringify."""
        mapping = self._semantic_id_to_label
        if not mapping:
            return str(object_id)
        try:
            if isinstance(object_id, str) and object_id.isdigit():
                object_id_int = int(object_id)
                return str(mapping.get(object_id_int, object_id))
            return str(mapping.get(object_id, object_id))
        except Exception:
            return str(object_id)

    @staticmethod
    def _entropy(probs: list[float]) -> float:
        p = np.asarray(probs, dtype=float)
        p = np.asarray(p[p > 0], dtype=float)
        if p.size < 1:
            return 0.0
        return float(np.sum(-p * np.log(p)))

    @staticmethod
    def _cosine_similarity_dict(a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = sorted(set(a.keys()) | set(b.keys()))
        va = np.array([a.get(k, 0.0) for k in keys], dtype=float)
        vb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom <= 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)
