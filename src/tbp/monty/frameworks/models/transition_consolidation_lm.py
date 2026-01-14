# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tbp.monty.frameworks.models.hippocampal_graph_lm import ReplayBatch
from tbp.monty.frameworks.models.abstract_monty_classes import GoalState, LearningModule
from tbp.monty.frameworks.models.hippocampal_graph_lm import ObjectObservation

_rng = np.random.default_rng(seed=42)


@dataclass
class TransitionConsolidationMemory:
    """A minimal transition memory over object ids.

    This is intended as a simple "cortical consolidation target" for hippocampal
    replay batches: it learns directed transitions between object identities from
    the temporal order of observations in a replay batch.
    """

    dedupe_consecutive_object_ids: bool = True
    transitions: Dict[Tuple[str, str], float] = field(default_factory=dict)
    outgoing_totals: Dict[str, float] = field(default_factory=dict)
    objects_seen: set[str] = field(default_factory=set)
    total_transitions: float = 0.0

    def clear(self) -> None:
        self.transitions.clear()
        self.outgoing_totals.clear()
        self.objects_seen.clear()
        self.total_transitions = 0.0

    def observe_replay_batch(self, batch: ReplayBatch, weight: float = 1.0) -> None:
        if not batch.observations:
            return

        weight = float(weight)
        if weight <= 0.0:
            return

        observations = sorted(batch.observations, key=lambda o: float(o.timestamp))
        object_ids = [str(obs.object_id) for obs in observations if obs.object_id]
        if self.dedupe_consecutive_object_ids:
            object_ids = self._dedupe_consecutive(object_ids)

        if len(object_ids) < 2:
            return

        self.objects_seen.update(object_ids)
        for a, b in zip(object_ids, object_ids[1:]):
            if not a or not b or a == b:
                continue
            self.transitions[(a, b)] = self.transitions.get((a, b), 0.0) + weight
            self.outgoing_totals[a] = self.outgoing_totals.get(a, 0.0) + weight
            self.total_transitions += weight

    def get_transition_weight(self, a: str, b: str) -> float:
        return float(self.transitions.get((a, b), 0.0))

    def predict_next(self, a: str) -> Optional[str]:
        a = str(a)
        candidates = [(b, w) for (src, b), w in self.transitions.items() if src == a]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t[1])[0]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "n_objects": len(self.objects_seen),
            "n_transitions": len(self.transitions),
            "total_transitions": float(self.total_transitions),
        }

    @staticmethod
    def _dedupe_consecutive(object_ids: list[str]) -> list[str]:
        if not object_ids:
            return []
        deduped = [object_ids[0]]
        for obj_id in object_ids[1:]:
            if obj_id != deduped[-1]:
                deduped.append(obj_id)
        return deduped


@dataclass
class CorticalRelation:
    """A compact relation summary updated via EMA (cortex-like consolidation)."""

    object_a: str
    object_b: str
    co_occurrence_weight: float = 0.0
    transition_weight: float = 0.0
    mean_spatial_displacement: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    mean_temporal_offset: float = 0.0
    forward_fraction: float = 0.0

    def update(
        self,
        spatial_displacement: Optional[np.ndarray],
        temporal_offset: Optional[float],
        is_transition: bool,
        lr: float,
        weight: float = 1.0,
    ) -> None:
        lr = float(np.clip(lr, 0.0, 1.0))
        weight = float(weight)
        self.co_occurrence_weight += weight
        if is_transition:
            self.transition_weight += weight

        if spatial_displacement is not None:
            disp = np.asarray(spatial_displacement, dtype=float)
            if disp.shape == (3,):
                self.mean_spatial_displacement = (1.0 - lr) * self.mean_spatial_displacement + lr * disp

        if temporal_offset is not None:
            t = float(temporal_offset)
            self.mean_temporal_offset = (1.0 - lr) * self.mean_temporal_offset + lr * t
            self.forward_fraction = (1.0 - lr) * self.forward_fraction + lr * float(t > 0.0)


@dataclass
class CorticalRelationalMemory:
    """Relational graph over object ids with EMA updates (no full episode storage)."""

    learning_rate: float = 0.05
    relations: Dict[Tuple[str, str], CorticalRelation] = field(default_factory=dict)
    objects_seen: set[str] = field(default_factory=set)

    def clear(self) -> None:
        self.relations.clear()
        self.objects_seen.clear()

    def add_relation(
        self,
        object_a: str,
        object_b: str,
        spatial_displacement: Optional[np.ndarray] = None,
        temporal_offset: Optional[float] = None,
        is_transition: bool = False,
        weight: float = 1.0,
    ) -> None:
        object_a = str(object_a)
        object_b = str(object_b)
        self.objects_seen.add(object_a)
        self.objects_seen.add(object_b)

        for src, dst, disp, t_off in [
            (object_a, object_b, spatial_displacement, temporal_offset),
            (object_b, object_a, -spatial_displacement if spatial_displacement is not None else None, -temporal_offset if temporal_offset is not None else None),
        ]:
            key = (src, dst)
            if key not in self.relations:
                self.relations[key] = CorticalRelation(object_a=src, object_b=dst)
            self.relations[key].update(
                spatial_displacement=disp,
                temporal_offset=t_off,
                is_transition=is_transition,
                lr=self.learning_rate,
                weight=weight,
            )

    def get_statistics(self) -> Dict[str, Any]:
        n_relations = len(self.relations) // 2
        n_transitions = sum(1 for rel in self.relations.values() if rel.transition_weight > 0) // 2
        return {
            "n_objects": len(self.objects_seen),
            "n_relations": n_relations,
            "n_transitions": n_transitions,
        }

    def predict_next(self, object_id: str) -> Optional[str]:
        object_id = str(object_id)
        candidates = [
            (dst, rel)
            for (src, dst), rel in self.relations.items()
            if src == object_id and rel.transition_weight > 0
        ]
        if not candidates:
            return None
        # Prefer strong transitions that are usually forward.
        best = max(candidates, key=lambda t: (t[1].transition_weight, t[1].forward_fraction))
        return best[0]


@dataclass
class SequenceReplayMemory:
    """Stores ordered object-id episodes for replay to a consolidation target."""

    max_episode_history: int = 200
    episode_history: List[ReplayBatch] = field(default_factory=list)

    def clear(self) -> None:
        self.episode_history.clear()

    def store_episode(self, episode: ReplayBatch) -> None:
        if not episode.observations:
            return
        self.episode_history.append(episode)
        while len(self.episode_history) > int(self.max_episode_history):
            self.episode_history.pop(0)

    def generate_episode_replay(
        self,
        num_replays: int = 1,
        context_filter: Optional[str] = None,
    ) -> List[ReplayBatch]:
        if not self.episode_history or num_replays <= 0:
            return []

        candidates = self.episode_history
        if context_filter:
            candidates = [
                ep for ep in self.episode_history if context_filter in (ep.context or "")
            ]
        if not candidates:
            return []

        replays: List[ReplayBatch] = []
        for i in range(int(num_replays)):
            ep = candidates[_rng.integers(0, len(candidates))]
            replays.append(
                ReplayBatch(
                    observations=ep.observations.copy(),
                    source_type="sequence_episode",
                    batch_id=f"{ep.batch_id}_replay_{i}",
                    context=ep.context,
                )
            )
        return replays


class _OutputStub:
    use_state = False


class _BufferStub:
    def __init__(self):
        self._last_obs_processed = False
        self._num_observations_on_object = 0

    def reset(self) -> None:
        self._last_obs_processed = False
        self._num_observations_on_object = 0

    def set_last_obs_processed(self, processed: bool) -> None:
        self._last_obs_processed = bool(processed)

    def get_last_obs_processed(self) -> bool:
        return bool(self._last_obs_processed)

    def increment_observations_on_object(self) -> None:
        self._num_observations_on_object += 1

    def get_num_observations_on_object(self) -> int:
        return int(self._num_observations_on_object)

    def update_last_stats_entry(self, stats: Dict[str, Any]) -> None:
        return None


class CorticalTransitionLM(LearningModule):
    """A minimal LearningModule that learns object-id transitions.

    This is a "cortical" consolidation target: it can be stepped online from
    object-recognition LM outputs and also trained offline via hippocampal replay.
    """

    def __init__(
        self,
        dedupe_consecutive_object_ids: bool = True,
        learning_rate: float = 0.05,
        max_observations_per_episode: int = 10_000,
    ):
        # "Cortical" long-term store: EMA updates rather than full episodic storage.
        self.memory = CorticalRelationalMemory(learning_rate=float(learning_rate))
        self._dedupe_consecutive_object_ids = bool(dedupe_consecutive_object_ids)
        self._prev_object_id: Optional[str] = None
        self._step_index: float = 0.0
        self._max_observations_per_episode = int(max_observations_per_episode)
        self._episode_observations: list[ObjectObservation] = []
        self._output = _OutputStub()
        self.learning_module_id = "cortical_transition_lm"
        self.rng = None
        self._experiment_mode: Optional[str] = None
        # Compatibility with MontyForGraphMatching logging hooks.
        self.buffer = _BufferStub()
        self.stepwise_targets_list: list[Optional[str]] = []
        self.stepwise_target_object: Optional[str] = None
        self.terminal_state: Optional[str] = None

    def add_lm_processing_to_buffer_stats(self, lm_processed: bool) -> None:
        # MontyForGraphMatching expects LMs to log whether they were stepped.
        # This LM doesn't use the standard Buffer, so we keep this as a no-op.
        return None

    def get_possible_matches(self):
        return []

    def collect_stats_to_save(self) -> Dict[str, Any]:
        return {}

    def set_individual_ts(self, terminal_state: Optional[str] = None):
        self.terminal_state = terminal_state

    def update_terminal_condition(self):
        # This LM is not used for termination decisions in graph matching.
        # Keep it in a neutral state so it doesn't block episodes.
        self.set_individual_ts(None)
        return self.terminal_state

    def on_replay_batch(self, batch: ReplayBatch) -> None:
        self._update_from_observations(batch.observations, allow_transitions=True)

    def observe_object_id(self, object_id: str, location: Optional[np.ndarray], timestamp: float) -> None:
        object_id = str(object_id)
        if (
            self._dedupe_consecutive_object_ids
            and self._prev_object_id is not None
            and object_id == self._prev_object_id
        ):
            return
        loc = np.asarray(location, dtype=float) if location is not None else np.zeros(3, dtype=float)
        obs = ObjectObservation(
            object_id=object_id,
            location=loc,
            timestamp=float(timestamp),
            confidence=1.0,
            source_lm=self.learning_module_id,
        )
        self._episode_observations.append(obs)
        if len(self._episode_observations) >= self._max_observations_per_episode:
            self._consolidate_episode()
        self._prev_object_id = object_id

    # --- LearningModule API ---
    def reset(self):
        self._prev_object_id = None
        self._step_index = 0.0
        self.buffer.reset()
        self._episode_observations = []

    def pre_episode(self, *args, **kwargs):
        # Other LMs accept a primary_target argument; keep compatibility.
        self.reset()

    def post_episode(self):
        self._consolidate_episode()
        return None

    def set_experiment_mode(self, mode):
        self._experiment_mode = mode

    def matching_step(self, sensory_inputs=None) -> None:
        self._step_index += 1.0
        self._output = _OutputStub()
        self.buffer.set_last_obs_processed(False)

        if not sensory_inputs:
            return

        best_obj = None
        best_conf = -1.0
        best_loc = None
        processed = False
        for inp in sensory_inputs:
            if inp is None or not getattr(inp, "use_state", False):
                continue
            processed = True
            nmf = getattr(inp, "non_morphological_features", {}) or {}
            obj_id = nmf.get("object_id")
            if not obj_id:
                continue
            conf = float(getattr(inp, "confidence", 0.0))
            loc = getattr(inp, "location", None)
            if conf > best_conf:
                best_conf = conf
                best_obj = str(obj_id)
                best_loc = loc
                self._output = inp

        self.buffer.set_last_obs_processed(processed)
        if best_obj is not None:
            self.observe_object_id(
                best_obj,
                location=best_loc,
                timestamp=self._step_index,
            )
            self.buffer.increment_observations_on_object()

    def exploratory_step(self, sensory_inputs=None) -> None:
        return self.matching_step(sensory_inputs)

    def receive_votes(self, votes):
        return None

    def send_out_vote(self):
        return None

    def propose_goal_states(self) -> list[GoalState]:
        return []

    def get_output(self):
        return self._output

    def state_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": float(self.memory.learning_rate),
            "objects_seen": sorted(self.memory.objects_seen),
            "relations": {
                f"{a}||{b}": {
                    "co_occurrence_weight": rel.co_occurrence_weight,
                    "transition_weight": rel.transition_weight,
                    "mean_spatial_displacement": rel.mean_spatial_displacement.tolist(),
                    "mean_temporal_offset": rel.mean_temporal_offset,
                    "forward_fraction": rel.forward_fraction,
                }
                for (a, b), rel in self.memory.relations.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.memory.learning_rate = float(state_dict.get("learning_rate", self.memory.learning_rate))
        self.memory.objects_seen = set(state_dict.get("objects_seen", []))
        self.memory.relations = {}
        for key, payload in (state_dict.get("relations", {}) or {}).items():
            try:
                a, b = key.split("||", 1)
            except ValueError:
                continue
            rel = CorticalRelation(object_a=a, object_b=b)
            rel.co_occurrence_weight = float(payload.get("co_occurrence_weight", 0.0))
            rel.transition_weight = float(payload.get("transition_weight", 0.0))
            rel.mean_spatial_displacement = np.asarray(payload.get("mean_spatial_displacement", [0.0, 0.0, 0.0]), dtype=float)
            rel.mean_temporal_offset = float(payload.get("mean_temporal_offset", 0.0))
            rel.forward_fraction = float(payload.get("forward_fraction", 0.0))
            self.memory.relations[(a, b)] = rel

    def _consolidate_episode(self) -> None:
        if len(self._episode_observations) < 2:
            self._episode_observations = []
            return
        self._update_from_observations(self._episode_observations, allow_transitions=True)
        self._episode_observations = []

    def _update_from_observations(
        self,
        observations: List[ObjectObservation],
        allow_transitions: bool,
    ) -> None:
        if not observations:
            return
        obs_sorted = sorted(observations, key=lambda o: float(o.timestamp))
        # Pairwise relations (co-occurrence + displacement + temporal offset)
        for i, obs_a in enumerate(obs_sorted):
            for obs_b in obs_sorted[i + 1:]:
                if obs_a.object_id == obs_b.object_id:
                    continue
                disp = obs_b.location - obs_a.location
                t_off = obs_b.timestamp - obs_a.timestamp
                self.memory.add_relation(
                    obs_a.object_id,
                    obs_b.object_id,
                    spatial_displacement=disp,
                    temporal_offset=t_off,
                    is_transition=False,
                    weight=1.0,
                )
        # Adjacent transitions
        if allow_transitions:
            for prev, nxt in zip(obs_sorted, obs_sorted[1:]):
                if prev.object_id == nxt.object_id:
                    continue
                disp = nxt.location - prev.location
                t_off = nxt.timestamp - prev.timestamp
                self.memory.add_relation(
                    prev.object_id,
                    nxt.object_id,
                    spatial_displacement=disp,
                    temporal_offset=t_off,
                    is_transition=True,
                    weight=1.0,
                )
