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
