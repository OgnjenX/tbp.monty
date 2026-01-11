# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np

from tbp.monty.frameworks.models.hippocampal_graph_lm import (
    ObjectObservation,
    ReplayBatch,
)
from tbp.monty.frameworks.models.transition_consolidation_lm import (
    CorticalTransitionLM,
    SequenceReplayMemory,
    TransitionConsolidationMemory,
)


class TestTransitionConsolidationMemory:
    def test_learns_adjacent_transitions(self):
        mem = TransitionConsolidationMemory()
        batch = ReplayBatch(
            observations=[
                ObjectObservation("A", location=np.zeros(3), timestamp=0.0),
                ObjectObservation("B", location=np.zeros(3), timestamp=1.0),
                ObjectObservation("C", location=np.zeros(3), timestamp=2.0),
            ],
            source_type="episode",
            batch_id="b0",
        )
        mem.observe_replay_batch(batch)

        assert np.isclose(mem.get_transition_weight("A", "B"), 1.0, rtol=1e-09, atol=1e-09)
        assert np.isclose(mem.get_transition_weight("B", "C"), 1.0, rtol=1e-09, atol=1e-09)
        assert np.isclose(mem.get_transition_weight("A", "C"), 0.0, rtol=1e-09, atol=1e-09)
        assert mem.predict_next("A") == "B"

    def test_dedupe_consecutive(self):
        mem = TransitionConsolidationMemory(dedupe_consecutive_object_ids=True)
        batch = ReplayBatch(
            observations=[
                ObjectObservation("A", location=np.zeros(3), timestamp=0.0),
                ObjectObservation("A", location=np.zeros(3), timestamp=0.5),
                ObjectObservation("B", location=np.zeros(3), timestamp=1.0),
            ],
            source_type="episode",
            batch_id="b1",
        )
        mem.observe_replay_batch(batch)

        assert np.isclose(mem.get_transition_weight("A", "B"), 1.0, rtol=1e-09, atol=1e-09)
        assert np.isclose(mem.total_transitions, 1.0, rtol=1e-09, atol=1e-09)

    def test_respects_timestamps_for_ordering(self):
        mem = TransitionConsolidationMemory()
        batch = ReplayBatch(
            observations=[
                ObjectObservation("B", location=np.zeros(3), timestamp=2.0),
                ObjectObservation("A", location=np.zeros(3), timestamp=1.0),
            ],
            source_type="episode",
            batch_id="b2",
        )
        mem.observe_replay_batch(batch)
        assert np.isclose(mem.get_transition_weight("A", "B"), 1.0, rtol=1e-09, atol=1e-09)


class TestSequenceReplayMemory:
    def test_stores_and_replays_episode(self):
        store = SequenceReplayMemory(max_episode_history=10)
        episode = ReplayBatch(
            observations=[
                ObjectObservation("A", location=np.zeros(3), timestamp=0.0),
                ObjectObservation("B", location=np.zeros(3), timestamp=1.0),
            ],
            source_type="sequence_episode",
            batch_id="seq0",
            context="ctx",
        )
        store.store_episode(episode)
        replays = store.generate_episode_replay(num_replays=2, context_filter="ctx")

        assert len(replays) == 2
        assert replays[0].observations[0].object_id == "A"
        assert replays[0].observations[1].object_id == "B"


class _LMOutput:
    def __init__(self, object_id: str, confidence: float):
        self.use_state = True
        self.non_morphological_features = {"object_id": object_id}
        self.confidence = confidence


class TestCorticalTransitionLM:
    def test_updates_transitions_from_online_steps(self):
        lm = CorticalTransitionLM(dedupe_consecutive_object_ids=True, learning_rate=0.5)
        lm.pre_episode()
        lm.matching_step([_LMOutput("A", 0.9)])
        lm.matching_step([_LMOutput("A", 0.9)])
        lm.matching_step([_LMOutput("B", 0.9)])
        lm.post_episode()

        # A->B should exist as a learned transition in the relational memory.
        rel = lm.memory.relations.get(("A", "B"))
        assert rel is not None
        assert rel.transition_weight > 0.0
