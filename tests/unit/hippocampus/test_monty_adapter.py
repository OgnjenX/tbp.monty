"""Unit tests for MontyAdapter (depends on tbp.monty)."""

from __future__ import annotations

import numpy as np

from tbp.hippocampus.adapters.monty_adapter import MontyAdapter
from tbp.hippocampus.entorhinal import EntorhinalCortex
from tbp.monty.frameworks.models.states import State


class MockLearningModule:
    """Mock LM that returns controllable vote and output data."""

    def __init__(
        self,
        lm_id: str = "mock_lm_0",
        vote_data: dict | None = None,
        output_data: State | None = None,
    ):
        self.learning_module_id = lm_id
        self._vote_data = vote_data
        self._output_data = output_data

    def send_out_vote(self) -> dict | None:
        return self._vote_data

    def get_output(self) -> State | None:
        return self._output_data


def make_sample_vote(graph_id: str = "obj_cup", n_hypotheses: int = 2) -> dict:
    """Create a sample vote dictionary with n hypotheses."""
    states = []
    for i in range(n_hypotheses):
        state = State(
            location=np.array([0.1 * i, 0.2 * i, 0.3 * i]),
            morphological_features={
                "pose_vectors": np.eye(3) * (i + 1),
                "pose_fully_defined": True,
            },
            non_morphological_features=None,
            confidence=0.8 + 0.1 * i,
            use_state=True,
            sender_id="mock_lm_0",
            sender_type="LM",
        )
        states.append(state)

    return {
        "possible_states": {graph_id: states},
        "sensed_pose_rel_body": np.array([[0, 0, 0.5], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    }


def make_sample_output(use_state: bool = True) -> State:
    """Create a sample LM output State."""
    return State(
        location=np.array([0.5, 0.5, 0.5]),
        morphological_features={
            "pose_vectors": np.eye(3),
            "pose_fully_defined": True,
            "on_object": True,
        },
        non_morphological_features={
            "object_id": 12345,
        },
        confidence=0.95,
        use_state=use_state,
        sender_id="mock_lm_0",
        sender_type="LM",
    )


# ==================== MontyAdapter Tests ====================


class TestMontyAdapter:
    def test_consume_vote_events(self):
        hippocampus = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        adapter = MontyAdapter(hippocampus, include_outputs=False)

        lm = MockLearningModule(
            lm_id="test_lm",
            vote_data=make_sample_vote("obj_cup", n_hypotheses=3),
            output_data=None,
        )

        n_events = adapter.consume_from_lms([lm])

        assert n_events == 3
        assert len(hippocampus.memory) == 3

        events = hippocampus.memory.get_all()
        for event in events:
            assert event.event_type == "vote"
            assert event.source_id == "test_lm"
            assert event.object_id == "obj_cup"

    def test_consume_output_events(self):
        hippocampus = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        adapter = MontyAdapter(hippocampus, include_votes=False)

        lm = MockLearningModule(
            lm_id="test_lm",
            vote_data=None,
            output_data=make_sample_output(use_state=True),
        )

        n_events = adapter.consume_from_lms([lm])

        assert n_events == 1
        events = hippocampus.memory.get_all()
        assert len(events) == 1

        event = events[0]
        assert event.event_type == "output"
        assert event.object_id == 12345
        assert event.confidence == 0.95
        np.testing.assert_array_equal(event.location, [0.5, 0.5, 0.5])

    def test_output_ignored_when_use_state_false(self):
        hippocampus = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        adapter = MontyAdapter(hippocampus, include_votes=False)

        lm = MockLearningModule(
            lm_id="test_lm",
            vote_data=None,
            output_data=make_sample_output(use_state=False),
        )

        n_events = adapter.consume_from_lms([lm])

        assert n_events == 0
        assert len(hippocampus.memory) == 0

    def test_consume_both_votes_and_outputs(self):
        hippocampus = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        adapter = MontyAdapter(hippocampus)

        lm = MockLearningModule(
            lm_id="test_lm",
            vote_data=make_sample_vote("obj_mug", n_hypotheses=2),
            output_data=make_sample_output(use_state=True),
        )

        n_events = adapter.consume_from_lms([lm])

        assert n_events == 3  # 2 votes + 1 output
        events = hippocampus.memory.get_all()
        vote_events = [e for e in events if e.event_type == "vote"]
        output_events = [e for e in events if e.event_type == "output"]
        assert len(vote_events) == 2
        assert len(output_events) == 1

    def test_multiple_lms(self):
        hippocampus = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        adapter = MontyAdapter(hippocampus, include_outputs=False)

        lms = [
            MockLearningModule(
                lm_id=f"lm_{i}",
                vote_data=make_sample_vote(f"obj_{i}", n_hypotheses=1),
                output_data=None,
            )
            for i in range(3)
        ]

        n_events = adapter.consume_from_lms(lms)

        assert n_events == 3
        events = hippocampus.memory.get_all()
        source_ids = {e.source_id for e in events}
        assert source_ids == {"lm_0", "lm_1", "lm_2"}

    def test_step_counter(self):
        hippocampus = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        adapter = MontyAdapter(hippocampus)
        lm = MockLearningModule(lm_id="test_lm")

        assert adapter.step_count == 0
        adapter.consume_from_lms([lm])
        assert adapter.step_count == 1
        adapter.consume_from_lms([lm])
        assert adapter.step_count == 2

        adapter.reset()
        assert adapter.step_count == 0

    def test_none_vote_handled_gracefully(self):
        hippocampus = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        adapter = MontyAdapter(hippocampus)

        lm = MockLearningModule(
            lm_id="test_lm",
            vote_data=None,
            output_data=None,
        )

        n_events = adapter.consume_from_lms([lm])
        assert n_events == 0

    def test_events_have_correct_orientation(self):
        hippocampus = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        adapter = MontyAdapter(hippocampus, include_outputs=False)

        lm = MockLearningModule(
            lm_id="test_lm",
            vote_data=make_sample_vote("obj", n_hypotheses=1),
        )

        adapter.consume_from_lms([lm])

        events = hippocampus.memory.get_all()
        assert len(events) == 1
        # Orientation should be 3x3
        assert events[0].orientation.shape == (3, 3)
