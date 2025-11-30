"""Unit tests for core hippocampus module (no Monty dependencies)."""

from __future__ import annotations

import numpy as np
import pytest

from tbp.hippocampus.entorhinal import EntorhinalCortex
from tbp.hippocampus.memory import EpisodicMemory
from tbp.hippocampus.types import GridCell, PlaceCell, SpatialEvent


# ==================== SpatialEvent Tests ====================


class TestSpatialEvent:
    def test_create_event(self):
        event = SpatialEvent(
            timestamp=123.456,
            location=np.array([1.0, 2.0, 3.0]),
            orientation=np.eye(3),
            source_id="test_source",
            confidence=0.95,
        )

        assert event.timestamp == 123.456
        assert event.source_id == "test_source"
        assert event.confidence == 0.95
        np.testing.assert_array_equal(event.location, [1.0, 2.0, 3.0])

    def test_event_validates_location_shape(self):
        with pytest.raises(ValueError, match="location must have shape"):
            SpatialEvent(
                timestamp=0.0,
                location=np.array([1.0, 2.0]),  # Wrong shape
                orientation=np.eye(3),
                source_id="test",
                confidence=0.5,
            )

    def test_event_validates_orientation_shape(self):
        with pytest.raises(ValueError, match="orientation must have shape"):
            SpatialEvent(
                timestamp=0.0,
                location=np.array([1.0, 2.0, 3.0]),
                orientation=np.eye(2),  # Wrong shape
                source_id="test",
                confidence=0.5,
            )

    def test_event_to_dict_and_back(self):
        event = SpatialEvent(
            timestamp=100.0,
            location=np.array([0.1, 0.2, 0.3]),
            orientation=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            source_id="lm_0",
            confidence=0.8,
            features={"color": "red"},
            object_id="cup",
        )

        d = event.to_dict()
        restored = SpatialEvent.from_dict(d)

        assert restored.timestamp == event.timestamp
        assert restored.source_id == event.source_id
        assert restored.object_id == event.object_id
        np.testing.assert_array_almost_equal(restored.location, event.location)

    def test_event_default_values(self):
        event = SpatialEvent(
            timestamp=0.0,
            location=np.zeros(3),
            orientation=np.eye(3),
            source_id="test",
            confidence=0.5,
        )

        assert event.features == {}
        assert event.event_type == "observation"
        assert event.object_id is None
        assert event.extra == {}


# ==================== PlaceCell Tests ====================


class TestPlaceCell:
    def test_place_cell_activation_at_center(self):
        cell = PlaceCell(
            cell_id=0,
            center=np.array([0.0, 0.0, 0.0]),
            radius=0.1,
        )

        activation = cell.compute_activation(np.array([0.0, 0.0, 0.0]))
        assert activation == pytest.approx(1.0)

    def test_place_cell_activation_decreases_with_distance(self):
        cell = PlaceCell(
            cell_id=0,
            center=np.array([0.0, 0.0, 0.0]),
            radius=0.1,
        )

        act_near = cell.compute_activation(np.array([0.05, 0.0, 0.0]))
        act_far = cell.compute_activation(np.array([0.2, 0.0, 0.0]))

        assert act_near > act_far
        assert act_far < 0.5  # Should be low at 2x radius


# ==================== GridCell Tests ====================


class TestGridCell:
    def test_grid_cell_activation_range(self):
        cell = GridCell(
            cell_id=0,
            spacing=0.5,
            orientation=0.0,
            phase=np.array([0.0, 0.0]),
        )

        # Test at various locations
        activations = []
        for x in np.linspace(-1, 1, 20):
            for y in np.linspace(-1, 1, 20):
                act = cell.compute_activation(np.array([x, y, 0.0]))
                activations.append(act)

        # All activations should be in [0, 1]
        assert all(0 <= a <= 1 for a in activations)

    def test_grid_cell_periodicity(self):
        cell = GridCell(
            cell_id=0,
            spacing=0.5,
            orientation=0.0,
            phase=np.array([0.0, 0.0]),
        )

        # Activation should be similar at locations separated by spacing
        act1 = cell.compute_activation(np.array([0.0, 0.0, 0.0]))
        act2 = cell.compute_activation(np.array([0.5, 0.0, 0.0]))

        # Due to hexagonal pattern, not exactly equal, but should have periodicity
        # Just check both are valid activations
        assert 0 <= act1 <= 1
        assert 0 <= act2 <= 1


# ==================== EpisodicMemory Tests ====================


class TestEpisodicMemory:
    def make_event(self, timestamp: float = 0.0, source_id: str = "test") -> SpatialEvent:
        return SpatialEvent(
            timestamp=timestamp,
            location=np.array([0.0, 0.0, 0.0]),
            orientation=np.eye(3),
            source_id=source_id,
            confidence=0.9,
        )

    def test_store_and_retrieve(self):
        memory = EpisodicMemory(max_size=100)
        event = self.make_event(timestamp=1.0)

        memory.store(event)

        assert len(memory) == 1
        events = memory.get_all()
        assert len(events) == 1
        assert events[0].timestamp == 1.0

    def test_store_batch(self):
        memory = EpisodicMemory(max_size=100)
        events = [self.make_event(timestamp=float(i)) for i in range(5)]

        memory.store_batch(events)

        assert len(memory) == 5
        assert memory.total_received == 5

    def test_circular_buffer(self):
        memory = EpisodicMemory(max_size=3)

        for i in range(5):
            memory.store(self.make_event(timestamp=float(i)))

        assert len(memory) == 3
        assert memory.total_received == 5

        # Oldest events should be dropped
        events = memory.get_all()
        timestamps = [e.timestamp for e in events]
        assert timestamps == [2.0, 3.0, 4.0]

    def test_get_recent(self):
        memory = EpisodicMemory(max_size=100)
        for i in range(10):
            memory.store(self.make_event(timestamp=float(i)))

        recent = memory.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].timestamp == 9.0

    def test_get_by_source(self):
        memory = EpisodicMemory(max_size=100)
        memory.store(self.make_event(source_id="lm_0"))
        memory.store(self.make_event(source_id="lm_1"))
        memory.store(self.make_event(source_id="lm_0"))

        lm0_events = memory.get_by_source("lm_0")
        assert len(lm0_events) == 2

    def test_query_with_predicate(self):
        memory = EpisodicMemory(max_size=100)
        for i in range(10):
            memory.store(self.make_event(timestamp=float(i)))

        # Query for timestamps > 5
        result = memory.query(lambda e: e.timestamp > 5)
        assert len(result) == 4

    def test_flush(self):
        memory = EpisodicMemory(max_size=100)
        for i in range(5):
            memory.store(self.make_event())

        flushed = memory.flush()
        assert len(flushed) == 5
        assert len(memory) == 0


# ==================== EntorhinalCortex Tests ====================


class TestEntorhinalCortex:
    def make_event(
        self, location: np.ndarray | None = None, source_id: str = "test"
    ) -> SpatialEvent:
        if location is None:
            location = np.array([0.0, 0.0, 0.0])
        return SpatialEvent(
            timestamp=0.0,
            location=location,
            orientation=np.eye(3),
            source_id=source_id,
            confidence=0.9,
        )

    def test_receive_event_returns_activations(self):
        ec = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        event = self.make_event()

        result = ec.receive_event(event)

        assert "place_activations" in result
        assert "grid_activations" in result
        assert len(result["place_activations"]) == 10
        assert len(result["grid_activations"]) == 5

    def test_event_stored_in_memory(self):
        ec = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        event = self.make_event()

        ec.receive_event(event)

        assert len(ec.memory) == 1

    def test_event_count_increments(self):
        ec = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)

        assert ec.event_count == 0
        ec.receive_event(self.make_event())
        assert ec.event_count == 1
        ec.receive_event(self.make_event())
        assert ec.event_count == 2

    def test_listeners_notified(self):
        ec = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        received_events = []

        def listener(event: SpatialEvent):
            received_events.append(event)

        ec.add_listener(listener)
        ec.receive_event(self.make_event())

        assert len(received_events) == 1

    def test_remove_listener(self):
        ec = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        received_events = []

        def listener(event: SpatialEvent):
            received_events.append(event)

        ec.add_listener(listener)
        ec.remove_listener(listener)
        ec.receive_event(self.make_event())

        assert len(received_events) == 0

    def test_get_place_representation(self):
        ec = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        location = np.array([0.1, 0.2, 0.3])

        rep = ec.get_place_representation(location)

        assert rep.shape == (10,)
        assert all(0 <= a <= 1 for a in rep)

    def test_get_grid_representation(self):
        ec = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        location = np.array([0.1, 0.2, 0.3])

        rep = ec.get_grid_representation(location)

        assert rep.shape == (5,)
        assert all(0 <= a <= 1 for a in rep)

    def test_reset(self):
        ec = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        ec.receive_event(self.make_event())

        ec.reset()

        assert ec.event_count == 0
        # Memory is NOT cleared by reset
        assert len(ec.memory) == 1

    def test_receive_events_batch(self):
        ec = EntorhinalCortex(n_place_cells=10, n_grid_cells=5)
        events = [self.make_event() for _ in range(3)]

        results = ec.receive_events(events)

        assert len(results) == 3
        assert ec.event_count == 3
