# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Tests for HippocampalGraphLM - fast episodic relational learning."""

import numpy as np
import pytest

from tbp.monty.frameworks.models.hippocampal_graph_lm import (
    HippocampalGraphLM,
    HippocampalGraphMemory,
    ObjectObservation,
    ObjectRelation,
    ReplayBatch,
)


class TestObjectObservation:
    """Test ObjectObservation dataclass."""
    
    def test_create_observation(self):
        """Test creating a basic observation."""
        obs = ObjectObservation(
            object_id="mug",
            location=np.array([1.0, 2.0, 3.0]),
            timestamp=0.5,
        )
        
        assert obs.object_id == "mug"
        assert np.allclose(obs.location, [1.0, 2.0, 3.0])
        assert obs.timestamp == pytest.approx(0.5)
        assert obs.confidence == pytest.approx(1.0)  # default
    
    def test_location_conversion(self):
        """Test that location gets converted to numpy array."""
        obs = ObjectObservation(
            object_id="cup",
            location=np.array([1, 2, 3]),
        )
        
        assert isinstance(obs.location, np.ndarray)
        assert np.allclose(obs.location, [1, 2, 3])


class TestObjectRelation:
    """Test ObjectRelation dataclass."""
    
    def test_create_relation(self):
        """Test creating a relation between objects."""
        rel = ObjectRelation(object_a="mug", object_b="spoon")
        
        assert rel.object_a == "mug"
        assert rel.object_b == "spoon"
        assert rel.co_occurrence_count == 0
        assert len(rel.spatial_displacements) == 0
    
    def test_add_observation(self):
        """Test adding observations to a relation."""
        rel = ObjectRelation(object_a="mug", object_b="spoon")
        
        rel.add_observation(
            spatial_displacement=np.array([0.2, 0, 0]),
            temporal_offset=1.0,
            context="kitchen",
        )
        
        assert rel.co_occurrence_count == 1
        assert len(rel.spatial_displacements) == 1
        assert len(rel.temporal_offsets) == 1
        assert "kitchen" in rel.contexts
    
    def test_average_spatial_displacement(self):
        """Test computing average spatial relation."""
        rel = ObjectRelation(object_a="mug", object_b="spoon")
        
        # Add multiple observations
        rel.add_observation(spatial_displacement=np.array([0.2, 0, 0]))
        rel.add_observation(spatial_displacement=np.array([0.3, 0.1, 0]))
        
        avg = rel.average_spatial_displacement
        assert avg is not None
        assert np.allclose(avg, [0.25, 0.05, 0])
    
    def test_average_temporal_offset(self):
        """Test computing average temporal offset."""
        rel = ObjectRelation(object_a="mug", object_b="spoon")
        
        rel.add_observation(temporal_offset=1.0)
        rel.add_observation(temporal_offset=1.5)
        
        assert rel.average_temporal_offset == pytest.approx(1.25)


class TestHippocampalGraphMemory:
    """Test HippocampalGraphMemory storage."""
    
    def test_initialization(self):
        """Test creating a memory instance."""
        memory = HippocampalGraphMemory(learning_rate=0.9)
        
        assert memory.learning_rate == pytest.approx(0.9)
        assert len(memory.relations) == 0
        assert len(memory.objects_seen) == 0
        assert memory.episode_count == 0
    
    def test_add_relation(self):
        """Test adding a relation between objects."""
        memory = HippocampalGraphMemory()
        
        memory.add_relation(
            object_a="mug",
            object_b="spoon",
            spatial_displacement=np.array([0.2, 0, 0]),
            temporal_offset=1.0,
        )
        
        # Check both directions are created
        assert ("mug", "spoon") in memory.relations
        assert ("spoon", "mug") in memory.relations
        
        # Check objects tracked
        assert "mug" in memory.objects_seen
        assert "spoon" in memory.objects_seen
        
        # Check relation data
        rel = memory.get_relation("mug", "spoon")
        assert rel is not None
        assert rel.co_occurrence_count == 1
        assert len(rel.spatial_displacements) == 1
    
    def test_get_related_objects(self):
        """Test querying related objects."""
        memory = HippocampalGraphMemory()
        
        # Add relations
        memory.add_relation("mug", "spoon")
        memory.add_relation("mug", "plate")
        memory.add_relation("spoon", "fork")
        
        # Query mug's relations
        related = memory.get_related_objects("mug")
        assert "spoon" in related
        assert "plate" in related
        assert "fork" not in related  # Not directly related to mug
    
    def test_get_most_common_neighbors(self):
        """Test getting most frequent co-occurrences."""
        memory = HippocampalGraphMemory()
        
        # Add relations with different frequencies
        memory.add_relation("mug", "spoon")
        memory.add_relation("mug", "spoon")  # Second time
        memory.add_relation("mug", "plate")
        memory.add_relation("mug", "fork")
        memory.add_relation("mug", "fork")  # Second time
        memory.add_relation("mug", "fork")  # Third time
        
        # Get top neighbors
        neighbors = memory.get_most_common_neighbors("mug", top_k=2)
        
        assert len(neighbors) == 2
        assert neighbors[0] == ("fork", 3)  # Most common
        assert neighbors[1] == ("spoon", 2)  # Second most
    
    def test_predict_next_object(self):
        """Test predicting next object in sequence."""
        memory = HippocampalGraphMemory()
        
        # Mug appears with spoon most often
        memory.add_relation("mug", "spoon", temporal_offset=1.0)
        memory.add_relation("mug", "spoon", temporal_offset=1.0)
        memory.add_relation("mug", "plate", temporal_offset=1.0)
        
        prediction = memory.predict_next_object("mug")
        assert prediction == "spoon"

    def test_predict_next_object_temporal(self):
        """Temporal prediction prefers objects that occur after the cue."""
        memory = HippocampalGraphMemory()

        # Both co-occur with mug, but only spoon tends to be after mug.
        for _ in range(3):
            memory.add_relation("mug", "spoon", temporal_offset=1.0)
        for _ in range(3):
            memory.add_relation("mug", "plate", temporal_offset=-1.0)

        prediction = memory.predict_next_object("mug", use_temporal=True)
        assert prediction == "spoon"
    
    def test_statistics(self):
        """Test getting memory statistics."""
        memory = HippocampalGraphMemory()
        
        memory.add_relation("mug", "spoon")
        memory.add_relation("mug", "plate")
        memory.episode_count = 2
        memory.total_observations = 5
        
        stats = memory.get_statistics()
        
        assert stats["n_objects"] == 3  # mug, spoon, plate
        assert stats["n_relations"] == 2  # 2 unique pairs (divided by 2)
        assert stats["episode_count"] == 2
        assert stats["total_observations"] == 5
    
    def test_clear(self):
        """Test clearing all memory."""
        memory = HippocampalGraphMemory()
        
        memory.add_relation("mug", "spoon")
        memory.episode_count = 1
        memory.total_observations = 10
        
        memory.clear()
        
        assert len(memory.relations) == 0
        assert len(memory.objects_seen) == 0
        assert memory.episode_count == 0
        assert memory.total_observations == 0


class TestHippocampalGraphLM:
    """Test HippocampalGraphLM learning module."""
    
    def test_initialization(self):
        """Test creating a hippocampal LM."""
        hipp_lm = HippocampalGraphLM(learning_rate=0.8)
        
        assert hipp_lm.graph_memory.learning_rate == pytest.approx(0.8)
        assert len(hipp_lm.current_episode_observations) == 0
    
    def test_observe_object(self):
        """Test observing an object during episode."""
        hipp_lm = HippocampalGraphLM()
        
        hipp_lm.observe_object(
            object_id="mug",
            location=np.array([1, 2, 3]),
            timestamp=0.0,
        )
        
        assert len(hipp_lm.current_episode_observations) == 1
        obs = hipp_lm.current_episode_observations[0]
        assert obs.object_id == "mug"
        assert np.allclose(obs.location, [1, 2, 3])
    
    def test_single_episode_learning(self):
        """Test learning relations from a single episode."""
        hipp_lm = HippocampalGraphLM()
        
        # Observe two objects in one episode
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
        
        # End episode to consolidate
        hipp_lm.end_episode()
        
        # Check relation was learned
        relation = hipp_lm.get_relation("mug", "spoon")
        assert relation is not None
        assert relation.co_occurrence_count == 1
        
        # Check spatial relation
        avg_disp = relation.average_spatial_displacement
        assert avg_disp is not None
        assert np.allclose(avg_disp, [0.2, 0, 0])
        
        # Check temporal relation
        assert relation.average_temporal_offset == pytest.approx(1.0)
    
    def test_multiple_episode_learning(self):
        """Test learning accumulates across episodes."""
        hipp_lm = HippocampalGraphLM()
        
        # Episode 1
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
        hipp_lm.end_episode()
        
        # Episode 2
        hipp_lm.observe_object("mug", location=np.array([1, 1, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([1.2, 1, 0]), timestamp=1)
        hipp_lm.end_episode()
        
        # Check relation strengthened
        relation = hipp_lm.get_relation("mug", "spoon")
        assert relation is not None
        assert relation.co_occurrence_count == 2
        
        # Check average spatial relation
        avg_disp = relation.average_spatial_displacement
        assert avg_disp is not None
        assert np.allclose(avg_disp, [0.2, 0, 0])  # Same displacement both times
    
    def test_multiple_objects_per_episode(self):
        """Test learning from episode with many objects."""
        hipp_lm = HippocampalGraphLM()
        
        # Episode with 3 objects
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
        hipp_lm.observe_object("plate", location=np.array([0.5, 0.3, 0]), timestamp=2)
        hipp_lm.end_episode()
        
        # Check all pairs have relations
        assert hipp_lm.get_relation("mug", "spoon") is not None
        assert hipp_lm.get_relation("mug", "plate") is not None
        assert hipp_lm.get_relation("spoon", "plate") is not None
        
        # Check counts
        stats = hipp_lm.get_statistics()
        assert stats["n_objects"] == 3
        assert stats["n_relations"] == 3  # 3 pairs
    
    def test_context_aware_learning(self):
        """Test learning with episode contexts."""
        hipp_lm = HippocampalGraphLM()
        
        # Episode in kitchen
        hipp_lm.set_episode_context("kitchen")
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
        hipp_lm.end_episode()
        
        # Episode in office
        hipp_lm.set_episode_context("office")
        hipp_lm.observe_object("mug", location=np.array([1, 0, 0]), timestamp=0)
        hipp_lm.observe_object("keyboard", location=np.array([1.5, 0, 0]), timestamp=1)
        hipp_lm.end_episode()
        
        # Check contexts stored
        mug_spoon = hipp_lm.get_relation("mug", "spoon")
        assert mug_spoon is not None
        assert "kitchen" in mug_spoon.contexts
        
        mug_keyboard = hipp_lm.get_relation("mug", "keyboard")
        assert mug_keyboard is not None
        assert "office" in mug_keyboard.contexts
    
    def test_get_related_objects(self):
        """Test querying related objects."""
        hipp_lm = HippocampalGraphLM()
        
        # Create relations
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
        hipp_lm.observe_object("plate", location=np.array([0.5, 0, 0]), timestamp=2)
        hipp_lm.end_episode()
        
        # Query
        related = hipp_lm.get_related_objects("mug")
        assert "spoon" in related
        assert "plate" in related
        assert len(related) == 2
    
    def test_predict_next_object(self):
        """Test predicting next object in sequence."""
        hipp_lm = HippocampalGraphLM()
        
        # Learn sequence: mug → spoon (twice)
        for _ in range(2):
            hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
            hipp_lm.observe_object("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
            hipp_lm.end_episode()
        
        # Learn sequence: mug → plate (once)
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("plate", location=np.array([0.5, 0, 0]), timestamp=1)
        hipp_lm.end_episode()
        
        # Prediction should favor spoon (more common)
        prediction = hipp_lm.predict_next_object("mug")
        assert prediction == "spoon"
    
    def test_get_most_common_neighbors(self):
        """Test getting most frequent co-occurrences."""
        hipp_lm = HippocampalGraphLM()
        
        # Create varying frequencies
        for _ in range(3):
            hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
            hipp_lm.observe_object("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
            hipp_lm.end_episode()
        
        for _ in range(1):
            hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
            hipp_lm.observe_object("plate", location=np.array([0.5, 0, 0]), timestamp=1)
            hipp_lm.end_episode()
        
        neighbors = hipp_lm.get_most_common_neighbors("mug", top_k=2)
        
        assert neighbors[0] == ("spoon", 3)
        assert neighbors[1] == ("plate", 1)
    
    def test_reset(self):
        """Test resetting episode buffer."""
        hipp_lm = HippocampalGraphLM()
        
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.set_episode_context("kitchen")
        
        hipp_lm.reset()
        
        assert len(hipp_lm.current_episode_observations) == 0
        assert hipp_lm.current_episode_context is None
    
    def test_clear_memory(self):
        """Test clearing all long-term memory."""
        hipp_lm = HippocampalGraphLM()
        
        # Learn some relations
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
        hipp_lm.end_episode()
        
        hipp_lm.clear_memory()
        
        # Check everything cleared
        stats = hipp_lm.get_statistics()
        assert stats["n_objects"] == 0
        assert stats["n_relations"] == 0
        assert stats["episode_count"] == 0
    
    def test_statistics(self):
        """Test getting comprehensive statistics."""
        hipp_lm = HippocampalGraphLM(learning_rate=0.75)
        
        # Create some data
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
        hipp_lm.end_episode()
        
        stats = hipp_lm.get_statistics()
        
        assert stats["n_objects"] == 2
        assert stats["n_relations"] == 1
        assert stats["episode_count"] == 1
        assert stats["total_observations"] == 2
        assert stats["learning_rate"] == pytest.approx(0.75)
        assert stats["current_episode_observations"] == 0  # After end_episode
    
    def test_repr(self):
        """Test string representation."""
        hipp_lm = HippocampalGraphLM()
        
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
        hipp_lm.end_episode()
        
        repr_str = repr(hipp_lm)
        assert "HippocampalGraphLM" in repr_str
        assert "objects=2" in repr_str
        assert "relations=1" in repr_str
        assert "episodes=1" in repr_str
    
    def test_no_self_relations(self):
        """Test that objects don't create relations with themselves."""
        hipp_lm = HippocampalGraphLM()
        
        # Observe same object twice
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("mug", location=np.array([0.1, 0, 0]), timestamp=1)
        hipp_lm.end_episode()
        
        # Check no self-relation created
        relation = hipp_lm.get_relation("mug", "mug")
        
        # Should either be None or have count 0
        if relation is not None:
            assert relation.co_occurrence_count == 0
    
    def test_auto_consolidation(self):
        """Test auto-consolidation when max observations reached."""
        hipp_lm = HippocampalGraphLM(max_observations_per_episode=3)
        
        # Add observations up to limit
        hipp_lm.observe_object("obj1", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("obj2", location=np.array([1, 0, 0]), timestamp=1)
        hipp_lm.observe_object("obj3", location=np.array([2, 0, 0]), timestamp=2)
        
        # This should trigger auto-consolidation
        assert hipp_lm.graph_memory.episode_count == 1
        assert len(hipp_lm.current_episode_observations) == 0  # Cleared after consolidation


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""
    
    def test_kitchen_scene_learning(self):
        """Test learning object relations in a kitchen scene."""
        hipp_lm = HippocampalGraphLM()
        
        # Episode 1: Kitchen table scene
        hipp_lm.set_episode_context("kitchen_table")
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([0.15, 0.05, 0]), timestamp=1)
        hipp_lm.observe_object("plate", location=np.array([0.3, 0.3, 0]), timestamp=2)
        hipp_lm.end_episode()
        
        # Episode 2: Same scene, slightly different positions
        hipp_lm.set_episode_context("kitchen_table")
        hipp_lm.observe_object("mug", location=np.array([1, 1, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([1.18, 1.02, 0]), timestamp=1)
        hipp_lm.observe_object("plate", location=np.array([1.35, 1.25, 0]), timestamp=2)
        hipp_lm.end_episode()
        
        # Check learned relations
        mug_neighbors = hipp_lm.get_most_common_neighbors("mug")
        assert len(mug_neighbors) == 2
        assert "spoon" in [obj for obj, _ in mug_neighbors]
        assert "plate" in [obj for obj, _ in mug_neighbors]
        
        # Check spatial relationships are consistent
        mug_spoon = hipp_lm.get_relation("mug", "spoon")
        assert mug_spoon is not None
        avg_disp = mug_spoon.average_spatial_displacement
        assert avg_disp is not None
        # Should be roughly [0.15-0.18, 0.02-0.05, 0]
        assert 0.1 < avg_disp[0] < 0.2
        assert -0.1 < avg_disp[1] < 0.1
    
    def test_compositional_object_learning(self):
        """Test learning compositional objects (e.g., mug with logo)."""
        hipp_lm = HippocampalGraphLM()
        
        # Episode: Observe mug and its logo as separate "objects"
        hipp_lm.set_episode_context("mug_with_logo")
        hipp_lm.observe_object("mug_body", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("nike_logo", location=np.array([0.05, 0.02, 0.08]), timestamp=0.1)
        hipp_lm.end_episode()
        
        # Another mug with same logo
        hipp_lm.set_episode_context("mug_with_logo")
        hipp_lm.observe_object("mug_body", location=np.array([1, 0, 0]), timestamp=0)
        hipp_lm.observe_object("nike_logo", location=np.array([1.05, 0.02, 0.08]), timestamp=0.1)
        hipp_lm.end_episode()
        
        # Check strong binding between mug and logo
        relation = hipp_lm.get_relation("mug_body", "nike_logo")
        assert relation is not None
        assert relation.co_occurrence_count == 2
        assert "mug_with_logo" in relation.contexts
        
        # Check spatial relation is consistent (logo on mug)
        avg_disp = relation.average_spatial_displacement
        assert avg_disp is not None
        assert np.allclose(avg_disp, [0.05, 0.02, 0.08], atol=0.01)
    
    def test_sequence_prediction(self):
        """Test predicting object sequences (e.g., meal preparation)."""
        hipp_lm = HippocampalGraphLM()
        
        # Learn sequence: plate → fork → knife (repeated)
        for _ in range(5):
            hipp_lm.observe_object("plate", location=np.array([0, 0, 0]), timestamp=0)
            hipp_lm.observe_object("fork", location=np.array([0.2, 0, 0]), timestamp=1)
            hipp_lm.observe_object("knife", location=np.array([0.4, 0, 0]), timestamp=2)
            hipp_lm.end_episode()
        
        # Test predictions - currently based on co-occurrence count
        # All three objects appear together equally often (5 times each)
        # So prediction just returns most common neighbor
        prediction = hipp_lm.predict_next_object("plate")
        assert prediction in ["fork", "knife"]  # Both are valid neighbors


class TestReplayBatch:
    """Test ReplayBatch dataclass."""
    
    def test_create_batch(self):
        """Test creating a replay batch."""
        obs = ObjectObservation(
            object_id="mug",
            location=np.array([0, 0, 0]),
            timestamp=0.0,
        )
        
        batch = ReplayBatch(
            observations=[obs],
            source_type="episode",
            batch_id="test_batch_1",
        )
        
        assert len(batch) == 1
        assert batch.source_type == "episode"
        assert batch.get_object_ids() == ["mug"]
    
    def test_batch_object_ids(self):
        """Test extracting object sequence from batch."""
        obs1 = ObjectObservation("mug", location=np.array([0, 0, 0]), timestamp=0)
        obs2 = ObjectObservation("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
        obs3 = ObjectObservation("plate", location=np.array([0.5, 0, 0]), timestamp=2)
        
        batch = ReplayBatch(
            observations=[obs1, obs2, obs3],
            source_type="relation_centric",
            batch_id="test_batch_2",
        )
        
        assert batch.get_object_ids() == ["mug", "spoon", "plate"]
        assert len(batch) == 3


class TestHippocampalReplay:
    """Test replay functionality."""
    
    def test_episode_replay_generation(self):
        """Test generating episode replay batches."""
        memory = HippocampalGraphMemory()
        
        # Store an episode
        obs1 = ObjectObservation("mug", location=np.array([0, 0, 0]), timestamp=0)
        obs2 = ObjectObservation("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
        episode = [obs1, obs2]
        
        memory.store_episode(episode)
        
        # Generate replay
        replays = memory.generate_episode_replay(num_replays=2)
        
        assert len(replays) == 2
        for batch in replays:
            assert batch.source_type == "episode"
            assert len(batch) == 2
            assert batch.get_object_ids() == ["mug", "spoon"]

    def test_episode_replay_context_filtering(self):
        """Test filtering stored episodes by context during replay."""
        memory = HippocampalGraphMemory()

        kitchen_episode = [
            ObjectObservation("mug", location=np.array([0, 0, 0]), timestamp=0),
            ObjectObservation("spoon", location=np.array([0.2, 0, 0]), timestamp=1),
        ]
        office_episode = [
            ObjectObservation("mug", location=np.array([1, 0, 0]), timestamp=0),
            ObjectObservation("keyboard", location=np.array([1.5, 0, 0]), timestamp=1),
        ]

        memory.store_episode(kitchen_episode, context="kitchen")
        memory.store_episode(office_episode, context="office")

        kitchen_replays = memory.generate_episode_replay(num_replays=5, context_filter="kitchen")
        assert len(kitchen_replays) == 5
        assert all(batch.context == "kitchen" for batch in kitchen_replays)
        assert all(batch.get_object_ids() == ["mug", "spoon"] for batch in kitchen_replays)

        missing_replays = memory.generate_episode_replay(num_replays=1, context_filter="garage")
        assert missing_replays == []

    def test_pattern_completion_retrieve_episodes_by_cue(self):
        """Test retrieving best-matching episodes from partial cues."""
        memory = HippocampalGraphMemory()

        kitchen_episode = [
            ObjectObservation("mug", location=np.array([0, 0, 0]), timestamp=0),
            ObjectObservation("spoon", location=np.array([0.2, 0, 0]), timestamp=1),
            ObjectObservation("plate", location=np.array([0.5, 0, 0]), timestamp=2),
        ]
        office_episode = [
            ObjectObservation("mug", location=np.array([1, 0, 0]), timestamp=0),
            ObjectObservation("keyboard", location=np.array([1.5, 0, 0]), timestamp=1),
        ]

        memory.store_episode(kitchen_episode, context="kitchen")
        memory.store_episode(office_episode, context="office")

        # Cue should retrieve kitchen episode
        matches = memory.retrieve_episodes_by_cue(["mug", "spoon"], top_k=1)
        assert len(matches) == 1
        assert matches[0].context == "kitchen"
        assert [o.object_id for o in matches[0].observations] == ["mug", "spoon", "plate"]

        # Context filtering should retrieve office episode
        matches_ctx = memory.retrieve_episodes_by_cue(
            ["mug"],
            context_filter="office",
            top_k=1,
        )
        assert len(matches_ctx) == 1
        assert matches_ctx[0].context == "office"

    def test_pattern_completion_complete_from_cue(self):
        """Test suggesting missing objects from partial cues."""
        memory = HippocampalGraphMemory()

        memory.store_episode(
            [
                ObjectObservation("mug", location=np.array([0, 0, 0]), timestamp=0),
                ObjectObservation("spoon", location=np.array([0.2, 0, 0]), timestamp=1),
                ObjectObservation("plate", location=np.array([0.5, 0, 0]), timestamp=2),
            ],
            context="kitchen",
        )
        memory.store_episode(
            [
                ObjectObservation("mug", location=np.array([1, 0, 0]), timestamp=0),
                ObjectObservation("keyboard", location=np.array([1.5, 0, 0]), timestamp=1),
            ],
            context="office",
        )

        # In kitchen context, completion from mug should not suggest office-only objects.
        suggestions = memory.complete_from_cue(["mug"], context_filter="kitchen", top_k_objects=5)
        suggested_ids = [obj_id for obj_id, _ in suggestions]
        assert "spoon" in suggested_ids
        assert "plate" in suggested_ids
        assert "keyboard" not in suggested_ids
    
    def test_relation_replay_generation(self):
        """Test generating relation-centric replay."""
        memory = HippocampalGraphMemory()
        
        # Create high-frequency relation
        for _ in range(5):
            memory.add_relation(
                "mug",
                "spoon",
                spatial_displacement=np.array([0.2, 0, 0]),
                temporal_offset=1.0,
                context="kitchen",
                timestamp=0.0,
            )
        
        # Create lower-frequency relation
        for _ in range(2):
            memory.add_relation(
                "mug",
                "plate",
                spatial_displacement=np.array([0.5, 0, 0]),
                temporal_offset=2.0,
                context="kitchen",
                timestamp=0.0,
            )
        
        # Generate relation-centric replays
        replays = memory.generate_relation_replay(
            num_replays=3,
            top_k_relations=2,
            temperature=1.0,
        )
        
        assert len(replays) == 3
        for batch in replays:
            assert batch.source_type == "relation_centric"
            assert len(batch) == 2  # Each mini-episode has 2 objects
            assert batch.get_object_ids() in [["mug", "spoon"], ["mug", "plate"]]
    
    def test_relation_replay_with_context_filter(self):
        """Test relation replay with context filtering."""
        memory = HippocampalGraphMemory()
        
        # Add relations in different contexts
        memory.add_relation(
            "mug", "spoon",
            context="kitchen",
            timestamp=0.0,
        )
        memory.add_relation(
            "mug", "keyboard",
            context="office",
            timestamp=0.0,
        )
        
        # Generate replays filtered to kitchen
        replays = memory.generate_relation_replay(
            num_replays=5,
            context_filter="kitchen",
        )
        
        # Should prefer kitchen relations
        kitchen_replays = sum(
            1 for batch in replays
            if "spoon" in batch.get_object_ids()
        )
        assert kitchen_replays > 0
    
    def test_lm_replay_callbacks(self):
        """Test registering and invoking replay callbacks."""
        hipp_lm = HippocampalGraphLM(enable_replay=True)
        
        # Record callbacks
        received_batches = []
        
        def collect_batch(batch: ReplayBatch):
            received_batches.append(batch)
        
        hipp_lm.register_replay_callback(collect_batch)
        
        # Learn an episode
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.observe_object("spoon", location=np.array([0.2, 0, 0]), timestamp=1)
        hipp_lm.end_episode()
        
        # Generate replays
        hipp_lm.replay_episodes(num_replays=2, invoke_callbacks=True)
        
        assert len(received_batches) == 2
        assert all(batch.source_type == "episode" for batch in received_batches)
    
    def test_relation_replay_temperature(self):
        """Test temperature parameter in relation replay."""
        memory = HippocampalGraphMemory()
        
        # Create relations with different frequencies
        for _ in range(10):
            memory.add_relation("a", "b", timestamp=0.0)
        for _ in range(1):
            memory.add_relation("a", "c", timestamp=0.0)
        
        # With temperature 0 (greedy), should always sample high-freq relation
        replays_greedy = memory.generate_relation_replay(
            num_replays=10,
            temperature=0.0,
        )
        greedy_b = sum(1 for b in replays_greedy if "b" in b.get_object_ids())
        
        # With high temperature, should sample more uniformly
        replays_hot = memory.generate_relation_replay(
            num_replays=10,
            temperature=10.0,
        )
        hot_b = sum(1 for b in replays_hot if "b" in b.get_object_ids())
        
        # Greedy should have higher proportion of high-freq relation
        assert greedy_b > hot_b
    
    def test_episode_history_pruning(self):
        """Test that episode history is pruned when exceeding max size."""
        hipp_lm = HippocampalGraphLM(enable_replay=True, max_episode_history=3)
        
        # Add 5 episodes
        for i in range(5):
            hipp_lm.observe_object(f"obj{i}", location=np.array([i, 0, 0]), timestamp=0)
            hipp_lm.end_episode()
        
        # History should only keep the last 3
        assert len(hipp_lm.graph_memory.episode_history) <= 3
    
    def test_disable_replay_storage(self):
        """Test that history is not stored when replay is disabled."""
        hipp_lm = HippocampalGraphLM(enable_replay=False)
        
        hipp_lm.observe_object("mug", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.end_episode()
        
        assert len(hipp_lm.graph_memory.episode_history) == 0
    
    def test_replay_with_multiple_callbacks(self):
        """Test multiple callbacks receiving the same replay batch."""
        hipp_lm = HippocampalGraphLM(enable_replay=True)
        
        batches_1 = []
        batches_2 = []
        
        hipp_lm.register_replay_callback(lambda b: batches_1.append(b))
        hipp_lm.register_replay_callback(lambda b: batches_2.append(b))
        
        # Learn and replay
        hipp_lm.observe_object("obj", location=np.array([0, 0, 0]), timestamp=0)
        hipp_lm.end_episode()
        
        hipp_lm.replay_episodes(num_replays=1, invoke_callbacks=True)
        
        assert len(batches_1) == 1
        assert len(batches_2) == 1
    
    def test_decay_rate_parameter(self):
        """Test that decay_rate is properly initialized."""
        hipp_lm = HippocampalGraphLM(decay_rate=0.1)
        
        assert hipp_lm.graph_memory.decay_rate == pytest.approx(0.1)
    
    def test_relation_replay_with_learned_displacement(self):
        """Test that relation replay uses learned spatial displacement."""
        memory = HippocampalGraphMemory()
        
        # Learn consistent spatial relation
        memory.add_relation(
            "mug",
            "spoon",
            spatial_displacement=np.array([0.15, 0.05, 0.0]),
            temporal_offset=0.5,
            timestamp=0.0,
        )
        
        # Generate replay
        replays = memory.generate_relation_replay(num_replays=1)
        assert len(replays) == 1
        
        batch = replays[0]
        obs_a = batch.observations[0]
        obs_b = batch.observations[1]
        
        # Check that displacement was used
        actual_displacement = obs_b.location - obs_a.location
        assert np.allclose(
            actual_displacement,
            np.array([0.15, 0.05, 0.0]),
            atol=0.01
        )
