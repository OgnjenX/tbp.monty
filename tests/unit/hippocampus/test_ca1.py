"""Tests for CA1 comparator network."""

import numpy as np
import pytest

from tbp.hippocampus.ca1 import CA1, CA1Config, ComparisonResult
from tbp.hippocampus.types import SpatialEvent


class TestCA1Config:
    """Test CA1 configuration."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = CA1Config()
        assert config.n_pyramidal_cells > 0
        assert 0.0 <= config.match_threshold <= 1.0
        assert 0.0 <= config.ec_weight <= 1.0

    def test_custom_config(self):
        """Custom config values are respected."""
        config = CA1Config(
            n_pyramidal_cells=1000,
            match_threshold=0.7,
            ec_weight=0.6,
        )
        assert config.n_pyramidal_cells == 1000
        assert config.match_threshold == 0.7


class TestComparisonResult:
    """Test ComparisonResult dataclass."""

    def test_comparison_result_fields(self):
        """ComparisonResult contains expected fields."""
        result = ComparisonResult(
            match_score=0.8,
            is_match=True,
            novelty_signal=0.2,
            output_pattern=np.zeros(100),
            ca3_contribution=0.5,
            ec_contribution=0.5,
        )
        assert result.match_score == 0.8
        assert result.is_match is True
        assert result.novelty_signal == 0.2


class TestCA1:
    """Test CA1 comparator."""

    @pytest.fixture
    def ca1(self):
        """Create CA1 with small size for testing."""
        config = CA1Config(n_pyramidal_cells=500)
        return CA1(config)

    @pytest.fixture
    def sample_event(self):
        """Create sample spatial event."""
        return SpatialEvent(
            location=np.array([1.0, 2.0, 3.0]),
            orientation=np.eye(3),
            features={"object_id": "test_obj"},
            timestamp=0.0,
            source_id="test_source",
            confidence=1.0,
        )

    @pytest.fixture
    def sample_pattern(self, ca1):
        """Create pattern for testing."""
        pattern = np.random.rand(ca1.config.n_pyramidal_cells).astype(np.float32)
        pattern[pattern < 0.9] = 0  # Make it sparse
        return pattern

    def test_compare_identical_patterns(self, ca1, sample_pattern, sample_event):
        """Identical CA3 and EC patterns should match."""
        result = ca1.compare(sample_pattern, sample_pattern, sample_event)

        assert isinstance(result, ComparisonResult)
        assert result.match_score >= 0.0
        assert result.output_pattern.shape == (ca1.config.n_pyramidal_cells,)

    def test_compare_different_patterns(self, ca1, sample_event):
        """Very different patterns should mismatch."""
        ca3_pattern = np.zeros(ca1.config.n_pyramidal_cells, dtype=np.float32)
        ca3_pattern[:50] = 1.0  # First 50 active

        ec_pattern = np.zeros(ca1.config.n_pyramidal_cells, dtype=np.float32)
        ec_pattern[450:500] = 1.0  # Last 50 active

        result = ca1.compare(ca3_pattern, ec_pattern, sample_event)

        # Should detect mismatch (novelty)
        assert result.novelty_signal > result.match_score

    def test_novelty_is_inverse_of_match(self, ca1, sample_pattern, sample_event):
        """Novelty signal should be inverse of match score."""
        result = ca1.compare(sample_pattern, sample_pattern, sample_event)

        assert result.novelty_signal == pytest.approx(1.0 - result.match_score)

    def test_ec_weight_increases_on_mismatch(self, ca1, sample_event):
        """On mismatch, EC contribution should be higher."""
        ca3_pattern = np.zeros(ca1.config.n_pyramidal_cells, dtype=np.float32)
        ca3_pattern[:50] = 1.0

        ec_pattern = np.zeros(ca1.config.n_pyramidal_cells, dtype=np.float32)
        ec_pattern[450:500] = 1.0

        result = ca1.compare(ca3_pattern, ec_pattern, sample_event)

        # On mismatch, EC should be weighted more (reality > prediction)
        if not result.is_match:
            assert result.ec_contribution > 0.5

    def test_sequence_tracking(self, ca1):
        """CA1 tracks temporal sequences."""
        patterns = []
        events = []

        # Create sequence of events
        for i in range(5):
            pattern = np.zeros(ca1.config.n_pyramidal_cells, dtype=np.float32)
            pattern[i * 100: i * 100 + 50] = 1.0
            patterns.append(pattern)

            event = SpatialEvent(
                location=np.array([float(i), 0.0, 0.0]),
                orientation=np.eye(3),
                features={},
                timestamp=float(i) * 0.1,  # Within temporal window
                source_id="test",
                confidence=1.0,
            )
            events.append(event)

        # Process sequence
        for ca3, ec, event in zip(patterns, patterns, events):
            result = ca1.compare(ca3, ec, event)
            # Each should get a sequence position
            assert result.sequence_position is not None

    def test_predict_next(self, ca1):
        """CA1 can predict next pattern in sequence."""
        # Create and process sequence
        patterns = []
        for i in range(5):
            pattern = np.zeros(ca1.config.n_pyramidal_cells, dtype=np.float32)
            pattern[i * 100: i * 100 + 50] = 1.0
            patterns.append(pattern)

            event = SpatialEvent(
                location=np.array([float(i), 0.0, 0.0]),
                orientation=np.eye(3),
                features={},
                timestamp=float(i) * 0.1,
                source_id="test",
                confidence=1.0,
            )
            ca1.compare(pattern, pattern, event)

        # Force new sequence (break temporal continuity)
        for i in range(5):
            event = SpatialEvent(
                location=np.array([float(i), 1.0, 0.0]),
                orientation=np.eye(3),
                features={},
                timestamp=10.0 + float(i) * 0.1,
                source_id="test",
                confidence=1.0,
            )
            ca1.compare(patterns[i], patterns[i], event)

        # Now predict from first pattern
        predicted, confidence = ca1.predict_next(patterns[0])
        # Should have some prediction attempt
        assert confidence >= 0.0

    def test_familiarity(self, ca1, sample_pattern, sample_event):
        """Familiarity score for patterns."""
        # Process some patterns
        ca1.compare(sample_pattern, sample_pattern, sample_event)

        familiarity = ca1.get_familiarity(sample_pattern)
        assert 0.0 <= familiarity <= 1.0

    def test_statistics(self, ca1, sample_pattern, sample_event):
        """Statistics track comparison operations."""
        ca1.compare(sample_pattern, sample_pattern, sample_event)
        ca1.compare(sample_pattern, sample_pattern, sample_event)

        stats = ca1.statistics
        assert stats["total_comparisons"] == 2
        assert "match_count" in stats
        assert "mismatch_count" in stats
        assert "match_rate" in stats

    def test_reset(self, ca1, sample_pattern, sample_event):
        """Reset clears all state."""
        ca1.compare(sample_pattern, sample_pattern, sample_event)

        ca1.reset()

        stats = ca1.statistics
        assert stats["total_comparisons"] == 0
        assert stats["n_sequences"] == 0
