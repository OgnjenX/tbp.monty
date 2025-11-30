"""Tests for Dentate Gyrus pattern separation."""

import numpy as np
import pytest

from tbp.hippocampus.dentate_gyrus import DentateGyrus, DGConfig, DGOutput
from tbp.hippocampus.types import SpatialEvent


def event_to_vector(event: SpatialEvent) -> np.ndarray:
    """Convert SpatialEvent to input vector for DG."""
    # Combine location (3) + flattened orientation (9) + confidence (1) = 13
    return np.concatenate([
        event.location,
        event.orientation.flatten(),
        [event.confidence]
    ])


class TestDGConfig:
    """Test DentateGyrus configuration."""

    def test_default_config(self):
        """Default config has biologically plausible values."""
        config = DGConfig()
        assert config.n_granule_cells > 0
        assert 0.001 <= config.sparsity <= 0.01  # 0.1-1% biological range
        assert config.neurogenesis_rate > 0

    def test_custom_config(self):
        """Custom config values are respected."""
        config = DGConfig(
            n_granule_cells=5000,
            sparsity=0.002,
            neurogenesis_rate=0.001,
        )
        assert config.n_granule_cells == 5000
        assert config.sparsity == 0.002


class TestDentateGyrus:
    """Test DentateGyrus pattern separation."""

    @pytest.fixture
    def dg(self):
        """Create DG with small size for testing."""
        config = DGConfig(n_input=20, n_granule_cells=1000, sparsity=0.01)
        return DentateGyrus(config)

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
    def sample_input(self, sample_event):
        """Create sample input vector from event."""
        return event_to_vector(sample_event)

    def test_encode_creates_sparse_pattern(self, dg, sample_input):
        """Encoding produces ultra-sparse patterns."""
        output = dg.encode(sample_input)

        assert isinstance(output, DGOutput)
        assert output.sparse_code.shape == (dg.config.n_granule_cells,)
        assert output.sparsity_achieved <= 0.02  # Should be very sparse

    def test_pattern_separation(self, dg, sample_event):
        """Similar inputs produce different sparse patterns."""
        input1 = event_to_vector(sample_event)
        output1 = dg.encode(input1)

        # Slightly different location
        similar_event = SpatialEvent(
            location=sample_event.location + 0.1,
            orientation=sample_event.orientation,
            features=sample_event.features,
            timestamp=1.0,
            source_id="test_source",
            confidence=1.0,
        )
        input2 = event_to_vector(similar_event)
        output2 = dg.encode(input2)

        # Patterns should have low overlap (pattern separation)
        overlap = np.sum(output1.sparse_code * output2.sparse_code) / (output1.n_active + 1e-10)
        # Some overlap is OK but shouldn't be perfect
        assert overlap < 1.0

    def test_output_fields(self, dg, sample_input):
        """DGOutput contains expected fields."""
        output = dg.encode(sample_input)

        assert output.sparse_code is not None
        assert output.active_indices is not None
        assert output.activation_values is not None
        assert 0.0 <= output.novelty_score <= 1.0
        assert output.n_active > 0

    def test_novelty_high_for_new_patterns(self, dg):
        """First pattern should have high novelty."""
        # Reset any state
        dg._recent_patterns = []

        input1 = np.random.randn(dg.config.n_input)
        output1 = dg.encode(input1)

        # First pattern should be novel
        assert output1.novelty_score >= 0.5

    def test_reset(self, dg, sample_input):
        """Reset clears internal state."""
        dg.encode(sample_input)
        dg.reset()

        # After reset, recent patterns should be cleared
        assert len(dg._recent_patterns) == 0

    def test_sparsity_constraint(self, dg, sample_input):
        """Output respects sparsity constraint."""
        output = dg.encode(sample_input)

        # Should be within 2x of target sparsity
        max_allowed_sparsity = dg.config.sparsity * 2
        assert output.sparsity_achieved <= max_allowed_sparsity
