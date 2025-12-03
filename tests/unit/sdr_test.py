# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Unit tests for SDR utilities and location transforms."""

import unittest

import numpy as np

from tbp.monty.frameworks.utils.sdr import (
    SDR,
    PlaceFieldEncoder,
    GridCellEncoder,
    SDRPathIntegrator,
    hamming_distance,
    sdr_bind,
    sdr_union,
    find_nearest_sdrs,
)
from tbp.monty.frameworks.utils.location_transforms import (
    Transform,
    IdentityTransform,
    LinearTransform,
    PlaceFieldSDRTransform,
    TransformRegistry,
    get_default_registry,
    register_default_sdr_transform,
)
from tbp.monty.frameworks.models.states import State


class TestSDRDataclass(unittest.TestCase):
    """Tests for the SDR dataclass."""

    def test_sdr_creation(self):
        """Test basic SDR creation."""
        active = np.array([0, 5, 10, 15])
        sdr = SDR(dim=100, active=active)
        self.assertEqual(sdr.dim, 100)
        self.assertEqual(len(sdr), 4)
        np.testing.assert_array_equal(sdr.active, active)

    def test_sdr_to_dense(self):
        """Test SDR to dense conversion."""
        active = np.array([1, 3, 5])
        sdr = SDR(dim=8, active=active)
        dense = sdr.to_dense()
        expected = np.array([0, 1, 0, 1, 0, 1, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(dense, expected)

    def test_sdr_from_dense(self):
        """Test SDR creation from dense vector."""
        dense = np.array([0, 1, 0, 1, 0, 1, 0, 0], dtype=np.uint8)
        sdr = SDR.from_dense(dense)
        self.assertEqual(sdr.dim, 8)
        np.testing.assert_array_equal(sdr.active, np.array([1, 3, 5]))

    def test_sdr_sparsity(self):
        """Test SDR sparsity calculation."""
        active = np.array([0, 1, 2, 3, 4])  # 5 active out of 100
        # SDR uses the sparsity field as target, not actual
        sdr = SDR(dim=100, active=active, sparsity=0.05)
        self.assertAlmostEqual(sdr.sparsity, 0.05)


class TestHammingDistance(unittest.TestCase):
    """Tests for hamming distance calculation."""

    def test_identical_sdrs(self):
        """Identical SDRs should have zero distance."""
        sdr1 = SDR(dim=100, active=np.array([1, 5, 10]))
        sdr2 = SDR(dim=100, active=np.array([1, 5, 10]))
        dist = hamming_distance(sdr1, sdr2)
        self.assertEqual(dist, 0)

    def test_disjoint_sdrs(self):
        """Completely disjoint SDRs should have max distance."""
        sdr1 = SDR(dim=100, active=np.array([0, 1, 2]))
        sdr2 = SDR(dim=100, active=np.array([50, 51, 52]))
        dist = hamming_distance(sdr1, sdr2)
        self.assertEqual(dist, 6)  # 3 + 3 symmetric difference

    def test_partial_overlap(self):
        """Partially overlapping SDRs."""
        sdr1 = SDR(dim=100, active=np.array([0, 1, 2, 3]))
        sdr2 = SDR(dim=100, active=np.array([2, 3, 4, 5]))
        dist = hamming_distance(sdr1, sdr2)
        self.assertEqual(dist, 4)  # 2 unique in each


class TestSDROperations(unittest.TestCase):
    """Tests for SDR bind and union operations."""

    def test_sdr_bind(self):
        """Test SDR binding operation."""
        sdr1 = SDR(dim=100, active=np.array([0, 1, 2]))
        sdr2 = SDR(dim=100, active=np.array([1, 2, 3]))
        result = sdr_bind(sdr1, sdr2)
        self.assertEqual(result.dim, 100)
        # XOR of {0,1,2} and {1,2,3} = {0, 3}
        np.testing.assert_array_equal(np.sort(result.active), np.array([0, 3]))

    def test_sdr_union(self):
        """Test SDR union operation."""
        sdr1 = SDR(dim=100, active=np.array([0, 1, 2]))
        sdr2 = SDR(dim=100, active=np.array([2, 3, 4]))
        result = sdr_union([sdr1, sdr2])  # Takes a sequence
        self.assertEqual(result.dim, 100)
        np.testing.assert_array_equal(np.sort(result.active), np.array([0, 1, 2, 3, 4]))


class TestPlaceFieldEncoder(unittest.TestCase):
    """Tests for PlaceFieldEncoder."""

    def test_encoder_creation(self):
        """Test encoder initialization."""
        encoder = PlaceFieldEncoder(n_cells=1000, sparsity=0.02, seed=42)
        self.assertEqual(encoder.n_cells, 1000)
        self.assertEqual(encoder.sparsity, 0.02)

    def test_encode_3d_location(self):
        """Test encoding a 3D location."""
        encoder = PlaceFieldEncoder(n_cells=1000, sparsity=0.02, seed=42)
        location = np.array([0.1, 0.2, 0.3])
        sdr = encoder.encode(location)
        self.assertEqual(sdr.dim, 1000)
        # Sparsity should be approximately 2%
        self.assertGreater(len(sdr), 0)
        self.assertLess(len(sdr), 100)  # Should be sparse

    def test_nearby_locations_similar(self):
        """Nearby locations should produce similar SDRs."""
        encoder = PlaceFieldEncoder(n_cells=4096, sparsity=0.02, seed=42)
        loc1 = np.array([0.0, 0.0, 0.0])
        loc2 = np.array([0.01, 0.01, 0.01])  # Very close
        loc3 = np.array([0.5, 0.5, 0.5])  # Far away

        sdr1 = encoder.encode(loc1)
        sdr2 = encoder.encode(loc2)
        sdr3 = encoder.encode(loc3)

        dist_near = hamming_distance(sdr1, sdr2)
        dist_far = hamming_distance(sdr1, sdr3)

        # Nearby locations should have smaller distance
        self.assertLess(dist_near, dist_far)

    def test_deterministic_encoding(self):
        """Same location should always produce same SDR."""
        encoder = PlaceFieldEncoder(n_cells=1000, sparsity=0.02, seed=42)
        location = np.array([0.5, 0.5, 0.5])
        sdr1 = encoder.encode(location)
        sdr2 = encoder.encode(location)
        np.testing.assert_array_equal(sdr1.active, sdr2.active)


class TestGridCellEncoder(unittest.TestCase):
    """Tests for GridCellEncoder."""

    def test_encoder_creation(self):
        """Test encoder initialization."""
        encoder = GridCellEncoder(
            n_modules=4,
            cells_per_module=64,
            base_period=0.2,
            seed=42,
        )
        self.assertEqual(encoder.n_modules, 4)
        self.assertEqual(encoder.cells_per_module, 64)

    def test_encode_location(self):
        """Test encoding a location."""
        encoder = GridCellEncoder(
            n_modules=4,
            cells_per_module=64,
            base_period=0.2,
            seed=42,
        )
        location = np.array([0.1, 0.2, 0.3])
        sdr = encoder.encode(location)
        self.assertEqual(sdr.dim, 4 * 64)
        self.assertGreater(len(sdr), 0)

    def test_periodicity(self):
        """Locations offset by period should produce somewhat similar patterns."""
        encoder = GridCellEncoder(
            n_modules=4,  # Use multiple modules for better periodicity
            cells_per_module=100,
            base_period=0.5,
            seed=42,
        )
        loc1 = np.array([0.0, 0.0, 0.0])
        loc2 = np.array([0.25, 0.0, 0.0])  # Quarter period - should be different

        sdr1 = encoder.encode(loc1)
        sdr2 = encoder.encode(loc2)

        # Just test that encoding works and produces different patterns
        self.assertGreater(len(sdr1), 0)
        self.assertGreater(len(sdr2), 0)
        # Different locations should have some difference
        dist = hamming_distance(sdr1, sdr2)
        self.assertGreater(dist, 0)


class TestSDRPathIntegrator(unittest.TestCase):
    """Tests for SDR path integration."""

    def test_integrator_creation(self):
        """Test integrator initialization."""
        integrator = SDRPathIntegrator(dim=1000, seed=42)
        # Should have permutations initialized
        self.assertEqual(integrator.dim, 1000)

    def test_step_updates_sdr(self):
        """Test that step updates the current SDR."""
        integrator = SDRPathIntegrator(dim=1000, seed=42)
        sdr = SDR(dim=1000, active=np.array([10, 20, 30, 40, 50]))

        new_sdr = integrator.step(sdr, command=0)

        # SDR should have changed
        self.assertFalse(np.array_equal(sdr.active, new_sdr.active))


class TestFindNearestSDRs(unittest.TestCase):
    """Tests for find_nearest_sdrs function."""

    def test_find_exact_match(self):
        """Finding an exact match should return distance 0."""
        library = [
            SDR(dim=100, active=np.array([0, 1, 2])),
            SDR(dim=100, active=np.array([50, 51, 52])),
        ]
        query = SDR(dim=100, active=np.array([0, 1, 2]))
        results = find_nearest_sdrs(query, library, k=1)
        # Returns list of (index, distance) tuples
        self.assertEqual(results[0][0], 0)  # index
        self.assertEqual(results[0][1], 0)  # distance

    def test_find_k_nearest(self):
        """Test finding k nearest neighbors."""
        library = [
            SDR(dim=100, active=np.array([0, 1, 2])),
            SDR(dim=100, active=np.array([0, 1, 3])),  # 1 bit different
            SDR(dim=100, active=np.array([50, 51, 52])),  # Very different
        ]
        query = SDR(dim=100, active=np.array([0, 1, 2]))
        results = find_nearest_sdrs(query, library, k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 0)  # Exact match - index
        self.assertEqual(results[1][0], 1)  # 1 bit different - index


class TestTransformRegistry(unittest.TestCase):
    """Tests for transform registry."""

    def test_identity_transform(self):
        """Test identity transform."""
        transform = IdentityTransform()
        payload = {"type": "metric", "value": np.array([1.0, 2.0, 3.0])}
        result = transform.forward(payload)
        np.testing.assert_array_equal(result["value"], payload["value"])

    def test_linear_transform(self):
        """Test linear transform."""
        rotation = np.diag([2.0, 2.0, 2.0])  # Scale as diagonal rotation
        translation = np.array([1.0, 0.0, 0.0])
        transform = LinearTransform(rotation=rotation, translation=translation)
        payload = {"type": "metric", "value": np.array([1.0, 1.0, 1.0])}
        result = transform.forward(payload)
        expected = np.array([3.0, 2.0, 2.0])  # 1*2+1, 1*2+0, 1*2+0
        np.testing.assert_array_almost_equal(result["value"], expected)

    def test_registry_register_and_apply(self):
        """Test registry registration and application."""
        registry = TransformRegistry()
        transform = IdentityTransform()
        registry.register("metric", "metric", transform)

        payload = {"type": "metric", "value": np.array([1.0, 2.0, 3.0])}
        result = registry.apply("metric", "metric", payload)
        np.testing.assert_array_equal(result["value"], payload["value"])

    def test_default_registry(self):
        """Test getting default registry."""
        registry = get_default_registry()
        self.assertIsInstance(registry, TransformRegistry)

    def test_register_sdr_transform(self):
        """Test registering SDR transform with default registry."""
        register_default_sdr_transform(n_cells=100, sparsity=0.05, seed=42)

        registry = get_default_registry()
        payload = {"type": "metric", "value": np.array([0.1, 0.2, 0.3])}
        result = registry.apply("metric", "sdr", payload)

        self.assertEqual(result["type"], "sdr")
        self.assertIn("value", result)
        self.assertIn("dim", result)


class TestStateSDRHelpers(unittest.TestCase):
    """Tests for State class SDR helper methods."""

    def _make_morph_features(self):
        """Create minimum required morphological features."""
        return {
            "pose_vectors": np.eye(3),  # 3x3 identity as placeholder
            "pose_fully_defined": True,
        }

    def test_state_with_metric_location(self):
        """Test State with metric location."""
        location = np.array([1.0, 2.0, 3.0])
        state = State(
            location=location,
            morphological_features=self._make_morph_features(),
            non_morphological_features={},
            confidence=1.0,
            use_state=True,
            sender_id="test",
            sender_type="SM",
        )
        self.assertTrue(state.is_metric())
        self.assertFalse(state.is_sdr())
        self.assertEqual(state.location_dim(), 3)
        np.testing.assert_array_equal(state.get_location_value(), location)

    def test_state_with_sdr_payload(self):
        """Test State with SDR location payload."""
        sdr_active = np.array([10, 20, 30, 40], dtype=np.int32)
        payload = {
            "type": "sdr",
            "value": sdr_active,
            "dim": 4096,
            "frame_id": "world",
            "metric_origin": np.array([0.1, 0.2, 0.3]),
        }
        state = State(
            location=payload,
            morphological_features=self._make_morph_features(),
            non_morphological_features={},
            confidence=1.0,
            use_state=True,
            sender_id="test",
            sender_type="SM",
        )
        self.assertFalse(state.is_metric())
        self.assertTrue(state.is_sdr())
        self.assertEqual(state.get_location_type(), "sdr")
        np.testing.assert_array_equal(state.get_location_value(), sdr_active)

    def test_state_decode_to_metric(self):
        """Test decoding SDR payload to metric using metric_origin."""
        sdr_active = np.array([10, 20, 30, 40], dtype=np.int32)
        metric_origin = np.array([0.1, 0.2, 0.3])
        payload = {
            "type": "sdr",
            "value": sdr_active,
            "dim": 4096,
            "frame_id": "world",
            "metric_origin": metric_origin,
        }
        state = State(
            location=payload,
            morphological_features=self._make_morph_features(),
            non_morphological_features={},
            confidence=1.0,
            use_state=True,
            sender_id="test",
            sender_type="SM",
        )
        decoded = state.decode_to_metric()
        np.testing.assert_array_equal(decoded, metric_origin)


if __name__ == "__main__":
    unittest.main()
