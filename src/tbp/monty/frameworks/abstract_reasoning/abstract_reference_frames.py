# abstract_reference_frames.py

from typing import Dict, Optional

import numpy as np

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_orthonormal_vectors,
)


class AbstractReferenceFrame:
    """Reference frame for abstract concept spaces.

    Maps abstract concepts to coordinates in a multidimensional semantic space.
    Enables transformation between different perspectives on the same concepts.
    """

    def __init__(
            self,
            frame_id: str,
            domain: str,
            base_dimensions: np.ndarray,
            origin_concept: Optional[str] = None,
    ):
        """Initialize an abstract reference frame.

        Args:
            frame_id: Unique identifier for this reference frame
            domain: Abstract domain this frame belongs to (e.g., "philosophy", "mathematics")
            base_dimensions: Orthogonal axes defining the primary dimensions of this frame
                Shape (n_dims, embedding_size)
            origin_concept: Concept at the origin of this frame (optional)
        """
        self.frame_id = frame_id
        self.domain = domain
        self.base_dimensions = base_dimensions
        self.origin_concept = origin_concept
        self.dimensionality = len(base_dimensions)

        # Validate orthogonality of base dimensions
        self._validate_base_dimensions()

    def _validate_base_dimensions(self):
        """Ensure base dimensions are orthogonal."""
        for i in range(len(self.base_dimensions)):
            for j in range(i + 1, len(self.base_dimensions)):
                dot_product = np.dot(self.base_dimensions[i], self.base_dimensions[j])
                if abs(dot_product) > 1e-6:  # Allow small numerical errors
                    raise ValueError(
                        f"Base dimensions {i} and {j} are not orthogonal. "
                        f"Dot product: {dot_product}"
                    )

    def position_concept(self, concept_embedding: np.ndarray) -> np.ndarray:
        """Map a concept to coordinates in this reference frame.

        Args:
            concept_embedding: Vector embedding of the concept

        Returns:
            Coordinates of the concept in this reference frame
        """
        # Project concept embedding onto the base dimensions
        coordinates = np.zeros(self.dimensionality)
        for i, dimension in enumerate(self.base_dimensions):
            coordinates[i] = np.dot(concept_embedding, dimension)
        return coordinates

    def transform_to(
            self,
            concept_position: np.ndarray,
            target_frame: 'AbstractReferenceFrame'
    ) -> np.ndarray:
        """Transform concept coordinates from this frame to target frame.

        Args:
            concept_position: Coordinates in this reference frame
            target_frame: Target reference frame

        Returns:
            Coordinates in the target reference frame
        """
        # Calculate transformation matrix between the reference frames
        transform_matrix, _ = align_orthonormal_vectors(
            self.base_dimensions,
            target_frame.base_dimensions,
            as_scipy=False,
        )

        # Apply transformation
        new_position = np.dot(transform_matrix, concept_position)
        return new_position

    def transform_graph(self, source_graph):
        # TODO document why this method is empty
        pass


class DomainReferenceFrameRegistry:
    """Registry for managing reference frames across different abstract domains."""

    def __init__(self):
        """Initialize the reference frame registry."""
        self.reference_frames = {}  # domain -> {frame_id -> AbstractReferenceFrame}

    def register_frame(self, frame: AbstractReferenceFrame):
        """Register a reference frame with the registry.

        Args:
            frame: The reference frame to register
        """
        if frame.domain not in self.reference_frames:
            self.reference_frames[frame.domain] = {}

        self.reference_frames[frame.domain][frame.frame_id] = frame

    def get_frame(self, domain: str, frame_id: str) -> AbstractReferenceFrame:
        """Get a reference frame by domain and ID.

        Args:
            domain: Abstract domain
            frame_id: Unique identifier for the frame

        Returns:
            The requested AbstractReferenceFrame

        Raises:
            KeyError: If the requested frame doesn't exist
        """
        if domain not in self.reference_frames or frame_id not in self.reference_frames[domain]:
            raise KeyError(f"Reference frame {frame_id} not found in domain {domain}")

        return self.reference_frames[domain][frame_id]

    def get_domain_frames(self, domain: str) -> Dict[str, AbstractReferenceFrame]:
        """Get all reference frames for a specific domain.

        Args:
            domain: Abstract domain

        Returns:
            Dictionary of frame_id -> AbstractReferenceFrame
        """
        if domain not in self.reference_frames:
            return {}

        return self.reference_frames[domain]


# Global registry for abstract reference frames
ABSTRACT_FRAME_REGISTRY = DomainReferenceFrameRegistry()
