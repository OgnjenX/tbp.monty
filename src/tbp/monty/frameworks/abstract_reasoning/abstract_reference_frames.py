# abstract_reference_frames.py

from typing import Dict, List, Optional, Tuple

import numpy as np

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_orthonormal_vectors,
)


class LearnedAbstractReferenceFrame:
    """Reference frame for abstract concept spaces learned through sensorimotor experience.

    Unlike traditional reference frames, this frame is built incrementally through
    temporal sequences of concept transitions and motor actions in abstract space.
    Aligns with TBT principles of sensorimotor learning and cortical column independence.
    """

    def __init__(
            self,
            frame_id: str,
            domain: str,
            max_dimensions: int = 10,
            learning_rate: float = 0.01,
            stability_threshold: float = 0.95,
    ):
        """Initialize a learned abstract reference frame.

        Args:
            frame_id: Unique identifier for this reference frame
            domain: Abstract domain this frame belongs to
            max_dimensions: Maximum number of dimensions to learn
            learning_rate: Rate at which the frame adapts to new experiences
            stability_threshold: Threshold for considering frame stable
        """
        self.frame_id = frame_id
        self.domain = domain
        self.max_dimensions = max_dimensions
        self.learning_rate = learning_rate
        self.stability_threshold = stability_threshold

        # Learned components - built through sensorimotor experience
        self.base_dimensions = []  # Learned orthogonal axes
        self.concept_positions = {}  # concept_id -> position in frame
        self.transition_history = []  # History of concept transitions
        self.motor_associations = {}  # motor_action -> expected_displacement

        # Learning state
        self.num_experiences = 0
        self.stability_score = 0.0
        self.is_stable = False

    def add_sensorimotor_experience(
        self,
        source_concept: str,
        target_concept: str,
        motor_action: np.ndarray,
        temporal_context: List[str]
    ) -> None:
        """Learn from a sensorimotor experience in abstract space.

        Args:
            source_concept: Starting concept
            target_concept: Ending concept after motor action
            motor_action: Motor action taken (displacement in abstract space)
            temporal_context: Recent sequence of concepts for context
        """
        self.num_experiences += 1

        # Record transition
        transition = {
            'source': source_concept,
            'target': target_concept,
            'motor_action': motor_action,
            'context': temporal_context.copy(),
            'timestamp': self.num_experiences
        }
        self.transition_history.append(transition)

        # Update motor associations
        self._update_motor_associations(motor_action, source_concept, target_concept)

        # Incrementally build reference frame
        self._update_reference_frame(source_concept, target_concept, motor_action)

        # Update stability
        self._update_stability()

    def _update_motor_associations(
        self,
        motor_action: np.ndarray,
        source: str,
        target: str
    ) -> None:
        """Update associations between motor actions and concept displacements."""
        action_key = tuple(motor_action)

        if action_key not in self.motor_associations:
            self.motor_associations[action_key] = {
                'expected_displacement': np.zeros_like(motor_action),
                'count': 0,
                'source_targets': []
            }

        # Record this source-target pair
        self.motor_associations[action_key]['source_targets'].append((source, target))
        self.motor_associations[action_key]['count'] += 1

    def _update_reference_frame(
        self,
        source_concept: str,
        target_concept: str,
        motor_action: np.ndarray
    ) -> None:
        """Incrementally update the reference frame based on new experience."""
        # Ensure concepts have positions
        if source_concept not in self.concept_positions:
            self.concept_positions[source_concept] = self._initialize_concept_position()
        if target_concept not in self.concept_positions:
            self.concept_positions[target_concept] = self._initialize_concept_position()

        # Calculate expected displacement based on motor action
        expected_displacement = self._motor_to_displacement(motor_action)

        # Calculate actual displacement
        source_pos = self.concept_positions[source_concept]
        target_pos = self.concept_positions[target_concept]
        actual_displacement = target_pos - source_pos

        # Update positions to reduce prediction error
        error = actual_displacement - expected_displacement
        adjustment = self.learning_rate * error

        # Adjust target position
        self.concept_positions[target_concept] -= adjustment / 2

        # Update base dimensions if needed
        self._adapt_base_dimensions(motor_action, actual_displacement)

    def _initialize_concept_position(self) -> np.ndarray:
        """Initialize position for a new concept."""
        # Start with small random position
        return np.random.normal(0, 0.1, self.max_dimensions)

    def _motor_to_displacement(self, motor_action: np.ndarray) -> np.ndarray:
        """Convert motor action to expected displacement in reference frame."""
        # Simple linear mapping - could be learned
        if len(self.base_dimensions) == 0:
            return np.zeros(self.max_dimensions)

        # Project motor action onto current base dimensions
        displacement = np.zeros(self.max_dimensions)
        for i, dimension in enumerate(self.base_dimensions):
            if i < len(motor_action):
                displacement[i] = motor_action[i]

        return displacement

    def _adapt_base_dimensions(
        self,
        motor_action: np.ndarray,
        displacement: np.ndarray
    ) -> None:
        """Adapt base dimensions based on observed motor-displacement relationships."""
        # Add new dimension if needed and we haven't reached max
        if len(self.base_dimensions) < self.max_dimensions:
            # Create new orthogonal dimension
            new_dim = self._create_orthogonal_dimension(displacement)
            if new_dim is not None:
                self.base_dimensions.append(new_dim)

    def _create_orthogonal_dimension(self, displacement: np.ndarray) -> Optional[np.ndarray]:
        """Create a new orthogonal dimension from displacement vector."""
        # Normalize displacement
        norm = np.linalg.norm(displacement)
        if norm < 1e-6:
            return None

        new_dim = displacement / norm

        # Make orthogonal to existing dimensions
        for existing_dim in self.base_dimensions:
            projection = np.dot(new_dim, existing_dim)
            new_dim = new_dim - projection * existing_dim

        # Check if still significant after orthogonalization
        final_norm = np.linalg.norm(new_dim)
        if final_norm < 1e-6:
            return None

        return new_dim / final_norm

    def _update_stability(self) -> None:
        """Update stability score based on consistency of recent experiences."""
        if len(self.transition_history) < 10:
            self.stability_score = 0.0
            return

        # Check consistency of recent motor-displacement relationships
        recent_transitions = self.transition_history[-10:]
        consistency_scores = []

        for transition in recent_transitions:
            predicted_displacement = self._motor_to_displacement(transition['motor_action'])
            if transition['target'] in self.concept_positions and transition['source'] in self.concept_positions:
                actual_displacement = (
                    self.concept_positions[transition['target']] -
                    self.concept_positions[transition['source']]
                )
                error = np.linalg.norm(predicted_displacement - actual_displacement)
                consistency_scores.append(1.0 / (1.0 + error))

        if consistency_scores:
            self.stability_score = np.mean(consistency_scores)
            self.is_stable = self.stability_score > self.stability_threshold

    def get_concept_position(self, concept_id: str) -> Optional[np.ndarray]:
        """Get position of a concept in this reference frame."""
        return self.concept_positions.get(concept_id)

    def predict_motor_action(self, source_concept: str, target_concept: str) -> Optional[np.ndarray]:
        """Predict motor action needed to transition from source to target concept."""
        if source_concept not in self.concept_positions or target_concept not in self.concept_positions:
            return None

        displacement = self.concept_positions[target_concept] - self.concept_positions[source_concept]

        # Convert displacement back to motor action (inverse of _motor_to_displacement)
        if len(self.base_dimensions) == 0:
            return np.zeros(3)  # Default 3D motor action

        motor_action = np.zeros(min(3, len(self.base_dimensions)))
        for i in range(len(motor_action)):
            if i < len(self.base_dimensions):
                motor_action[i] = displacement[i]

        return motor_action

    def get_nearby_concepts(self, concept_id: str, radius: float = 1.0) -> List[Tuple[str, float]]:
        """Get concepts within a certain radius of the given concept."""
        if concept_id not in self.concept_positions:
            return []

        concept_pos = self.concept_positions[concept_id]
        nearby = []

        for other_id, other_pos in self.concept_positions.items():
            if other_id != concept_id:
                distance = np.linalg.norm(concept_pos - other_pos)
                if distance <= radius:
                    nearby.append((other_id, distance))

        return sorted(nearby, key=lambda x: x[1])
