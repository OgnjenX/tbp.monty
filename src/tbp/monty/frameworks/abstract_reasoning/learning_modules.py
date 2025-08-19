import copy
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np

from tbp.monty.frameworks.abstract_reasoning.abstract_reference_frames import (
    LearnedAbstractReferenceFrame,
)
from tbp.monty.frameworks.abstract_reasoning.concept_embeddings import (
    TemporalConceptEmbedding,
    LocalConceptEmbeddingManager,
)
from tbp.monty.frameworks.abstract_reasoning.abstract_motor_system import (
    AbstractMotorAction,
    AbstractMotorSystem,
    AbstractMotorPolicy,
)
from tbp.monty.frameworks.models.evidence_matching import EvidenceGraphLM
from tbp.monty.frameworks.models.states import State, GoalState
from tbp.monty.frameworks.utils.spatial_arithmetics import align_orthonormal_vectors


class SensorimotorAbstractReasoningLM(EvidenceGraphLM):
    """Sensorimotor learning module for abstract reasoning domains.

    Implements TBT principles for abstract reasoning through sensorimotor learning,
    temporal sequence processing, and learned reference frames.
    """

    def __init__(
            self,
            lm_id: str,
            reasoning_domain: str,
            temporal_window: int = 15,
            embedding_dim: int = 50,
            motor_system: Optional[AbstractMotorSystem] = None,
            **kwargs,
    ):
        """Initialize a sensorimotor abstract reasoning learning module.

        Args:
            lm_id: Unique identifier for this learning module
            reasoning_domain: Domain of abstract reasoning
            temporal_window: Size of temporal sequence buffer
            embedding_dim: Dimensionality of concept embeddings
            motor_system: Motor system for abstract actions
            **kwargs: Additional arguments for EvidenceGraphLM
        """
        super().__init__(lm_id=lm_id, **kwargs)
        self.lm_id = lm_id
        self.reasoning_domain = reasoning_domain
        self.temporal_window = temporal_window
        self.embedding_dim = embedding_dim

        # Local components - no global dependencies
        self.concept_manager = LocalConceptEmbeddingManager(
            module_id=lm_id,
            embedding_dim=embedding_dim,
            temporal_window=temporal_window
        )

        self.reference_frame = LearnedAbstractReferenceFrame(
            frame_id=f"{lm_id}_learned_frame",
            domain=reasoning_domain,
            max_dimensions=min(10, embedding_dim)
        )

        # Motor system for abstract exploration
        self.motor_system = motor_system or self._create_default_motor_system()

        # Temporal processing components
        self.state_sequence = deque(maxlen=temporal_window)
        self.concept_sequence = deque(maxlen=temporal_window)
        self.motor_sequence = deque(maxlen=temporal_window)
        self.evidence_sequence = deque(maxlen=temporal_window)

        # Sensorimotor learning state
        self.current_concept = None
        self.current_position = np.zeros(3)
        self.last_motor_action = None
        self.exploration_goal = None

        # Learning statistics
        self.step_count = 0
        self.successful_transitions = 0
        self.total_transitions = 0

    def _create_default_motor_system(self) -> AbstractMotorSystem:
        """Create default motor system for abstract exploration."""
        motor_system = AbstractMotorSystem(f"{self.lm_id}_motor_system")

        # Add default exploration policy
        policy = AbstractMotorPolicy(
            policy_id=f"{self.lm_id}_exploration_policy",
            exploration_probability=0.4,
            step_size_range=(0.05, 0.15)
        )
        motor_system.add_policy(policy)

        return motor_system

    def _extract_concept_from_state(self, state: State) -> Optional[TemporalConceptEmbedding]:
        """Extract concept information from a State object.
        
        Args:
            state: Input state
            
        Returns:
            Extracted concept embedding or None if not found
        """
        if not state.use_state:
            return None

        # Accept both ASM (old) and SASM (new sensorimotor) sensor modules
        if state.sender_type not in ["ASM", "SASM"]:
            return None

        # Extract concept information from non-morphological features
        non_morph = state.non_morphological_features

        if "concept_id" not in non_morph:
            return None

        concept_id = non_morph["concept_id"]
        domain = non_morph.get("domain", self.reasoning_domain)
        temporal_context = non_morph.get("temporal_context", [])

        # Get or create concept using local manager (no global registry)
        concept = self.concept_manager.get_or_create_embedding(
            concept_id=concept_id,
            domain=domain,
            temporal_context=temporal_context
        )

        return concept

    def _extract_reference_frame_from_state(self, state: State) -> Optional[AbstractReferenceFrame]:
        """Extract reference frame information from a State object.
        
        Args:
            state: Input state
            
        Returns:
            Reference frame or None if not found
        """
        if not state.use_state:
            return None

        # Extract reference frame from morphological features
        morph = state.morphological_features

        if "reference_frame_id" not in morph or "concept_domain" not in morph:
            return self.primary_reference_frame

        ref_frame_id = morph["reference_frame_id"]
        domain = morph["concept_domain"]

        try:
            ref_frame = ABSTRACT_FRAME_REGISTRY.get_frame(domain, ref_frame_id)
            return ref_frame
        except KeyError:
            # Fall back to primary reference frame
            return self.primary_reference_frame

    def _transform_state_reference_frame(
            self,
            state: State,
            source_frame: AbstractReferenceFrame,
            target_frame: AbstractReferenceFrame,
    ) -> State:
        """Transform a state from one reference frame to another.
        
        Args:
            state: The state to transform
            source_frame: Source reference frame
            target_frame: Target reference frame
            
        Returns:
            Transformed state
        """
        # Create a copy of the state to avoid modifying the original
        new_state = copy.deepcopy(state)

        # Transform the location
        new_location = source_frame.transform_to(state.location, target_frame)

        # Use only the first 3 dimensions if needed
        if len(new_location) > 3:
            new_location = new_location[:3]
        elif len(new_location) < 3:
            new_location = np.pad(new_location, (0, 3 - len(new_location)))

        new_state.location = new_location

        # Update reference frame information in morphological features
        if "reference_frame_id" in new_state.morphological_features:
            new_state.morphological_features["reference_frame_id"] = target_frame.frame_id

        # Transform pose vectors if present
        if "pose_vectors" in new_state.morphological_features:
            pose_vectors = new_state.morphological_features["pose_vectors"]

            # Create rotation matrix between the frames
            rotation_matrix, _ = align_orthonormal_vectors(
                source_frame.base_dimensions[:3],
                target_frame.base_dimensions[:3],
                as_scipy=False,
            )

            # Apply rotation to pose vectors
            transformed_pose = np.zeros_like(pose_vectors)
            for i in range(pose_vectors.shape[0]):
                transformed_pose[i] = np.dot(rotation_matrix, pose_vectors[i])

            new_state.morphological_features["pose_vectors"] = transformed_pose

        return new_state

    def _apply_inference_rules(self, matched_result: Dict) -> Dict:
        """Apply domain-specific inference rules to extend matched results.
        
        Args:
            matched_result: Result from standard matching
            
        Returns:
            Enhanced result with inferences
        """
        # Implementation depends on the specific domain
        # This is a placeholder for domain-specific logic

        if self.reasoning_domain == "mathematics":
            return self._apply_mathematical_inference_rules(matched_result)
        elif self.reasoning_domain == "philosophy":
            return self._apply_philosophical_inference_rules(matched_result)
        elif self.reasoning_domain == "physics":
            return self._apply_physics_inference_rules(matched_result)

        return matched_result

    def _apply_mathematical_inference_rules(self, matched_result: Dict) -> Dict:
        """Apply mathematics-specific inference rules.
        
        Args:
            matched_result: Result from standard matching
            
        Returns:
            Enhanced result with mathematical inferences
        """
        # Example: If we matched "triangle" and "right angle",
        # we might infer "Pythagorean theorem"
        # This is highly domain-specific
        return matched_result

    def _apply_philosophical_inference_rules(self, matched_result: Dict) -> Dict:
        """Apply philosophy-specific inference rules.
        
        Args:
            matched_result: Result from standard matching
            
        Returns:
            Enhanced result with philosophical inferences
        """
        # Example: If we matched "determinism" and "free will",
        # we might infer "compatibilism" as a potential resolution
        return matched_result

    def _apply_physics_inference_rules(self, matched_result: Dict) -> Dict:
        """Apply physics-specific inference rules.
        
        Args:
            matched_result: Result from standard matching
            
        Returns:
            Enhanced result with physics inferences
        """
        # Example: If we matched "mass" and "acceleration",
        # we might infer "force = mass * acceleration"
        return matched_result

    def matching_step(self, inputs: Dict[str, State]) -> Dict:
        """Match incoming abstract concepts against stored knowledge.
        
        Args:
            inputs: Dictionary mapping input keys to States
            
        Returns:
            Dictionary of outputs from matching process
        """
        # Transform all inputs to primary reference frame if needed
        transformed_inputs = {}
        for key, state in inputs.items():
            if not state.use_state:
                transformed_inputs[key] = state
                continue

            # Extract the source reference frame
            source_frame = self._extract_reference_frame_from_state(state)

            if source_frame is None or source_frame.frame_id == self.primary_reference_frame_id:
                # Already in the right frame or no frame information
                transformed_inputs[key] = state
            else:
                # Transform to primary reference frame
                transformed_inputs[key] = self._transform_state_reference_frame(
                    state, source_frame, self.primary_reference_frame
                )

        # Call the parent's matching implementation
        matched_result = super().matching_step(transformed_inputs)

        # Apply domain-specific inference rules
        enriched_result = self._apply_inference_rules(matched_result)

        return enriched_result

    def exploratory_step(self, inputs: Dict[str, State]) -> Dict:
        """Perform sensorimotor exploratory step in abstract space.

        Args:
            inputs: Dictionary mapping input keys to States

        Returns:
            Dictionary of outputs from exploration process
        """
        self.step_count += 1

        # Process inputs and extract concepts
        current_concepts = []
        processed_inputs = {}

        for key, state in inputs.items():
            processed_inputs[key] = state

            if not state.use_state:
                continue

            concept = self._extract_concept_from_state(state)
            if concept is not None:
                current_concepts.append(concept)

                # Update sequences
                self.state_sequence.append(state)
                self.concept_sequence.append(concept.concept_id)

        if not current_concepts:
            # No concepts extracted, perform random exploration
            self._perform_random_exploration()
            return super().exploratory_step(processed_inputs)

        # Use most salient concept as current
        primary_concept = current_concepts[0]
        self.current_concept = primary_concept.concept_id

        # Generate motor action for exploration
        motor_action = self._generate_exploration_action(primary_concept)

        # Update position and learn sensorimotor associations
        if motor_action is not None:
            self._execute_motor_action(motor_action)
            self._learn_sensorimotor_association(primary_concept, motor_action)

        # Call parent's exploratory implementation
        return super().exploratory_step(processed_inputs)

    def _perform_random_exploration(self) -> None:
        """Perform random exploration when no concepts are available."""
        if self.motor_system:
            # Generate random exploration action
            random_action = self.motor_system.propose_action(
                current_concept="unknown",
                current_position=self.current_position
            )
            if random_action:
                self._execute_motor_action(random_action)

    def _generate_exploration_action(self, concept: TemporalConceptEmbedding) -> Optional[AbstractMotorAction]:
        """Generate motor action for exploring around a concept."""
        if not self.motor_system:
            return None

        # Get current position in reference frame
        concept_position = self.reference_frame.get_concept_position(concept.concept_id)
        if concept_position is None:
            concept_position = self.current_position

        # Get nearby concepts for context
        nearby_concepts = self.reference_frame.get_nearby_concepts(concept.concept_id)

        return self.motor_system.propose_action(
            current_concept=concept.concept_id,
            current_position=concept_position,
            nearby_concepts=nearby_concepts
        )

    def _execute_motor_action(self, motor_action: AbstractMotorAction) -> None:
        """Execute a motor action and update position."""
        self.current_position += motor_action.displacement
        self.motor_sequence.append(motor_action)
        self.last_motor_action = motor_action

    def _learn_sensorimotor_association(
        self,
        concept: TemporalConceptEmbedding,
        motor_action: AbstractMotorAction
    ) -> None:
        """Learn association between concepts and motor actions."""
        if len(self.concept_sequence) < 2:
            return

        # Get previous concept
        prev_concept_id = self.concept_sequence[-2]
        current_concept_id = concept.concept_id

        # Add experience to reference frame
        temporal_context = list(self.concept_sequence)[-5:]  # Last 5 concepts
        self.reference_frame.add_sensorimotor_experience(
            source_concept=prev_concept_id,
            target_concept=current_concept_id,
            motor_action=motor_action.displacement,
            temporal_context=temporal_context
        )

        # Update motor system with success/failure
        success = self._evaluate_transition_success(prev_concept_id, current_concept_id)
        self.motor_system.update_from_experience(
            action=motor_action,
            success=success,
            source_concept=prev_concept_id,
            target_concept=current_concept_id
        )

        # Update statistics
        self.total_transitions += 1
        if success:
            self.successful_transitions += 1

    def _evaluate_transition_success(self, source_concept: str, target_concept: str) -> bool:
        """Evaluate whether a concept transition was successful."""
        # Simple heuristic: success if we moved to a related concept
        source_embedding = self.concept_manager.get_embedding(source_concept)
        target_embedding = self.concept_manager.get_embedding(target_concept)

        if source_embedding is None or target_embedding is None:
            return False

        # Consider transition successful if concepts are moderately similar
        similarity = source_embedding.similarity(target_embedding)
        return 0.3 <= similarity <= 0.8  # Not too similar, not too different

    def propose_goal_state(self) -> Optional[GoalState]:
        """Propose a goal state based on sensorimotor abstract reasoning.

        Returns:
            Goal state or None if no goal state is proposed
        """
        # Get the most likely hypothesis
        mlh = self.current_mlh

        if mlh is None or "object_id" not in mlh:
            return None

        # Predict next concept based on temporal patterns
        next_concept = None
        if self.current_concept:
            prediction = self.concept_manager.predict_next_concept(
                self.current_concept,
                self.reasoning_domain
            )
            if prediction:
                next_concept = prediction[0]

        # Get target position in learned reference frame
        target_position = None
        if next_concept:
            target_position = self.reference_frame.get_concept_position(next_concept)

        # Create goal state with sensorimotor information
        goal_state = GoalState(
            location=target_position[:3] if target_position is not None else self.current_position,
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": self.reference_frame.is_stable,
                "reference_frame_stability": self.reference_frame.stability_score,
            },
            non_morphological_features={
                "concept_id": mlh["object_id"],
                "domain": self.reasoning_domain,
                "target_concept": next_concept,
                "current_concept": self.current_concept,
                "temporal_context": list(self.concept_sequence)[-3:],
                "motor_exploration": True,
                "reference_frame_id": self.reference_frame.frame_id,
            },
            confidence=mlh.get("evidence", 0.0) * self.reference_frame.stability_score,
            use_state=True,
            sender_id=self.lm_id,
            sender_type="LM",
            goal_tolerances=None,
            info={
                "sensorimotor_reasoning": True,
                "step_count": self.step_count,
                "successful_transitions": self.successful_transitions,
                "total_transitions": self.total_transitions,
            },
        )

        return goal_state


class SensorimotorPhilosophicalReasoningLM(SensorimotorAbstractReasoningLM):
    """Sensorimotor learning module specialized for philosophical reasoning."""

    def __init__(
            self,
            lm_id: str,
            philosophical_school: str = "general",
            temporal_window: int = 15,
            embedding_dim: int = 50,
            **kwargs,
    ):
        """Initialize a sensorimotor philosophical reasoning module.

        Args:
            lm_id: Unique identifier for this learning module
            philosophical_school: Specific school of philosophy to use
            temporal_window: Size of temporal sequence buffer
            embedding_dim: Dimensionality of concept embeddings
            **kwargs: Additional arguments for SensorimotorAbstractReasoningLM
        """
        super().__init__(
            lm_id=lm_id,
            reasoning_domain="philosophy",
            temporal_window=temporal_window,
            embedding_dim=embedding_dim,
            **kwargs,
        )

        self.philosophical_school = philosophical_school

        # School-specific biases for concept similarity
        self.school_concept_biases = self._initialize_school_biases()

    def _initialize_school_biases(self) -> Dict[str, float]:
        """Initialize school-specific concept biases.

        Returns:
            Dictionary of concept biases for the philosophical school
        """
        # Biologically plausible biases instead of complex inference rules
        if self.philosophical_school == "utilitarian":
            return {
                "happiness": 1.2,
                "pleasure": 1.1,
                "suffering": -0.5,
                "pain": -0.4,
                "utility": 1.0,
            }
        elif self.philosophical_school == "kantian":
            return {
                "duty": 1.3,
                "categorical": 1.2,
                "universal": 1.1,
                "maxim": 1.0,
                "autonomy": 1.1,
            }
        elif self.philosophical_school == "virtue_ethics":
            return {
                "virtue": 1.2,
                "character": 1.1,
                "excellence": 1.0,
                "flourishing": 1.1,
                "wisdom": 1.0,
            }

        # Default general biases
        return {}

    def _evaluate_transition_success(self, source_concept: str, target_concept: str) -> bool:
        """Evaluate transition success with philosophical school biases."""
        # Get base success evaluation
        base_success = super()._evaluate_transition_success(source_concept, target_concept)

        # Apply school-specific biases
        bias_factor = 1.0
        for concept_key, bias in self.school_concept_biases.items():
            if concept_key in target_concept.lower():
                bias_factor *= bias
                break

        # Adjust success probability based on bias
        if bias_factor > 1.0:
            return True  # Favor transitions to school-relevant concepts
        elif bias_factor < 0:
            return False  # Avoid negatively biased concepts

        return base_success


class SensorimotorMathematicalReasoningLM(SensorimotorAbstractReasoningLM):
    """Sensorimotor learning module specialized for mathematical reasoning."""

    def __init__(
            self,
            lm_id: str,
            math_domain: str = "general",
            temporal_window: int = 15,
            embedding_dim: int = 50,
            **kwargs,
    ):
        """Initialize a sensorimotor mathematical reasoning module.

        Args:
            lm_id: Unique identifier for this learning module
            math_domain: Specific branch of mathematics
            temporal_window: Size of temporal sequence buffer
            embedding_dim: Dimensionality of concept embeddings
            **kwargs: Additional arguments for SensorimotorAbstractReasoningLM
        """
        super().__init__(
            lm_id=lm_id,
            reasoning_domain="mathematics",
            temporal_window=temporal_window,
            embedding_dim=embedding_dim,
            **kwargs,
        )

        self.math_domain = math_domain

        # Domain-specific concept relationships
        self.mathematical_concept_relationships = self._initialize_math_relationships()

    def _initialize_math_relationships(self) -> Dict[str, List[str]]:
        """Initialize mathematical concept relationships for the domain."""
        if self.math_domain == "geometry":
            return {
                "triangle": ["polygon", "shape", "three_sides", "angles"],
                "circle": ["shape", "round", "radius", "diameter"],
                "square": ["polygon", "rectangle", "four_sides", "equal_sides"],
                "angle": ["measurement", "degrees", "radians"],
            }
        elif self.math_domain == "algebra":
            return {
                "equation": ["expression", "equality", "solve", "variable"],
                "variable": ["unknown", "symbol", "x", "y"],
                "polynomial": ["expression", "terms", "degree", "coefficient"],
                "function": ["mapping", "input", "output", "domain"],
            }
        elif self.math_domain == "calculus":
            return {
                "derivative": ["rate", "change", "slope", "limit"],
                "integral": ["area", "accumulation", "antiderivative"],
                "limit": ["approach", "infinity", "continuous"],
                "function": ["continuous", "differentiable", "domain"],
            }

        return {}

    def _evaluate_transition_success(self, source_concept: str, target_concept: str) -> bool:
        """Evaluate transition success with mathematical domain knowledge."""
        # Check if concepts are mathematically related
        for concept, related_concepts in self.mathematical_concept_relationships.items():
            if concept in source_concept.lower():
                # If target is related to source, consider it successful
                for related in related_concepts:
                    if related in target_concept.lower():
                        return True

        # Fall back to base evaluation
        return super()._evaluate_transition_success(source_concept, target_concept)


class SensorimotorPhysicsReasoningLM(SensorimotorAbstractReasoningLM):
    """Sensorimotor learning module specialized for physics reasoning."""

    def __init__(
            self,
            lm_id: str,
            physics_framework: str = "classical",
            temporal_window: int = 15,
            embedding_dim: int = 50,
            **kwargs,
    ):
        """Initialize a sensorimotor physics reasoning module.

        Args:
            lm_id: Unique identifier for this learning module
            physics_framework: Theoretical framework (classical, quantum, relativistic)
            temporal_window: Size of temporal sequence buffer
            embedding_dim: Dimensionality of concept embeddings
            **kwargs: Additional arguments for SensorimotorAbstractReasoningLM
        """
        super().__init__(
            lm_id=lm_id,
            reasoning_domain="physics",
            temporal_window=temporal_window,
            embedding_dim=embedding_dim,
            **kwargs,
        )

        self.physics_framework = physics_framework

        # Framework-specific concept relationships
        self.physics_concept_relationships = self._initialize_physics_relationships()

    def _initialize_physics_relationships(self) -> Dict[str, List[str]]:
        """Initialize physics concept relationships for the framework."""
        if self.physics_framework == "classical":
            return {
                "force": ["mass", "acceleration", "newton", "motion"],
                "mass": ["weight", "inertia", "matter", "kilogram"],
                "velocity": ["speed", "direction", "motion", "time"],
                "energy": ["work", "power", "kinetic", "potential"],
                "gravity": ["force", "mass", "distance", "acceleration"],
                "momentum": ["mass", "velocity", "conservation", "collision"],
            }
        elif self.physics_framework == "quantum":
            return {
                "photon": ["light", "particle", "wave", "energy"],
                "electron": ["charge", "particle", "orbital", "spin"],
                "wave": ["frequency", "amplitude", "interference", "particle"],
                "energy": ["quantum", "discrete", "photon", "level"],
                "uncertainty": ["position", "momentum", "measurement", "principle"],
            }
        elif self.physics_framework == "relativistic":
            return {
                "spacetime": ["space", "time", "curvature", "gravity"],
                "mass": ["energy", "equivalence", "rest", "relativistic"],
                "light": ["speed", "constant", "photon", "spacetime"],
                "gravity": ["curvature", "spacetime", "mass", "acceleration"],
            }

        return {}

    def _evaluate_transition_success(self, source_concept: str, target_concept: str) -> bool:
        """Evaluate transition success with physics domain knowledge."""
        # Check if concepts are physically related
        for concept, related_concepts in self.physics_concept_relationships.items():
            if concept in source_concept.lower():
                # If target is related to source, consider it successful
                for related in related_concepts:
                    if related in target_concept.lower():
                        return True

        # Fall back to base evaluation
        return super()._evaluate_transition_success(source_concept, target_concept)
