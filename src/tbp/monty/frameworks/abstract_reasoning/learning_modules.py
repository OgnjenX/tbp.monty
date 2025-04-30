import copy
import logging
from typing import Dict, Optional

import numpy as np

from tbp.monty.frameworks.abstract_reasoning.abstract_reference_frames import (
    AbstractReferenceFrame,
    ABSTRACT_FRAME_REGISTRY,
)
from tbp.monty.frameworks.abstract_reasoning.concept_embeddings import (
    ConceptEmbedding,
    CONCEPT_EMBEDDING_REGISTRY,
)
from tbp.monty.frameworks.models.evidence_matching import EvidenceGraphLM
from tbp.monty.frameworks.models.states import State, GoalState
from tbp.monty.frameworks.utils.spatial_arithmetics import align_orthonormal_vectors


class AbstractReasoningLM(EvidenceGraphLM):
    """Learning module for abstract reasoning domains.
    
    Extends EvidenceGraphLM with capabilities for handling abstract concepts
    and reference frames.
    """

    def __init__(
            self,
            lm_id: str,
            reasoning_domain: str,
            primary_reference_frame: str,
            domain_inference_rules: Optional[Dict] = None,
            **kwargs,
    ):
        """Initialize the abstract reasoning learning module.
        
        Args:
            lm_id: Unique identifier for this learning module
            reasoning_domain: Abstract domain this module handles
            primary_reference_frame: Default reference frame ID to use
            domain_inference_rules: Domain-specific inference rules
            **kwargs: Additional arguments for EvidenceGraphLM
        """
        super().__init__(lm_id=lm_id, **kwargs)

        self.lm_id = None
        self.reasoning_domain = reasoning_domain
        self.primary_reference_frame_id = primary_reference_frame
        self.inference_rules = domain_inference_rules or {}

        # Get the primary reference frame
        try:
            self.primary_reference_frame = ABSTRACT_FRAME_REGISTRY.get_frame(
                reasoning_domain, primary_reference_frame
            )
        except KeyError:
            logging.error(
                f"Primary reference frame {primary_reference_frame} not found "
                f"for domain {reasoning_domain}. Abstract reasoning module may not function correctly."
            )
            self.primary_reference_frame = None

        # Track concepts seen in this session
        self.observed_concepts = {}  # concept_id -> ConceptEmbedding

    def _extract_concept_from_state(self, state: State) -> Optional[ConceptEmbedding]:
        """Extract concept information from a State object.
        
        Args:
            state: Input state
            
        Returns:
            Extracted concept embedding or None if not found
        """
        if not state.use_state:
            return None

        if state.sender_type != "ASM":
            # Not from an abstract sensor module
            return None

        # Extract concept information from non-morphological features
        non_morph = state.non_morphological_features

        if "concept_id" not in non_morph or "domain" not in non_morph:
            return None

        concept_id = non_morph["concept_id"]
        domain = non_morph["domain"]

        # Check if we've already seen this concept
        if concept_id in self.observed_concepts:
            return self.observed_concepts[concept_id]

        # Get the concept from the registry
        concept = CONCEPT_EMBEDDING_REGISTRY.get_embedding(domain, concept_id)

        if concept is not None:
            # Store for future reference
            self.observed_concepts[concept_id] = concept

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
        """Explore abstract concept space based on inputs.
        
        Args:
            inputs: Dictionary mapping input keys to States
            
        Returns:
            Dictionary of outputs from exploration process
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

        # Extract concepts from inputs
        for key, state in transformed_inputs.items():
            concept = self._extract_concept_from_state(state)
            if concept is not None:
                # Store concept for future reference
                self.observed_concepts[concept.concept_id] = concept

        # Call the parent's exploratory implementation
        return super().exploratory_step(transformed_inputs)

    def propose_goal_state(self) -> Optional[GoalState]:
        """Propose a goal state based on abstract reasoning.
        
        Returns:
            Goal state or None if no goal state is proposed
        """
        # Get the most likely hypothesis
        mlh = self.current_mlh

        if mlh is None or "object_id" not in mlh:
            return None

        # Create a goal state
        goal_state = GoalState(
            location=None,  # Abstract goal states often don't have physical locations
            morphological_features=None,
            non_morphological_features={
                "concept_id": mlh["object_id"],
                "domain": self.reasoning_domain,
                "abstract_goal": True,
                "reference_frame_id": self.primary_reference_frame_id,
            },
            confidence=mlh.get("evidence", 0.0),
            use_state=True,
            sender_id=self.lm_id,
            sender_type="LM",
            goal_tolerances=None,
            info={"abstract_reasoning": True},
        )

        return goal_state


class PhilosophicalReasoningLM(AbstractReasoningLM):
    """Learning module specialized for philosophical reasoning."""

    def __init__(
            self,
            lm_id: str,
            primary_reference_frame: str,
            philosophical_school: str = "general",
            **kwargs,
    ):
        """Initialize a philosophical reasoning module.
        
        Args:
            lm_id: Unique identifier for this learning module
            primary_reference_frame: Default reference frame ID to use
            philosophical_school: Specific school of philosophy to use
            **kwargs: Additional arguments for AbstractReasoningLM
        """
        super().__init__(
            lm_id=lm_id,
            reasoning_domain="philosophy",
            primary_reference_frame=primary_reference_frame,
            **kwargs,
        )

        self.philosophical_school = philosophical_school

        # School-specific inference rules
        self.school_inference_rules = self._initialize_school_inference_rules()

    def _initialize_school_inference_rules(self) -> Dict:
        """Initialize school-specific inference rules.
        
        Returns:
            Dictionary of inference rules
        """
        # These would be more complex in a real implementation
        if self.philosophical_school == "utilitarian":
            return {
                "happiness": lambda x: x * 1.2,
                "suffering": lambda x: x * -1.5,
            }
        elif self.philosophical_school == "kantian":
            return {
                "duty": lambda x: x * 1.5,
                "universality": lambda x: x * 1.3,
            }

        # Default general rules
        return {}

    def _apply_philosophical_inference_rules(self, matched_result: Dict) -> Dict:
        """Apply philosophy-specific inference rules with school specialization.
        
        Args:
            matched_result: Result from standard matching
            
        Returns:
            Enhanced result with philosophical inferences
        """
        # Apply general philosophical inference rules
        result = super()._apply_philosophical_inference_rules(matched_result)

        # Apply school-specific rules
        if "object_id" in result and result["object_id"] in self.school_inference_rules:
            rule = self.school_inference_rules[result["object_id"]]
            if "evidence" in result:
                result["evidence"] = rule(result["evidence"])

        return result


class MathematicalReasoningLM(AbstractReasoningLM):
    """Learning module specialized for mathematical reasoning."""

    def __init__(
            self,
            lm_id: str,
            primary_reference_frame: str,
            math_domain: str = "general",
            **kwargs,
    ):
        """Initialize a mathematical reasoning module.
        
        Args:
            lm_id: Unique identifier for this learning module
            primary_reference_frame: Default reference frame ID to use
            math_domain: Specific branch of mathematics
            **kwargs: Additional arguments for AbstractReasoningLM
        """
        super().__init__(
            lm_id=lm_id,
            reasoning_domain="mathematics",
            primary_reference_frame=primary_reference_frame,
            **kwargs,
        )

        self.math_domain = math_domain

        # Domain-specific mathematical structures
        self.mathematical_structures = {}

    def _apply_mathematical_inference_rules(self, matched_result: Dict) -> Dict:
        """Apply mathematics-specific inference rules.
        
        Args:
            matched_result: Result from standard matching
            
        Returns:
            Enhanced result with mathematical inferences
        """
        # Apply general mathematical inference rules
        result = super()._apply_mathematical_inference_rules(matched_result)

        # Apply domain-specific rules
        if self.math_domain == "geometry" and "object_id" in result:
            # Example: Infer properties of geometric objects
            if result["object_id"] == "triangle":
                result["inferred_properties"] = {
                    "angles_sum": 180,
                    "sides": 3,
                }

        return result


class PhysicsReasoningLM(AbstractReasoningLM):
    """Learning module specialized for physics reasoning."""

    def __init__(
            self,
            lm_id: str,
            primary_reference_frame: str,
            physics_framework: str = "classical",
            **kwargs,
    ):
        """Initialize a physics reasoning module.
        
        Args:
            lm_id: Unique identifier for this learning module
            primary_reference_frame: Default reference frame ID to use
            physics_framework: Theoretical framework (classical, quantum, relativistic)
            **kwargs: Additional arguments for AbstractReasoningLM
        """
        super().__init__(
            lm_id=lm_id,
            reasoning_domain="physics",
            primary_reference_frame=primary_reference_frame,
            **kwargs,
        )

        self.physics_framework = physics_framework

        # Framework-specific constants and equations
        self.physical_constants = self._initialize_physical_constants()
        self.physical_equations = self._initialize_physical_equations()

    def _initialize_physical_constants(self) -> Dict:
        """Initialize physical constants for the current framework.
        
        Returns:
            Dictionary of physical constants
        """
        if self.physics_framework == "classical":
            return {
                "G": 6.67430e-11,  # Gravitational constant
                "c": 299792458,  # Speed of light
            }
        elif self.physics_framework == "quantum":
            return {
                "h": 6.62607015e-34,  # Planck constant
                "ℏ": 1.0545718e-34,  # Reduced Planck constant
            }

        return {}

    def _initialize_physical_equations(self) -> Dict:
        """Initialize physical equations for the current framework.
        
        Returns:
            Dictionary of physical equations
        """
        # Just placeholders - would be more complex in a real implementation
        if self.physics_framework == "classical":
            return {
                "F = ma": lambda m, a: m * a,
                "E = mc²": lambda m: m * self.physical_constants["c"] ** 2,
            }

        return {}

    def _apply_physics_inference_rules(self, matched_result: Dict) -> Dict:
        """Apply physics-specific inference rules.
        
        Args:
            matched_result: Result from standard matching
            
        Returns:
            Enhanced result with physics inferences
        """
        # Apply general physics inference rules
        result = super()._apply_physics_inference_rules(matched_result)

        # Apply framework-specific rules
        if "object_id" in result:
            concept = result["object_id"]

            # Example: If we detected "mass" and "distance", infer gravitational force
            if concept == "mass" and "related_concepts" in result.get("non_morphological_features", {}):
                related = result["non_morphological_features"]["related_concepts"]
                if "distance" in related and self.physics_framework == "classical":
                    result["inferred_concepts"] = ["gravitational_force"]

        return result
