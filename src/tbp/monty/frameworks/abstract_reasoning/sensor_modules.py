# sensor_modules.py

import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import re
from collections import deque

from tbp.monty.frameworks.models.monty_base import SensorModuleBase
from tbp.monty.frameworks.models.states import State
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
)


class SensorimotorAbstractSensorModule(SensorModuleBase):
    """Sensorimotor-based sensor module for abstract domains.

    Processes temporal sequences of abstract inputs with motor actions,
    implementing TBT principles of sensorimotor learning in abstract spaces.
    """

    def __init__(
            self,
            sensor_module_id: str,
            domain: str,
            temporal_window: int = 10,
            motor_system: Optional[AbstractMotorSystem] = None,
            embedding_dim: int = 50,
    ):
        """Initialize the sensorimotor abstract sensor module.

        Args:
            sensor_module_id: Unique identifier for this sensor module
            domain: Abstract domain this sensor processes
            temporal_window: Size of temporal sequence buffer
            motor_system: Motor system for abstract actions
            embedding_dim: Dimensionality of concept embeddings
        """
        super().__init__(sensor_module_id)
        self.domain = domain
        self.temporal_window = temporal_window
        self.embedding_dim = embedding_dim

        # Local components - no global dependencies
        self.concept_manager = LocalConceptEmbeddingManager(
            module_id=sensor_module_id,
            embedding_dim=embedding_dim,
            temporal_window=temporal_window
        )

        self.reference_frame = LearnedAbstractReferenceFrame(
            frame_id=f"{sensor_module_id}_frame",
            domain=domain,
            max_dimensions=min(10, embedding_dim)
        )

        # Motor system for abstract actions
        self.motor_system = motor_system

        # Temporal processing components
        self.input_sequence = deque(maxlen=temporal_window)
        self.concept_sequence = deque(maxlen=temporal_window)
        self.motor_sequence = deque(maxlen=temporal_window)
        self.current_position = np.zeros(3)  # Current position in abstract space

        # Sensorimotor learning state
        self.current_concept = None
        self.last_motor_action = None
        self.step_count = 0

    def _parse_temporal_input(self, data: Any) -> Dict:
        """Parse input data in temporal context.

        Args:
            data: Input data (format depends on specific sensor implementation)

        Returns:
            Parsed representation including temporal context
        """
        raise NotImplementedError(
            "Sensorimotor sensor modules must implement _parse_temporal_input method"
        )

    def _extract_concepts_from_sequence(
        self,
        parsed_data: Dict,
        temporal_context: List[str]
    ) -> List[TemporalConceptEmbedding]:
        """Extract concepts from parsed data using temporal context.

        Args:
            parsed_data: Structured representation of the input
            temporal_context: Recent sequence of concepts

        Returns:
            List of temporal concept embeddings
        """
        raise NotImplementedError(
            "Sensorimotor sensor modules must implement _extract_concepts_from_sequence method"
        )

    def _generate_motor_action(
        self,
        current_concept: str,
        goal_concept: Optional[str] = None
    ) -> Optional[AbstractMotorAction]:
        """Generate motor action for exploration or goal-directed movement.

        Args:
            current_concept: Current concept being processed
            goal_concept: Target concept (if any)

        Returns:
            Motor action to take in abstract space
        """
        if self.motor_system is None:
            return None

        # Get current position in reference frame
        current_position = self.reference_frame.get_concept_position(current_concept)
        if current_position is None:
            current_position = self.current_position

        # Get goal position if available
        goal_position = None
        if goal_concept:
            goal_position = self.reference_frame.get_concept_position(goal_concept)

        # Get nearby concepts for context
        nearby_concepts = self.reference_frame.get_nearby_concepts(current_concept)

        return self.motor_system.propose_action(
            current_concept=current_concept,
            current_position=current_position,
            goal_concept=goal_concept,
            goal_position=goal_position,
            nearby_concepts=nearby_concepts
        )

    def _create_sensorimotor_morphology(
            self,
            concept: TemporalConceptEmbedding,
            motor_action: Optional[AbstractMotorAction] = None,
    ) -> Dict:
        """Create morphological features based on sensorimotor experience.

        Args:
            concept: Temporal concept embedding
            motor_action: Recent motor action (if any)

        Returns:
            Dictionary of morphological features
        """
        # Create pose vectors based on learned reference frame
        pose_vectors = np.eye(3)  # Default orthonormal basis

        # If we have learned base dimensions, use them
        if len(self.reference_frame.base_dimensions) > 0:
            for i in range(min(3, len(self.reference_frame.base_dimensions))):
                base_dim = self.reference_frame.base_dimensions[i]
                # Ensure 3D by padding or truncating
                if len(base_dim) >= 3:
                    pose_vectors[i] = base_dim[:3]
                else:
                    pose_vectors[i][:len(base_dim)] = base_dim

                # Normalize
                norm = np.linalg.norm(pose_vectors[i])
                if norm > 1e-6:
                    pose_vectors[i] = pose_vectors[i] / norm

        # Incorporate motor action information if available
        motor_influence = 0.0
        if motor_action is not None:
            motor_influence = np.linalg.norm(motor_action.displacement)

        return {
            "pose_vectors": pose_vectors,
            "pose_fully_defined": len(self.reference_frame.base_dimensions) >= 3,
            "reference_frame_id": self.reference_frame.frame_id,
            "concept_domain": concept.domain,
            "motor_influence": motor_influence,
            "temporal_context_size": len(self.concept_sequence),
        }

    def step(self, data: Any) -> State:
        """Process abstract domain input using sensorimotor learning principles.

        Args:
            data: Input data (format depends on specific sensor implementation)

        Returns:
            State representation incorporating temporal and motor information
        """
        self.step_count += 1

        # Parse input in temporal context
        parsed_data = self._parse_temporal_input(data)

        # Get current temporal context
        temporal_context = list(self.concept_sequence)

        # Extract concepts using temporal context
        concepts = self._extract_concepts_from_sequence(parsed_data, temporal_context)

        if not concepts:
            logging.warning(f"No concepts extracted from input by {self.sensor_module_id}")
            return self._create_empty_state(data)

        # Use the most salient concept
        primary_concept = concepts[0]
        self.current_concept = primary_concept.concept_id

        # Generate motor action for next step
        motor_action = self._generate_motor_action(self.current_concept)

        # Update position based on motor action
        if motor_action is not None:
            self.current_position += motor_action.displacement
            self.motor_sequence.append(motor_action)
            self.last_motor_action = motor_action

        # Learn sensorimotor associations in reference frame
        if len(self.concept_sequence) > 0 and self.last_motor_action is not None:
            prev_concept = self.concept_sequence[-1]
            self.reference_frame.add_sensorimotor_experience(
                source_concept=prev_concept,
                target_concept=self.current_concept,
                motor_action=self.last_motor_action.displacement,
                temporal_context=temporal_context
            )

        # Update sequences
        self.input_sequence.append(data)
        self.concept_sequence.append(self.current_concept)

        # Get position in learned reference frame
        concept_position = self.reference_frame.get_concept_position(self.current_concept)
        if concept_position is not None:
            # Use first 3 dimensions for location
            abstract_location = concept_position[:3] if len(concept_position) >= 3 else np.pad(concept_position, (0, 3 - len(concept_position)))
        else:
            abstract_location = self.current_position

        # Create morphological features based on sensorimotor experience
        morphological_features = self._create_sensorimotor_morphology(
            primary_concept, motor_action
        )

        # Non-morphological features include temporal and motor information
        non_morphological_features = {
            "concept_id": primary_concept.concept_id,
            "domain": primary_concept.domain,
            "related_concepts": [c.concept_id for c in concepts[1:]],
            "temporal_context": temporal_context[-5:],  # Last 5 concepts
            "motor_action_type": motor_action.action_type if motor_action else None,
            "step_count": self.step_count,
            "reference_frame_stability": self.reference_frame.stability_score,
        }

        # Confidence based on reference frame stability and concept familiarity
        confidence = (
            0.5 * self.reference_frame.stability_score +
            0.3 * min(1.0, primary_concept.activation_count / 10.0) +
            0.2 * (1.0 if len(temporal_context) > 0 else 0.0)
        )

        return State(
            location=abstract_location,
            morphological_features=morphological_features,
            non_morphological_features=non_morphological_features,
            confidence=confidence,
            use_state=True,
            sender_id=self.sensor_module_id,
            sender_type="SASM",  # Sensorimotor Abstract Sensor Module
        )

    def _create_empty_state(self, data: Any) -> State:
        """Create empty state when no concepts are extracted."""
        return State(
            location=self.current_position,
            morphological_features={
                "pose_vectors": np.eye(3),
                "pose_fully_defined": False,
                "motor_influence": 0.0,
                "temporal_context_size": len(self.concept_sequence),
            },
            non_morphological_features={
                "raw_input": data,
                "step_count": self.step_count,
                "temporal_context": list(self.concept_sequence)[-5:],
            },
            confidence=0.0,
            use_state=False,
            sender_id=self.sensor_module_id,
            sender_type="SASM",
        )


class SensorimotorTextSensorModule(SensorimotorAbstractSensorModule):
    """Processes textual descriptions using sensorimotor learning principles."""

    def __init__(
            self,
            sensor_module_id: str,
            domain: str,
            temporal_window: int = 10,
            motor_system: Optional[AbstractMotorSystem] = None,
            embedding_dim: int = 50,
    ):
        """Initialize the sensorimotor text sensor module.

        Args:
            sensor_module_id: Unique identifier for this sensor module
            domain: Abstract domain this sensor processes
            temporal_window: Size of temporal sequence buffer
            motor_system: Motor system for abstract actions
            embedding_dim: Dimensionality of concept embeddings
        """
        super().__init__(sensor_module_id, domain, temporal_window, motor_system, embedding_dim)

    def _parse_temporal_input(self, data: str) -> Dict:
        """Parse textual input in temporal context.

        Args:
            data: Text input

        Returns:
            Parsed representation including temporal context
        """
        # Simple parsing - extract sentences and words
        sentences = re.split(r'[.!?]', data)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Extract words and filter for potential concepts
        words = re.findall(r'\b\w+\b', data.lower())
        potential_concepts = [w for w in words if len(w) > 3]

        # Analyze temporal patterns with previous inputs
        temporal_similarity = 0.0
        if len(self.input_sequence) > 0:
            prev_data = self.input_sequence[-1]
            if isinstance(prev_data, str):
                # Simple word overlap similarity
                prev_words = set(re.findall(r'\b\w+\b', prev_data.lower()))
                curr_words = set(words)
                if prev_words or curr_words:
                    temporal_similarity = len(prev_words.intersection(curr_words)) / len(prev_words.union(curr_words))

        return {
            "text": data,
            "sentences": sentences,
            "potential_concepts": potential_concepts,
            "temporal_similarity": temporal_similarity,
            "word_count": len(words),
            "sentence_count": len(sentences),
        }

    def _extract_concepts_from_sequence(
        self,
        parsed_data: Dict,
        temporal_context: List[str]
    ) -> List[TemporalConceptEmbedding]:
        """Extract concepts from text using temporal context.

        Args:
            parsed_data: Structured representation of the text input
            temporal_context: Recent sequence of concepts

        Returns:
            List of temporal concept embeddings
        """
        concepts = []
        text = parsed_data["text"]

        # Extract main concept from text
        # Use hash for consistent concept ID generation
        main_concept_id = f"text_concept_{hash(text) % 10000}"

        main_concept = self.concept_manager.get_or_create_embedding(
            concept_id=main_concept_id,
            domain=self.domain,
            description=text,
            temporal_context=temporal_context
        )
        concepts.append(main_concept)

        # Extract sub-concepts from sentences if multiple sentences
        sentences = parsed_data["sentences"]
        if len(sentences) > 1:
            for i, sentence in enumerate(sentences[:3]):  # Limit to first 3 sentences
                sentence_id = f"sentence_{hash(sentence) % 10000}"
                sentence_concept = self.concept_manager.get_or_create_embedding(
                    concept_id=sentence_id,
                    domain=self.domain,
                    description=sentence,
                    temporal_context=temporal_context + [main_concept_id]
                )
                concepts.append(sentence_concept)

        return concepts


class SensorimotorSymbolicSensorModule(SensorimotorAbstractSensorModule):
    """Processes symbolic expressions using sensorimotor learning principles."""

    def __init__(
            self,
            sensor_module_id: str,
            domain: str,
            temporal_window: int = 10,
            motor_system: Optional[AbstractMotorSystem] = None,
            embedding_dim: int = 50,
            symbol_system: str = "general",
    ):
        """Initialize the sensorimotor symbolic sensor module.

        Args:
            sensor_module_id: Unique identifier for this sensor module
            domain: Abstract domain this sensor processes
            temporal_window: Size of temporal sequence buffer
            motor_system: Motor system for abstract actions
            embedding_dim: Dimensionality of concept embeddings
            symbol_system: Type of symbolic system to process
        """
        super().__init__(sensor_module_id, domain, temporal_window, motor_system, embedding_dim)
        self.symbol_system = symbol_system

    def _parse_temporal_input(self, data: str) -> Dict:
        """Parse symbolic expressions in temporal context.

        Args:
            data: Symbolic expression as text

        Returns:
            Parsed representation including temporal patterns
        """
        # Basic symbol and operator extraction
        symbols = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', data)
        operators = re.findall(r'[\+\-\*/\^\(\)=<>]', data)
        numbers = re.findall(r'\d+\.?\d*', data)

        # Analyze structural complexity
        complexity_score = len(symbols) + len(operators) + len(numbers)

        # Check for temporal patterns with previous expressions
        structural_similarity = 0.0
        if len(self.input_sequence) > 0:
            prev_data = self.input_sequence[-1]
            if isinstance(prev_data, str):
                prev_symbols = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', prev_data))
                curr_symbols = set(symbols)
                if prev_symbols or curr_symbols:
                    structural_similarity = len(prev_symbols.intersection(curr_symbols)) / len(prev_symbols.union(curr_symbols))

        return {
            "expression": data,
            "symbols": symbols,
            "operators": operators,
            "numbers": numbers,
            "symbol_system": self.symbol_system,
            "complexity_score": complexity_score,
            "structural_similarity": structural_similarity,
        }

    def _extract_concepts_from_sequence(
        self,
        parsed_data: Dict,
        temporal_context: List[str]
    ) -> List[TemporalConceptEmbedding]:
        """Extract concepts from symbolic expressions using temporal context.

        Args:
            parsed_data: Structured representation of the expression
            temporal_context: Recent sequence of concepts

        Returns:
            List of temporal concept embeddings
        """
        concepts = []
        expression = parsed_data["expression"]

        # Main expression concept
        expr_concept_id = f"expr_{hash(expression) % 10000}"
        expr_concept = self.concept_manager.get_or_create_embedding(
            concept_id=expr_concept_id,
            domain=self.domain,
            description=expression,
            temporal_context=temporal_context
        )
        concepts.append(expr_concept)

        # Individual symbol concepts if expression is complex
        if parsed_data["complexity_score"] > 5:
            for symbol in parsed_data["symbols"][:3]:  # Limit to first 3 symbols
                symbol_id = f"symbol_{symbol}_{self.symbol_system}"
                symbol_concept = self.concept_manager.get_or_create_embedding(
                    concept_id=symbol_id,
                    domain=self.domain,
                    description=f"Symbol {symbol} in {self.symbol_system}",
                    temporal_context=temporal_context + [expr_concept_id]
                )
                concepts.append(symbol_concept)

        return concepts
