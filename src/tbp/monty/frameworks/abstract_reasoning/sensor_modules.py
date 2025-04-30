# sensor_modules.py

import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
import re

from tbp.monty.frameworks.models.monty_base import SensorModuleBase
from tbp.monty.frameworks.models.states import State
from tbp.monty.frameworks.abstract_reasoning.abstract_reference_frames import (
    AbstractReferenceFrame,
    ABSTRACT_FRAME_REGISTRY,
)
from tbp.monty.frameworks.abstract_reasoning.concept_embeddings import (
    ConceptEmbedding,
    CONCEPT_EMBEDDING_REGISTRY,
)


class AbstractSensorModule(SensorModuleBase):
    """Base sensor module for abstract domains.

    Processes inputs from abstract domains and converts them to spatial
    representations compatible with tbp.monty's architecture.
    """

    def __init__(
            self,
            sensor_module_id: str,
            domain: str,
            default_reference_frame: str,
    ):
        """Initialize the abstract sensor module.

        Args:
            sensor_module_id: Unique identifier for this sensor module
            domain: Abstract domain this sensor processes
            default_reference_frame: Default reference frame ID to use
        """
        super().__init__(sensor_module_id)
        self.domain = domain
        self.default_reference_frame_id = default_reference_frame

        # Get the default reference frame
        try:
            self.default_reference_frame = ABSTRACT_FRAME_REGISTRY.get_frame(
                domain, default_reference_frame
            )
        except KeyError:
            logging.error(
                f"Default reference frame {default_reference_frame} not found "
                f"for domain {domain}. Abstract sensor module may not function correctly."
            )
            self.default_reference_frame = None

    def _parse_input(self, data: Any) -> Dict:
        """Parse input data into structured representation.

        Args:
            data: Input data (a format depends on specific sensor implementation)

        Returns:
            Parsed representation of the input
        """
        raise NotImplementedError(
            "Abstract sensor modules must implement _parse_input method"
        )

    def _extract_concepts(self, parsed_data: Dict) -> List[ConceptEmbedding]:
        """Extract concepts from parsed data.

        Args:
            parsed_data: Structured representation of the input

        Returns:
            List of concept embeddings
        """
        raise NotImplementedError(
            "Abstract sensor modules must implement _extract_concepts method"
        )

    def _create_abstract_morphology(
            self,
            concept: ConceptEmbedding,
            reference_frame: AbstractReferenceFrame,
    ) -> Dict:
        """Create morphological features for a concept.

        Args:
            concept: Concept embedding
            reference_frame: Reference frame to use

        Returns:
            Dictionary of morphological features
        """
        # Create an orthonormal set of vectors representing the concept's "pose"
        # This is analogous to pose vectors in physical sensor modules
        pose_vectors = np.zeros((3, 3))

        # Use the first 3 base dimensions (or fewer if dimensionality < 3)
        for i in range(min(3, reference_frame.dimensionality)):
            pose_vectors[i] = reference_frame.base_dimensions[i]

        # If dimensionality < 3, fill in with orthogonal vectors
        if reference_frame.dimensionality < 3:
            # Create random orthogonal vectors to complete the set
            for i in range(reference_frame.dimensionality, 3):
                while True:
                    v = np.random.randn(len(reference_frame.base_dimensions[0]))
                    v = v / np.linalg.norm(v)

                    # Make v orthogonal to all previous vectors
                    for j in range(i):
                        v = v - np.dot(v, pose_vectors[j]) * pose_vectors[j]

                    # Normalize and check if valid
                    norm = np.linalg.norm(v)
                    if norm > 1e-6:  # Ensure not too close to zero
                        pose_vectors[i] = v / norm
                        break

        return {
            "pose_vectors": pose_vectors,
            "pose_fully_defined": True,
            "reference_frame_id": reference_frame.frame_id,
            "concept_domain": concept.domain,
        }

    def step(self, data: Any) -> State:
        """Process abstract domain input into a State.

        Args:
            data: Input data (format depends on specific sensor implementation)

        Returns:
            State representation of the processed input
        """
        # Parse the input data
        parsed_data = self._parse_input(data)

        # Extract concepts from the parsed data
        concepts = self._extract_concepts(parsed_data)

        if not concepts:
            logging.warning(f"No concepts extracted from input by {self.sensor_module_id}")
            return State(
                location=np.zeros(3),
                morphological_features={
                    "pose_vectors": np.eye(3),
                    "pose_fully_defined": False,
                },
                non_morphological_features={"raw_input": data},
                confidence=0.0,
                use_state=False,
                sender_id=self.sensor_module_id,
                sender_type="ASM",  # Abstract Sensor Module
            )

        # Use the most salient concept (first one for now)
        primary_concept = concepts[0]

        # Map the concept to coordinates in the reference frame
        if self.default_reference_frame is not None:
            abstract_location = self.default_reference_frame.position_concept(
                primary_concept.embedding
            )

            # Ensure location is 3D (pad with zeros if needed)
            if len(abstract_location) < 3:
                abstract_location = np.pad(
                    abstract_location, (0, 3 - len(abstract_location))
                )
            elif len(abstract_location) > 3:
                # Take only the first 3 dimensions
                abstract_location = abstract_location[:3]

            # Create morphological features
            morphological_features = self._create_abstract_morphology(
                primary_concept, self.default_reference_frame
            )

            # Non-morphological features include the concept information
            non_morphological_features = {
                "concept_id": primary_concept.concept_id,
                "domain": primary_concept.domain,
                "related_concepts": [c.concept_id for c in concepts[1:]],
                "metadata": primary_concept.metadata,
            }

            # High confidence since we found a concept
            confidence = 1.0
            use_state = True
        else:
            # Fallback if no reference frame is available
            abstract_location = np.zeros(3)
            morphological_features = {
                "pose_vectors": np.eye(3),
                "pose_fully_defined": False,
            }
            non_morphological_features = {"raw_input": data}
            confidence = 0.0
            use_state = False

        return State(
            location=abstract_location,
            morphological_features=morphological_features,
            non_morphological_features=non_morphological_features,
            confidence=confidence,
            use_state=use_state,
            sender_id=self.sensor_module_id,
            sender_type="ASM",  # Abstract Sensor Module
        )


class TextAbstractionSensorModule(AbstractSensorModule):
    """Processes textual descriptions of abstract concepts."""

    def __init__(
            self,
            sensor_module_id: str,
            domain: str,
            default_reference_frame: str,
    ):
        """Initialize the text abstraction sensor module.

        Args:
            sensor_module_id: Unique identifier for this sensor module
            domain: Abstract domain this sensor processes
            default_reference_frame: Default reference frame ID to use
        """
        super().__init__(sensor_module_id, domain, default_reference_frame)

    def _parse_input(self, data: str) -> Dict:
        """Parse textual input into structured representation.

        Args:
            data: Text input

        Returns:
            Parsed representation of the input
        """
        # Simple parsing for now - extract sentences and potential concept mentions
        sentences = re.split(r'[.!?]', data)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Extract potential concept mentions (simplistic approach)
        # In a real implementation, this would use NLP tools for entity extraction
        words = re.findall(r'\b\w+\b', data.lower())
        potential_concepts = [w for w in words if len(w) > 3]  # Basic filtering

        return {
            "text": data,
            "sentences": sentences,
            "potential_concepts": potential_concepts,
        }

    def _extract_concepts(self, parsed_data: Dict) -> List[ConceptEmbedding]:
        """Extract concepts from parsed text data.

        Args:
            parsed_data: Structured representation of the text input

        Returns:
            List of concept embeddings
        """
        concepts = []

        # Use the entire text as a concept description
        text = parsed_data["text"]

        # Create a unique ID based on the text content
        concept_id = f"concept_{hash(text) % 10000}"

        # Get or create the embedding
        concept = CONCEPT_EMBEDDING_REGISTRY.get_or_create_embedding(
            concept_id=concept_id,
            domain=self.domain,
            description=text,
            metadata={"source_text": text},
        )

        concepts.append(concept)

        # Optionally, extract more fine-grained concepts from sentences
        # This is a simplified implementation

        return concepts


class SymbolicReasoningSensorModule(AbstractSensorModule):
    """Processes formal symbolic expressions (mathematical, logical)."""

    def __init__(
            self,
            sensor_module_id: str,
            domain: str,
            default_reference_frame: str,
            symbol_system: str = "general",
    ):
        """Initialize the symbolic reasoning sensor module.

        Args:
            sensor_module_id: Unique identifier for this sensor module
            domain: Abstract domain this sensor processes
            default_reference_frame: Default reference frame ID to use
            symbol_system: Type of symbolic system to process
        """
        super().__init__(sensor_module_id, domain, default_reference_frame)
        self.symbol_system = symbol_system

    def _parse_input(self, data: str) -> Dict:
        """Parse symbolic expressions into structured representation.

        Args:
            data: Symbolic expression as text

        Returns:
            Parsed representation of the expression
        """
        # This is a simplified implementation
        # A real implementation would use domain-specific parsers (e.g., sympy for math)

        # Basic symbol extraction
        symbols = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', data)
        operators = re.findall(r'[\+\-\*/\^\(\)=<>]', data)

        return {
            "expression": data,
            "symbols": symbols,
            "operators": operators,
            "symbol_system": self.symbol_system,
        }

    def _extract_concepts(self, parsed_data: Dict) -> List[ConceptEmbedding]:
        """Extract concepts from parsed symbolic expressions.

        Args:
            parsed_data: Structured representation of the expression

        Returns:
            List of concept embeddings
        """
        # Create a concept from the entire expression
        expression = parsed_data["expression"]

        # Create a unique ID based on the expression
        concept_id = f"expr_{hash(expression) % 10000}"

        # Get or create the embedding
        concept = CONCEPT_EMBEDDING_REGISTRY.get_or_create_embedding(
            concept_id=concept_id,
            domain=self.domain,
            description=expression,
            metadata={
                "expression": expression,
                "symbol_system": parsed_data["symbol_system"],
                "symbols": parsed_data["symbols"],
            },
        )

        return [concept]
