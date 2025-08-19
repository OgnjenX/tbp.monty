# concept_embeddings.py

import logging
from typing import Dict, List, Optional, Tuple
from collections import deque

import numpy as np


class TemporalConceptEmbedding:
    """Biologically plausible concept representation learned through temporal sequences.

    Instead of using transformer embeddings, this uses simple associative learning
    and temporal context to build concept representations that align with TBT principles.
    """

    def __init__(
            self,
            concept_id: str,
            domain: str,
            embedding_dim: int = 50,
            temporal_window: int = 5,
            learning_rate: float = 0.01,
    ):
        """Initialize a temporal concept embedding.

        Args:
            concept_id: Unique identifier for this concept
            domain: Abstract domain this concept belongs to
            embedding_dim: Dimensionality of the embedding space
            temporal_window: Size of temporal context window
            learning_rate: Rate of adaptation for the embedding
        """
        self.concept_id = concept_id
        self.domain = domain
        self.embedding_dim = embedding_dim
        self.temporal_window = temporal_window
        self.learning_rate = learning_rate

        # Core embedding - learned through experience
        self.embedding = np.random.normal(0, 0.1, embedding_dim)

        # Temporal context tracking
        self.temporal_context = deque(maxlen=temporal_window)
        self.co_occurrence_counts = {}  # concept_id -> count
        self.transition_counts = {}  # (prev_concept, next_concept) -> count

        # Learning state
        self.activation_count = 0
        self.last_activation_time = 0

        # Metadata for compatibility
        self.metadata = {}

    def activate_in_context(
        self,
        temporal_context: List[str],
        current_time: int
    ) -> None:
        """Activate this concept in a temporal context.

        Args:
            temporal_context: Recent sequence of concept activations
            current_time: Current time step
        """
        self.activation_count += 1
        self.last_activation_time = current_time

        # Update temporal context
        self.temporal_context.extend(temporal_context)

        # Update co-occurrence counts
        for context_concept in temporal_context:
            if context_concept != self.concept_id:
                if context_concept not in self.co_occurrence_counts:
                    self.co_occurrence_counts[context_concept] = 0
                self.co_occurrence_counts[context_concept] += 1

        # Update transition counts
        if len(temporal_context) >= 2:
            for i in range(len(temporal_context) - 1):
                transition = (temporal_context[i], temporal_context[i + 1])
                if transition not in self.transition_counts:
                    self.transition_counts[transition] = 0
                self.transition_counts[transition] += 1

        # Adapt embedding based on context
        self._adapt_embedding(temporal_context)

    def _adapt_embedding(self, temporal_context: List[str]) -> None:
        """Adapt embedding based on temporal context using simple Hebbian learning."""
        if not temporal_context:
            return

        # Create context vector from co-occurrence patterns
        context_vector = np.zeros(self.embedding_dim)

        # Simple hash-based context representation (biologically plausible)
        for context_concept in temporal_context:
            # Use hash to create consistent but distributed representation
            concept_hash = hash(context_concept) % (2**32)
            for i in range(self.embedding_dim):
                bit_position = (concept_hash >> i) & 1
                context_vector[i] += (2 * bit_position - 1) / len(temporal_context)

        # Normalize context vector
        context_norm = np.linalg.norm(context_vector)
        if context_norm > 0:
            context_vector = context_vector / context_norm

            # Hebbian update: strengthen connections with active context
            self.embedding += self.learning_rate * context_vector

    def similarity(self, other: 'TemporalConceptEmbedding') -> float:
        """Calculate similarity with another concept using multiple factors.

        Args:
            other: Another concept embedding

        Returns:
            Similarity score between 0 and 1
        """
        # Embedding similarity (cosine)
        embedding_sim = self._cosine_similarity(self.embedding, other.embedding)

        # Co-occurrence similarity
        cooccur_sim = self._co_occurrence_similarity(other)

        # Temporal transition similarity
        transition_sim = self._transition_similarity(other)

        # Weighted combination
        total_sim = (
            0.5 * embedding_sim +
            0.3 * cooccur_sim +
            0.2 * transition_sim
        )

        return max(0.0, min(1.0, total_sim))

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return 0.0
        return dot_product / norm_product

    def _co_occurrence_similarity(self, other: 'TemporalConceptEmbedding') -> float:
        """Calculate similarity based on shared co-occurrence patterns."""
        self_concepts = set(self.co_occurrence_counts.keys())
        other_concepts = set(other.co_occurrence_counts.keys())

        if not self_concepts and not other_concepts:
            return 1.0

        intersection = self_concepts.intersection(other_concepts)
        union = self_concepts.union(other_concepts)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _transition_similarity(self, other: 'TemporalConceptEmbedding') -> float:
        """Calculate similarity based on shared transition patterns."""
        self_transitions = set(self.transition_counts.keys())
        other_transitions = set(other.transition_counts.keys())

        if not self_transitions and not other_transitions:
            return 1.0

        intersection = self_transitions.intersection(other_transitions)
        union = self_transitions.union(other_transitions)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def get_related_concepts(self, threshold: float = 0.1) -> List[Tuple[str, int]]:
        """Get concepts that frequently co-occur with this one.

        Args:
            threshold: Minimum co-occurrence count threshold

        Returns:
            List of (concept_id, count) tuples
        """
        related = []
        for concept_id, count in self.co_occurrence_counts.items():
            if count >= threshold * self.activation_count:
                related.append((concept_id, count))

        return sorted(related, key=lambda x: x[1], reverse=True)

    def predict_next_concepts(self, current_context: List[str]) -> List[Tuple[str, float]]:
        """Predict likely next concepts based on current context.

        Args:
            current_context: Current temporal context

        Returns:
            List of (concept_id, probability) tuples
        """
        if not current_context:
            return []

        # Look for transitions from recent context
        predictions = {}

        for prev_concept in current_context[-2:]:  # Look at last 2 concepts
            for (source, target), count in self.transition_counts.items():
                if source == prev_concept:
                    if target not in predictions:
                        predictions[target] = 0
                    predictions[target] += count

        # Normalize to probabilities
        total_count = sum(predictions.values())
        if total_count == 0:
            return []

        result = [(concept, count / total_count) for concept, count in predictions.items()]
        return sorted(result, key=lambda x: x[1], reverse=True)

    def __repr__(self) -> str:
        """String representation of the concept embedding."""
        return f"TemporalConceptEmbedding(id={self.concept_id}, domain={self.domain}, activations={self.activation_count})"


class LocalConceptEmbeddingManager:
    """Local manager for concept embeddings within a single learning module.

    Replaces global registries to maintain cortical column independence.
    Uses biologically plausible learning mechanisms instead of transformers.
    """

    def __init__(
        self,
        module_id: str,
        embedding_dim: int = 50,
        temporal_window: int = 5,
        learning_rate: float = 0.01
    ):
        """Initialize local concept embedding manager.

        Args:
            module_id: Unique identifier for the owning module
            embedding_dim: Dimensionality of concept embeddings
            temporal_window: Size of temporal context window
            learning_rate: Learning rate for embedding adaptation
        """
        self.module_id = module_id
        self.embedding_dim = embedding_dim
        self.temporal_window = temporal_window
        self.learning_rate = learning_rate

        # Local storage - no global dependencies
        self.embeddings = {}  # concept_id -> TemporalConceptEmbedding
        self.temporal_sequence = deque(maxlen=temporal_window * 2)
        self.current_time = 0

        # Domain-specific parameters
        self.domain_contexts = {}  # domain -> recent_concepts

    def get_or_create_embedding(
        self,
        concept_id: str,
        domain: str,
        description: Optional[str] = None,
        temporal_context: Optional[List[str]] = None
    ) -> TemporalConceptEmbedding:
        """Get existing embedding or create new one through sensorimotor experience.

        Args:
            concept_id: Unique identifier for this concept
            domain: Abstract domain this concept belongs to
            description: Textual description (used for initialization only)
            temporal_context: Current temporal context

        Returns:
            TemporalConceptEmbedding for the concept
        """
        if concept_id not in self.embeddings:
            # Create new embedding
            self.embeddings[concept_id] = TemporalConceptEmbedding(
                concept_id=concept_id,
                domain=domain,
                embedding_dim=self.embedding_dim,
                temporal_window=self.temporal_window,
                learning_rate=self.learning_rate
            )

            # Initialize with simple description-based features if available
            if description:
                self._initialize_from_description(self.embeddings[concept_id], description)

        # Activate in current temporal context
        if temporal_context is None:
            temporal_context = list(self.temporal_sequence)

        self.embeddings[concept_id].activate_in_context(temporal_context, self.current_time)

        # Update temporal sequence
        self.temporal_sequence.append(concept_id)
        self.current_time += 1

        # Update domain context
        if domain not in self.domain_contexts:
            self.domain_contexts[domain] = deque(maxlen=self.temporal_window)
        self.domain_contexts[domain].append(concept_id)

        return self.embeddings[concept_id]

    def _initialize_from_description(
        self,
        embedding: TemporalConceptEmbedding,
        description: str
    ) -> None:
        """Initialize embedding from description using simple hashing.

        This replaces transformer-based initialization with a biologically
        plausible approach using consistent hashing.
        """
        # Use multiple hash functions to create distributed representation
        for i in range(self.embedding_dim):
            # Create different hash seeds for each dimension
            hash_seed = hash(description + str(i)) % (2**32)
            # Convert to value between -1 and 1
            embedding.embedding[i] = (hash_seed / (2**31)) - 1

        # Normalize
        norm = np.linalg.norm(embedding.embedding)
        if norm > 0:
            embedding.embedding = embedding.embedding / norm

    def get_embedding(self, concept_id: str) -> Optional[TemporalConceptEmbedding]:
        """Get existing embedding without creating new one."""
        return self.embeddings.get(concept_id)

    def get_similar_concepts(
        self,
        concept_id: str,
        threshold: float = 0.5,
        max_results: int = 10
    ) -> List[Tuple[str, float]]:
        """Get concepts similar to the given concept.

        Args:
            concept_id: Reference concept
            threshold: Minimum similarity threshold
            max_results: Maximum number of results

        Returns:
            List of (concept_id, similarity) tuples
        """
        if concept_id not in self.embeddings:
            return []

        reference_embedding = self.embeddings[concept_id]
        similarities = []

        for other_id, other_embedding in self.embeddings.items():
            if other_id != concept_id:
                similarity = reference_embedding.similarity(other_embedding)
                if similarity >= threshold:
                    similarities.append((other_id, similarity))

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]

    def get_temporal_context(self, domain: Optional[str] = None) -> List[str]:
        """Get current temporal context, optionally filtered by domain."""
        if domain is None:
            return list(self.temporal_sequence)
        else:
            return list(self.domain_contexts.get(domain, []))

    def predict_next_concept(
        self,
        current_concept: str,
        domain: Optional[str] = None
    ) -> Optional[Tuple[str, float]]:
        """Predict most likely next concept based on temporal patterns."""
        if current_concept not in self.embeddings:
            return None

        embedding = self.embeddings[current_concept]
        context = self.get_temporal_context(domain)

        predictions = embedding.predict_next_concepts(context)
        if predictions:
            return predictions[0]  # Return most likely

        return None

    def get_concept_statistics(self) -> Dict[str, Any]:
        """Get statistics about the local concept embeddings."""
        return {
            'total_concepts': len(self.embeddings),
            'total_activations': sum(emb.activation_count for emb in self.embeddings.values()),
            'current_time': self.current_time,
            'domains': list(self.domain_contexts.keys()),
            'temporal_sequence_length': len(self.temporal_sequence)
        }


# Note: Global registries removed to maintain cortical column independence
# Each learning module now has its own LocalConceptEmbeddingManager
