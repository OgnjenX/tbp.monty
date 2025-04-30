# concept_embeddings.py

import logging
from typing import Dict, Optional

import numpy as np

try:
    import torch
    import transformers

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("transformers package not found, using fallback embeddings")


class ConceptEmbedding:
    """Embedding representation for abstract concepts."""

    def __init__(
            self,
            concept_id: str,
            domain: str,
            embedding: np.ndarray,
            metadata: Optional[Dict] = None,
    ):
        """Initialize a concept embedding.

        Args:
            concept_id: Unique identifier for this concept
            domain: Abstract domain this concept belongs to
            embedding: Vector representation of the concept
            metadata: Additional information about the concept
        """
        self.concept_id = concept_id
        self.domain = domain
        self.embedding = embedding
        self.metadata = metadata or {}
        self.embedding_dim = len(embedding)

    def similarity(self, other: 'ConceptEmbedding') -> float:
        """Calculate cosine similarity with another concept.

        Args:
            other: Another concept embedding

        Returns:
            Similarity score between 0 and 1
        """
        dot_product = np.dot(self.embedding, other.embedding)
        norm_product = np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding)
        return dot_product / norm_product

    def __repr__(self) -> str:
        """String representation of the concept embedding."""
        return f"ConceptEmbedding(id={self.concept_id}, domain={self.domain}, dim={self.embedding_dim})"


class ConceptEmbeddingGenerator:
    """Generates embeddings for abstract concepts using language models."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize the concept embedding generator.

        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model."""
        if HAS_TRANSFORMERS:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.use_fallback = False
            except ImportError:
                logging.warning("SentenceTransformer not found, using fallback embeddings")
                self.use_fallback = True
                self.embedding_dim = 100
        else:
            self.use_fallback = True
            self.embedding_dim = 100

    def generate_embedding(
            self,
            concept_id: str,
            domain: str,
            description: str,
            metadata: Optional[Dict] = None,
    ) -> ConceptEmbedding:
        """Generate embedding for a concept.

        Args:
            concept_id: Unique identifier for this concept
            domain: Abstract domain this concept belongs to
            description: Textual description of the concept
            metadata: Additional information about the concept

        Returns:
            ConceptEmbedding for the concept
        """
        if not self.use_fallback:
            # Use transformers model for embedding
            with torch.no_grad():
                embedding = self.model.encode(description)
        else:
            # Fallback embedding (random but consistent for same concept_id)
            np.random.seed(hash(concept_id) % 2 ** 32)
            embedding = np.random.normal(0, 1, self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)

        return ConceptEmbedding(
            concept_id=concept_id,
            domain=domain,
            embedding=embedding,
            metadata=metadata,
        )


class ConceptEmbeddingRegistry:
    """Registry for storing and retrieving concept embeddings."""

    def __init__(self):
        """Initialize the concept embedding registry."""
        self.embeddings = {}  # domain -> {concept_id -> ConceptEmbedding}
        self.generator = ConceptEmbeddingGenerator()

    def register_embedding(self, embedding: ConceptEmbedding):
        """Register a concept embedding with the registry.

        Args:
            embedding: The concept embedding to register
        """
        if embedding.domain not in self.embeddings:
            self.embeddings[embedding.domain] = {}

        self.embeddings[embedding.domain][embedding.concept_id] = embedding

    def get_embedding(self, domain: str, concept_id: str) -> Optional[ConceptEmbedding]:
        """Get a concept embedding by domain and ID.

        Args:
            domain: Abstract domain
            concept_id: Unique identifier for the concept

        Returns:
            The requested ConceptEmbedding or None if not found
        """
        if domain not in self.embeddings or concept_id not in self.embeddings[domain]:
            return None

        return self.embeddings[domain][concept_id]

    def get_or_create_embedding(
            self,
            concept_id: str,
            domain: str,
            description: str,
            metadata: Optional[Dict] = None,
    ) -> ConceptEmbedding:
        """Get an existing embedding or create a new one.

        Args:
            concept_id: Unique identifier for this concept
            domain: Abstract domain this concept belongs to
            description: Textual description of the concept
            metadata: Additional information about the concept

        Returns:
            ConceptEmbedding for the concept
        """
        existing = self.get_embedding(domain, concept_id)
        if existing is not None:
            return existing

        new_embedding = self.generator.generate_embedding(
            concept_id=concept_id,
            domain=domain,
            description=description,
            metadata=metadata,
        )

        self.register_embedding(new_embedding)
        return new_embedding


# Global registry for concept embeddings
CONCEPT_EMBEDDING_REGISTRY = ConceptEmbeddingRegistry()
