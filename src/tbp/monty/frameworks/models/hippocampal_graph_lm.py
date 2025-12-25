# Copyright 2025 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Hippocampal Graph Learning Module for fast episodic relational learning.

This module implements a fast-learning relational memory system inspired by
the hippocampus, but using explicit graph structures similar to neocortical
learning modules. Key differences from neocortical LMs:

- Nodes represent recognized objects (not sensor features)
- Edges represent co-occurrence and spatial relations (not just displacements)
- Learning is one-shot or few-shot (not gradual consolidation)
- Operates on object-level inputs from neocortical LMs

This allows learning relations like:
- "Object A often appears with object B"
- "Object A is usually to the left of object B"
- "After seeing A, I usually see B next"
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from tbp.monty.frameworks.models.graph_matching import GraphLM

# Use modern numpy random generation
# Note: `seed=42` makes the generator deterministic for reproducible
# tests and debugging. The value 42 is arbitrary (a cultural convention);
# any fixed integer would work. For production use prefer a configurable
# seed (constructor arg or environment) or omit the seed for non-determinism.
_rng = np.random.default_rng(seed=42)

logger = logging.getLogger(__name__)


@dataclass
class ObjectObservation:
    """A single observation of a recognized object.
    
    Attributes:
        object_id: Unique identifier for the object.
        location: 3D location where object was observed.
        pose: Rotation matrix (3x3) of the object.
        timestamp: When the observation occurred.
        confidence: Recognition confidence (0-1).
        source_lm: Which learning module detected this object.
    """
    object_id: str
    location: np.ndarray
    pose: Optional[np.ndarray] = None
    timestamp: float = 0.0
    confidence: float = 1.0
    source_lm: Optional[str] = None
    
    def __post_init__(self):
        """Convert location to numpy array if needed."""
        if not isinstance(self.location, np.ndarray):
            self.location = np.array(self.location)


@dataclass
class ObjectRelation:
    """A learned relation between two objects.
    
    Attributes:
        object_a: First object ID.
        object_b: Second object ID.
        co_occurrence_count: How many times they appeared together.
        spatial_displacements: List of relative positions (B - A).
        temporal_offsets: List of time differences (when B was seen after A).
        contexts: Set of episode contexts where this relation was observed.
        last_update_time: Timestamp of most recent update.
    """
    object_a: str
    object_b: str
    co_occurrence_count: int = 0
    co_occurrence_weight: float = 0.0
    spatial_displacements: List[np.ndarray] = field(default_factory=list)
    temporal_offsets: List[float] = field(default_factory=list)
    contexts: Set[str] = field(default_factory=set)
    last_update_time: float = 0.0
    
    def add_observation(
        self,
        spatial_displacement: Optional[np.ndarray] = None,
        temporal_offset: Optional[float] = None,
        context: Optional[str] = None,
        timestamp: float = 0.0,
        weight: float = 1.0,
    ):
        """Record a new observation of this relation."""
        self.co_occurrence_count += 1
        self.co_occurrence_weight += float(weight)
        self.last_update_time = timestamp
        
        if spatial_displacement is not None:
            self.spatial_displacements.append(spatial_displacement)
        
        if temporal_offset is not None:
            self.temporal_offsets.append(temporal_offset)
        
        if context is not None:
            self.contexts.add(context)
    
    @property
    def average_spatial_displacement(self) -> Optional[np.ndarray]:
        """Get average spatial relation between objects."""
        if not self.spatial_displacements:
            return None
        return np.mean(self.spatial_displacements, axis=0)
    
    @property
    def average_temporal_offset(self) -> Optional[float]:
        """Get average time between observations."""
        if not self.temporal_offsets:
            return None
        return float(np.mean(self.temporal_offsets))


@dataclass
class ReplayBatch:
    """A batch of observations for replay to downstream LMs.
    
    Attributes:
        observations: Ordered sequence of ObjectObservations.
        source_type: 'episode' or 'relation_centric'.
        batch_id: Unique identifier for this replay batch.
        context: Optional context label.
    """
    observations: List[ObjectObservation]
    source_type: str  # 'episode' or 'relation_centric'
    batch_id: str
    context: Optional[str] = None
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def get_object_ids(self) -> List[str]:
        """Get sequence of object IDs in this batch."""
        return [obs.object_id for obs in self.observations]


@dataclass(frozen=True)
class EpisodeRecord:
    """Stored episode for replay.

    Attributes:
        observations: Ordered sequence of ObjectObservations.
        context: Optional context label associated with the episode.
    """
    observations: List[ObjectObservation]
    context: Optional[str] = None


class HippocampalGraphMemory:
    """Memory store for object relations using graph structure.
    
    This stores a relational graph where:
    - Nodes = unique object IDs
    - Edges = learned relations (co-occurrence, spatial, temporal)
    
    Also maintains episode history for replay operations.
    
    Attributes:
        relations: Dict mapping (obj_a, obj_b) -> ObjectRelation
        objects_seen: Set of all object IDs encountered
        episode_count: Total number of episodes processed
        episode_history: List of past episodes for replay
        decay_rate: Exponential decay factor for relation weighting (0=no decay)
    """
    
    def __init__(self, learning_rate: float = 0.8, decay_rate: float = 0.0):
        """Initialize hippocampal graph memory.
        
        Args:
            learning_rate: How quickly to update relations (0-1).
                Higher = faster learning. For one-shot learning, use ~0.8-1.0.
            decay_rate: Exponential decay for older relations (0-1).
                0 = no decay (infinite memory). Higher = faster forgetting.
                Biologically motivated: hippocampus does consolidate to cortex.
        """
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # Relational graph storage
        self.relations: Dict[Tuple[str, str], ObjectRelation] = {}
        self.objects_seen: Set[str] = set()
        
        # Episode history for replay
        self.episode_history: List[EpisodeRecord] = []
        
        # Statistics
        self.episode_count = 0
        self.total_observations = 0
        self.current_time = 0.0  # Logical time for decay calculations
    
    def add_relation(
        self,
        object_a: str,
        object_b: str,
        spatial_displacement: Optional[np.ndarray] = None,
        temporal_offset: Optional[float] = None,
        context: Optional[str] = None,
        timestamp: float = 0.0,
        weight: float = 1.0,
    ) -> None:
        """Add or update a relation between two objects.
        
        Args:
            object_a: First object ID.
            object_b: Second object ID.
            spatial_displacement: Relative position (B location - A location).
            temporal_offset: Time difference (B timestamp - A timestamp).
            context: Episode context identifier.
            timestamp: When this relation was observed.
        """
        # Track objects
        self.objects_seen.add(object_a)
        self.objects_seen.add(object_b)
        
        # Create bidirectional relations (undirected graph for co-occurrence)
        for pair in [(object_a, object_b), (object_b, object_a)]:
            if pair not in self.relations:
                self.relations[pair] = ObjectRelation(
                    object_a=pair[0],
                    object_b=pair[1],
                )
            
            # For spatial displacement, reverse for second direction
            if spatial_displacement is not None and pair[1] == object_a:
                spatial_displacement = -spatial_displacement
            
            self.relations[pair].add_observation(
                spatial_displacement=spatial_displacement,
                temporal_offset=temporal_offset,
                context=context,
                timestamp=timestamp,
                weight=weight,
            )
    
    def get_related_objects(
        self,
        object_id: str,
        min_count: int = 1,
    ) -> Dict[str, ObjectRelation]:
        """Get all objects that have co-occurred with the given object.
        
        Args:
            object_id: Object to query relations for.
            min_count: Minimum co-occurrence count to include.
        
        Returns:
            Dict mapping related object IDs to their ObjectRelation.
        """
        related = {}
        for (obj_a, obj_b), relation in self.relations.items():
            if obj_a == object_id and relation.co_occurrence_count >= min_count:
                related[obj_b] = relation
        return related
    
    def get_relation(
        self,
        object_a: str,
        object_b: str,
    ) -> Optional[ObjectRelation]:
        """Get the relation between two specific objects.
        
        Returns:
            ObjectRelation if exists, None otherwise.
        """
        return self.relations.get((object_a, object_b))
    
    def get_most_common_neighbors(
        self,
        object_id: str,
        top_k: int = 5,
    ) -> List[Tuple[str, int]]:
        """Get the k most frequently co-occurring objects.
        
        Args:
            object_id: Object to query.
            top_k: Number of neighbors to return.
        
        Returns:
            List of (object_id, count) tuples, sorted by count descending.
        """
        related = self.get_related_objects(object_id)
        sorted_relations = sorted(
            related.items(),
            key=lambda x: x[1].co_occurrence_count,
            reverse=True,
        )
        return [(obj_id, rel.co_occurrence_count) for obj_id, rel in sorted_relations[:top_k]]
    
    def predict_next_object(
        self,
        current_object: str,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """Predict what object is likely to appear next.
        
        Based on temporal sequence statistics.
        
        Args:
            current_object: Object currently being observed.
            context: Optional context to filter predictions.
        
        Returns:
            Most likely next object ID, or None if no data.
        """
        related = self.get_related_objects(current_object)
        
        # Filter by context if provided
        if context:
            related = {
                obj_id: rel for obj_id, rel in related.items()
                if context in rel.contexts
            }
        
        if not related:
            return None
        
        # Return object with highest co-occurrence count
        # (could be made more sophisticated with temporal ordering)
        best_object = max(
            related.items(),
            key=lambda x: x[1].co_occurrence_count,
        )[0]
        
        return best_object
    
    def clear(self) -> None:
        """Clear all stored relations."""
        self.relations.clear()
        self.objects_seen.clear()
        self.episode_history.clear()
        self.episode_count = 0
        self.total_observations = 0
        self.current_time = 0.0
    
    def store_episode(
        self,
        observations: List[ObjectObservation],
        context: Optional[str] = None,
    ) -> None:
        """Store an episode in history for potential replay.
        
        Args:
            observations: List of observations in the episode.
            context: Optional context label for the episode.
        """
        self.episode_history.append(
            EpisodeRecord(observations=observations.copy(), context=context)
        )
    
    def generate_episode_replay(
        self,
        num_replays: int = 1,
        context_filter: Optional[str] = None,
    ) -> List[ReplayBatch]:
        """Generate replay batches from stored episodes.
        
        Args:
            num_replays: Number of replay batches to generate.
            context_filter: If set, only replay episodes with this context.
        
        Returns:
            List of ReplayBatch objects with sampled episodes.
        """
        if not self.episode_history:
            return []
        
        # Filter episodes by context if requested
        candidates = self.episode_history
        if context_filter:
            candidates = [
                ep for ep in self.episode_history
                if context_filter in (ep.context or "")
            ]
        
        if not candidates:
            return []
        
        replays = []
        for i in range(num_replays):
            # Sample an episode
            episode = candidates[_rng.integers(0, len(candidates))]
            
            batch = ReplayBatch(
                observations=episode.observations.copy(),
                source_type='episode',
                batch_id=f'episode_replay_{self.episode_count}_{i}',
                context=episode.context,
            )
            replays.append(batch)
        
        return replays

    def retrieve_episodes_by_cue(
        self,
        cue_object_ids: List[str],
        context_filter: Optional[str] = None,
        top_k: int = 1,
        min_jaccard: float = 0.0,
    ) -> List[EpisodeRecord]:
        """Retrieve stored episodes that best match a partial cue.

        This is an engineering analogue of hippocampal pattern completion:
        given a subset of objects, return the most similar stored episodes.

        Similarity is computed as Jaccard overlap between the cue object set
        and the episode's object set, with ties broken by recency.

        Args:
            cue_object_ids: Partial set of object IDs observed (the cue).
            context_filter: Optional context to restrict candidate episodes.
            top_k: Maximum number of episodes to return.
            min_jaccard: Minimum Jaccard similarity for a match.

        Returns:
            List of EpisodeRecord, sorted best-first.
        """
        if top_k <= 0:
            return []

        cue_set = set(cue_object_ids)
        if not cue_set or not self.episode_history:
            return []

        scored: List[Tuple[float, int, EpisodeRecord]] = []
        for idx, ep in enumerate(self.episode_history):
            if context_filter and context_filter not in (ep.context or ""):
                continue

            episode_set = {obs.object_id for obs in ep.observations}
            intersection = len(cue_set & episode_set)
            if intersection == 0:
                continue

            union = len(cue_set | episode_set)
            jaccard = intersection / max(1, union)
            if jaccard >= min_jaccard:
                scored.append((jaccard, idx, ep))

        # Best-first: higher similarity, then more recent (larger idx).
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [ep for _, _, ep in scored[:top_k]]

    def complete_from_cue(
        self,
        cue_object_ids: List[str],
        context_filter: Optional[str] = None,
        top_k_episodes: int = 5,
        top_k_objects: int = 5,
        min_jaccard: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Suggest likely missing objects given a partial cue.

        This uses retrieved episodes as "completions" and aggregates their
        objects not already present in the cue.

        Args:
            cue_object_ids: Partial set of object IDs observed (the cue).
            context_filter: Optional context to restrict candidate episodes.
            top_k_episodes: How many best-matching episodes to use.
            top_k_objects: How many object suggestions to return.
            min_jaccard: Minimum Jaccard similarity for an episode to contribute.

        Returns:
            List of (object_id, score) sorted by score descending.
        """
        if top_k_objects <= 0:
            return []

        cue_set = set(cue_object_ids)
        episodes = self.retrieve_episodes_by_cue(
            cue_object_ids=cue_object_ids,
            context_filter=context_filter,
            top_k=top_k_episodes,
            min_jaccard=min_jaccard,
        )
        if not episodes:
            return []

        scores: Dict[str, float] = {}
        for ep in episodes:
            episode_set = {obs.object_id for obs in ep.observations}
            intersection = len(cue_set & episode_set)
            union = len(cue_set | episode_set)
            jaccard = intersection / max(1, union)

            for obj_id in episode_set - cue_set:
                scores[obj_id] = scores.get(obj_id, 0.0) + jaccard

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k_objects]
    
    def _score_relations(
        self,
        context_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Tuple[Tuple[str, str], ObjectRelation, int]]:
        """Extract and score high-frequency relations.
        
        Args:
            context_filter: Optional context to filter relations.
            top_k: Number of top relations to return.
        
        Returns:
            List of [(obj_pair, relation, count), ...] sorted by count.
        """
        scored_relations = []
        
        for (obj_a, obj_b), relation in self.relations.items():
            if obj_a >= obj_b:  # Skip symmetric duplicates
                continue
            
            # Score by co-occurrence weight (falls back to count if unset),
            # optionally filtered by context.
            count = int(relation.co_occurrence_weight) if relation.co_occurrence_weight > 0 else relation.co_occurrence_count
            if context_filter and context_filter not in relation.contexts:
                count = 0
            
            if count > 0:
                scored_relations.append(((obj_a, obj_b), relation, count))
        
        # Sort by frequency and return top-k
        scored_relations.sort(key=lambda x: x[2], reverse=True)
        return scored_relations[:top_k]
    
    def _compute_sampling_probs(
        self,
        counts: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Compute sampling probabilities with temperature control.
        
        Args:
            counts: Co-occurrence counts for relations.
            temperature: Sampling temperature (0=greedy, 1=uniform, >1=exploratory).
        
        Returns:
            Normalized probability distribution.
        """
        if temperature > 0:
            probs = np.exp(np.log(counts + 1e-10) / temperature)
            probs /= probs.sum()
        else:
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
        
        return probs
    
    def _create_relation_mini_episode(
        self,
        obj_a: str,
        obj_b: str,
        relation: ObjectRelation,
    ) -> ReplayBatch:
        """Create a mini-episode from a relation.
        
        Args:
            obj_a: First object ID.
            obj_b: Second object ID.
            relation: Relation stats.
        
        Returns:
            ReplayBatch with synthetic mini-episode [A, B].
        """
        # Create mini-episode: A at origin, then B with learned displacement
        obs_a = ObjectObservation(
            object_id=obj_a,
            location=np.array([0.0, 0.0, 0.0]),
            timestamp=0.0,
            confidence=1.0,
        )
        
        # Use average displacement from learned relation
        avg_displacement = relation.average_spatial_displacement
        if avg_displacement is None:
            avg_displacement = np.array([0.0, 0.0, 0.0])
        
        # Use average temporal offset
        avg_offset = relation.average_temporal_offset
        if avg_offset is None:
            avg_offset = 1.0
        
        obs_b = ObjectObservation(
            object_id=obj_b,
            location=obs_a.location + avg_displacement,
            timestamp=obs_a.timestamp + avg_offset,
            confidence=1.0,
        )
        
        return ReplayBatch(
            observations=[obs_a, obs_b],
            source_type='relation_centric',
            batch_id=f'relation_replay_{self.episode_count}_{obj_a}_{obj_b}',
            context=list(relation.contexts)[0] if relation.contexts else None,
        )
    
    def generate_relation_replay(
        self,
        num_replays: int = 1,
        top_k_relations: int = 10,
        temperature: float = 1.0,
        context_filter: Optional[str] = None,
    ) -> List[ReplayBatch]:
        """Generate relation-centric replay batches.
        
        Creates mini-episodes from high-frequency relations, allowing
        downstream models to strengthen bindings without replaying full episodes.
        
        Args:
            num_replays: Number of replay batches to generate.
            top_k_relations: Consider only top-k relations by frequency.
            temperature: Sampling temperature (1.0=probabilistic, 0=greedy).
            context_filter: If set, only include relations from this context.
        
        Returns:
            List of ReplayBatch objects with relation-centric mini-episodes.
        """
        if not self.relations:
            return []
        
        # Get top relations
        top_relations = self._score_relations(context_filter, top_k_relations)
        if not top_relations:
            return []
        
        # Get sampling probabilities
        counts = np.array([score for _, _, score in top_relations], dtype=float)
        probs = self._compute_sampling_probs(counts, temperature)
        
        # Generate replay batches
        replays = []
        for _ in range(num_replays):
            idx = _rng.choice(len(top_relations), p=probs)
            obj_a, obj_b = top_relations[idx][0]
            relation = top_relations[idx][1]
            batch = self._create_relation_mini_episode(obj_a, obj_b, relation)
            replays.append(batch)
        
        return replays
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dict with statistics about the relational graph.
        """
        return {
            "n_objects": len(self.objects_seen),
            "n_relations": len(self.relations) // 2,  # Divide by 2 for bidirectional
            "episode_count": self.episode_count,
            "total_observations": self.total_observations,
            "avg_relations_per_object": (
                len(self.relations) / max(1, len(self.objects_seen))
            ),
            "stored_episodes": len(self.episode_history),
        }


class HippocampalGraphLM(GraphLM):
    """Fast-learning graph LM for episodic object relations.
    
    This learning module operates at the object level, receiving recognized
    objects from neocortical LMs and learning relations between them. Unlike
    neocortical LMs which learn slowly over many episodes, this learns
    relations in one or few episodes (one-shot learning).
    
    Key capabilities:
    - Learn object co-occurrence ("A and B appear together")
    - Learn spatial relations ("B is usually to the right of A")
    - Learn temporal sequences ("B usually appears after A")
    - Query relations ("What objects appear with A?")
    - Predict next object ("Given A, what comes next?")
    
    Example:
        >>> hipp_lm = HippocampalGraphLM()
        >>> # Episode 1: observe mug and spoon
        >>> hipp_lm.observe_object('mug', location=[0, 0, 0], timestamp=0)
        >>> hipp_lm.observe_object('spoon', location=[0.2, 0, 0], timestamp=1)
        >>> hipp_lm.end_episode()
        >>> # Episode 2: observe them again
        >>> hipp_lm.observe_object('mug', location=[1, 1, 0], timestamp=0)
        >>> hipp_lm.observe_object('spoon', location=[1.2, 1, 0], timestamp=1)
        >>> hipp_lm.end_episode()
        >>> # Query: What appears with mug?
        >>> relations = hipp_lm.get_related_objects('mug')
        >>> print(relations['spoon'].co_occurrence_count)  # 2
    """
    
    def __init__(
        self,
        learning_rate: float = 0.8,
        max_observations_per_episode: int = 1000,
        decay_rate: float = 0.0,
        enable_replay: bool = True,
        max_episode_history: int = 100,
    ):
        """Initialize hippocampal graph learning module.
        
        Args:
            learning_rate: How quickly to learn relations (0-1).
                For one-shot learning, use 0.8-1.0.
            max_observations_per_episode: Maximum observations to store
                per episode before consolidation is triggered.
            decay_rate: Exponential decay for older relations (0-1).
                0 = no decay. Higher = faster forgetting.
            enable_replay: Whether to store episode history for replay.
            max_episode_history: Maximum episodes to keep in history.
        """
        super().__init__()
        
        self.graph_memory = HippocampalGraphMemory(
            learning_rate=learning_rate,
            decay_rate=decay_rate,
        )
        self.max_observations_per_episode = max_observations_per_episode
        self.enable_replay = enable_replay
        self.max_episode_history = max_episode_history
        
        # Current episode buffer
        self.current_episode_observations: List[ObjectObservation] = []
        self.current_episode_context: Optional[str] = None
        
        # Replay callbacks
        self.replay_callbacks: List[Callable[[ReplayBatch], None]] = []
        
        logger.info(
            f"Initialized HippocampalGraphLM with learning_rate={learning_rate}, "
            f"decay_rate={decay_rate}, enable_replay={enable_replay}"
        )
    
    def observe_object(
        self,
        object_id: str,
        location: np.ndarray,
        pose: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
        confidence: float = 1.0,
        source_lm: Optional[str] = None,
    ) -> None:
        """Observe a recognized object during the current episode.
        
        Args:
            object_id: Unique identifier for the recognized object.
            location: 3D location where object was observed.
            pose: Optional rotation matrix of the object.
            timestamp: When this observation occurred.
            confidence: Recognition confidence (0-1).
            source_lm: Which learning module detected this object.
        """
        observation = ObjectObservation(
            object_id=object_id,
            location=location,
            pose=pose,
            timestamp=timestamp,
            confidence=confidence,
            source_lm=source_lm,
        )
        
        self.current_episode_observations.append(observation)
        self.graph_memory.total_observations += 1
        
        # Auto-consolidate if too many observations
        if len(self.current_episode_observations) >= self.max_observations_per_episode:
            logger.warning(
                f"Auto-consolidating episode after "
                f"{self.max_observations_per_episode} observations"
            )
            self.end_episode()
    
    def set_episode_context(self, context: str) -> None:
        """Set context label for current episode (e.g., 'kitchen', 'office').
        
        Args:
            context: Context identifier for the current episode.
        """
        self.current_episode_context = context
    
    def end_episode(self) -> None:
        """End current episode and consolidate observations into relational graph.
        
        This builds relations between all objects observed in the episode:
        - Co-occurrence relations
        - Spatial relations (relative positions)
        - Temporal relations (observation order)
        
        Also stores episode in history for potential replay if enabled.
        """
        if not self.current_episode_observations:
            logger.debug("No observations in episode, skipping consolidation")
            return
        
        n_objects = len(self.current_episode_observations)
        logger.info(
            f"Consolidating episode with {n_objects} object observations"
        )
        
        # Store episode in history if replay is enabled
        if self.enable_replay:
            self.graph_memory.store_episode(
                self.current_episode_observations,
                context=self.current_episode_context,
            )
            
            # Prune old episodes if history exceeds max size
            if len(self.graph_memory.episode_history) > self.max_episode_history:
                self.graph_memory.episode_history.pop(0)
                logger.debug(
                    f"Pruned episode history to {self.max_episode_history} episodes"
                )
        
        # Build relations between all pairs of objects in the episode
        for i, obs_a in enumerate(self.current_episode_observations):
            for obs_b in self.current_episode_observations[i + 1:]:
                # Skip self-relations
                if obs_a.object_id == obs_b.object_id:
                    continue
                
                # Compute spatial displacement
                spatial_displacement = obs_b.location - obs_a.location
                
                # Compute temporal offset
                temporal_offset = obs_b.timestamp - obs_a.timestamp
                
                # Add to memory with timestamp
                self.graph_memory.add_relation(
                    object_a=obs_a.object_id,
                    object_b=obs_b.object_id,
                    spatial_displacement=spatial_displacement,
                    temporal_offset=temporal_offset,
                    context=self.current_episode_context,
                    timestamp=self.graph_memory.current_time,
                )
        
        # Update episode count and time
        self.graph_memory.episode_count += 1
        self.graph_memory.current_time += 1.0
        
        # Clear episode buffer
        self.current_episode_observations = []
        self.current_episode_context = None
        
        logger.debug(
            f"Episode consolidated. Total episodes: "
            f"{self.graph_memory.episode_count}"
        )
    
    def get_related_objects(
        self,
        object_id: str,
        min_count: int = 1,
    ) -> Dict[str, ObjectRelation]:
        """Get all objects that have co-occurred with the given object.
        
        Args:
            object_id: Object to query relations for.
            min_count: Minimum co-occurrence count to include.
        
        Returns:
            Dict mapping related object IDs to their ObjectRelation.
        """
        return self.graph_memory.get_related_objects(object_id, min_count)
    
    def get_relation(
        self,
        object_a: str,
        object_b: str,
    ) -> Optional[ObjectRelation]:
        """Get the relation between two specific objects.
        
        Args:
            object_a: First object ID.
            object_b: Second object ID.
        
        Returns:
            ObjectRelation if exists, None otherwise.
        """
        return self.graph_memory.get_relation(object_a, object_b)
    
    def predict_next_object(
        self,
        current_object: str,
        context: Optional[str] = None,
    ) -> Optional[str]:
        """Predict what object is likely to appear next.
        
        Args:
            current_object: Object currently being observed.
            context: Optional context to filter predictions.
        
        Returns:
            Most likely next object ID, or None if no data.
        """
        return self.graph_memory.predict_next_object(current_object, context)
    
    def get_most_common_neighbors(
        self,
        object_id: str,
        top_k: int = 5,
    ) -> List[Tuple[str, int]]:
        """Get the k most frequently co-occurring objects.
        
        Args:
            object_id: Object to query.
            top_k: Number of neighbors to return.
        
        Returns:
            List of (object_id, count) tuples, sorted by count descending.
        """
        return self.graph_memory.get_most_common_neighbors(object_id, top_k)
    
    def reset(self) -> None:
        """Reset current episode buffer (but keep long-term memory)."""
        self.current_episode_observations = []
        self.current_episode_context = None
        logger.debug("Reset episode buffer")
    
    def clear_memory(self) -> None:
        """Clear all long-term relational memory."""
        self.graph_memory.clear()
        self.current_episode_observations = []
        self.current_episode_context = None
        logger.info("Cleared all hippocampal memory")
    
    def register_replay_callback(self, callback: Callable[[ReplayBatch], None]) -> None:
        """Register a callback to receive replay batches.
        
        Callbacks are invoked when replay batches are generated.
        This allows downstream models to be trained on replayed data.
        
        Args:
            callback: Function taking ReplayBatch as argument.
        """
        self.replay_callbacks.append(callback)
        logger.debug(f"Registered replay callback: {callback}")
    
    def replay_episodes(
        self,
        num_replays: int = 1,
        context_filter: Optional[str] = None,
        invoke_callbacks: bool = True,
    ) -> List[ReplayBatch]:
        """Generate and optionally invoke replay for past episodes.
        
        Args:
            num_replays: Number of episodes to sample and replay.
            context_filter: If set, only replay episodes from this context.
            invoke_callbacks: Whether to invoke registered callbacks.
        
        Returns:
            List of generated ReplayBatch objects.
        """
        replays = self.graph_memory.generate_episode_replay(
            num_replays=num_replays,
            context_filter=context_filter,
        )
        
        logger.info(f"Generated {len(replays)} episode replays")
        
        if invoke_callbacks:
            for batch in replays:
                for callback in self.replay_callbacks:
                    callback(batch)
        
        return replays
    
    def replay_relations(
        self,
        num_replays: int = 1,
        top_k_relations: int = 10,
        temperature: float = 1.0,
        context_filter: Optional[str] = None,
        invoke_callbacks: bool = True,
    ) -> List[ReplayBatch]:
        """Generate and optionally invoke relation-centric replay.
        
        This samples high-frequency relations and creates mini-episodes,
        useful for reinforcement without full episode replay.
        
        Args:
            num_replays: Number of relation-centric batches to generate.
            top_k_relations: Consider only top-k relations by frequency.
            temperature: Sampling temperature (1.0=probabilistic, 0=greedy).
            context_filter: If set, prioritize relations from this context.
            invoke_callbacks: Whether to invoke registered callbacks.
        
        Returns:
            List of generated ReplayBatch objects.
        """
        replays = self.graph_memory.generate_relation_replay(
            num_replays=num_replays,
            top_k_relations=top_k_relations,
            temperature=temperature,
            context_filter=context_filter,
        )
        
        logger.info(
            f"Generated {len(replays)} relation-centric replays "
            f"(top {top_k_relations}, temp={temperature})"
        )
        
        if invoke_callbacks:
            for batch in replays:
                for callback in self.replay_callbacks:
                    callback(batch)
        
        return replays

    def retrieve_episodes_by_cue(
        self,
        cue_object_ids: List[str],
        context_filter: Optional[str] = None,
        top_k: int = 1,
        min_jaccard: float = 0.0,
    ) -> List[EpisodeRecord]:
        """Retrieve stored episodes that best match a partial cue."""
        return self.graph_memory.retrieve_episodes_by_cue(
            cue_object_ids=cue_object_ids,
            context_filter=context_filter,
            top_k=top_k,
            min_jaccard=min_jaccard,
        )

    def complete_from_cue(
        self,
        cue_object_ids: List[str],
        context_filter: Optional[str] = None,
        top_k_episodes: int = 5,
        top_k_objects: int = 5,
        min_jaccard: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Suggest likely missing objects given a partial cue."""
        return self.graph_memory.complete_from_cue(
            cue_object_ids=cue_object_ids,
            context_filter=context_filter,
            top_k_episodes=top_k_episodes,
            top_k_objects=top_k_objects,
            min_jaccard=min_jaccard,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about learned relations.
        
        Returns:
            Dict with statistics about objects, relations, and episodes.
        """
        stats = self.graph_memory.get_statistics()
        stats.update({
            "current_episode_observations": len(self.current_episode_observations),
            "learning_rate": self.graph_memory.learning_rate,
        })
        return stats
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"HippocampalGraphLM("
            f"objects={stats['n_objects']}, "
            f"relations={stats['n_relations']}, "
            f"episodes={stats['episode_count']})"
        )
