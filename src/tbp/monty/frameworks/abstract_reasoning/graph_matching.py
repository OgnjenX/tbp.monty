# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from tbp.monty.frameworks.abstract_reasoning.abstract_reference_frames import (
    AbstractReferenceFrame,
    ABSTRACT_FRAME_REGISTRY,
)
from tbp.monty.frameworks.abstract_reasoning.concept_embeddings import (
    ConceptEmbedding,
    CONCEPT_EMBEDDING_REGISTRY,
)
from tbp.monty.frameworks.models.graph_matching import GraphLM, GraphMemory
from tbp.monty.frameworks.models.states import State, GoalState
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_scaled_evidences,
    get_custom_distances,
)


class AbstractGraphMatcher:
    """Specialized graph matching for abstract reasoning domains.
    
    This class provides methods for matching conceptual graphs in abstract reasoning
    domains such as philosophy, mathematics, and physics.
    """
    
    def __init__(
        self,
        reference_frame: Optional[Union[str, AbstractReferenceFrame]] = None,
        concept_embedding: Optional[Union[str, ConceptEmbedding]] = None,
        similarity_threshold: float = 0.7,
        max_graph_distance: float = 10.0,
        feature_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize the abstract graph matcher.
        
        Args:
            reference_frame: Reference frame ID or instance to use for matching
            concept_embedding: Concept embedding ID or instance to use
            similarity_threshold: Threshold for considering nodes similar
            max_graph_distance: Maximum distance between graphs to consider a match
            feature_weights: Weights for different features when calculating similarity
        """
        # Set up reference frame
        if reference_frame is None:
            self.reference_frame = None
        elif isinstance(reference_frame, str):
            if reference_frame in ABSTRACT_FRAME_REGISTRY:
                self.reference_frame = ABSTRACT_FRAME_REGISTRY[reference_frame]
            else:
                raise ValueError(f"Unknown reference frame: {reference_frame}")
        else:
            self.reference_frame = reference_frame
            
        # Set up concept embedding
        if concept_embedding is None:
            self.concept_embedding = None
        elif isinstance(concept_embedding, str):
            if concept_embedding in CONCEPT_EMBEDDING_REGISTRY:
                self.concept_embedding = CONCEPT_EMBEDDING_REGISTRY[concept_embedding]
            else:
                raise ValueError(f"Unknown concept embedding: {concept_embedding}")
        else:
            self.concept_embedding = concept_embedding
            
        self.similarity_threshold = similarity_threshold
        self.max_graph_distance = max_graph_distance
        self.feature_weights = feature_weights or {
            "semantic_vector": 1.0,
            "relational_structure": 0.8,
            "contextual_relevance": 0.6,
        }
        
    def match_graphs(
        self, 
        source_graph: Dict, 
        target_graph: Dict,
        transform_reference_frame: bool = True,
    ) -> Tuple[float, Dict]:
        """Match two abstract concept graphs and return similarity score.
        
        Args:
            source_graph: Source graph to match from
            target_graph: Target graph to match to
            transform_reference_frame: Whether to transform graphs to same reference frame
            
        Returns:
            Tuple of (similarity_score, node_mapping)
        """
        # Transform graphs to same reference frame if needed
        if transform_reference_frame and self.reference_frame is not None:
            source_graph = self.reference_frame.transform_graph(source_graph)
            target_graph = self.reference_frame.transform_graph(target_graph)
            
        # Extract nodes and edges from graphs
        source_nodes = source_graph.get("nodes", [])
        source_edges = source_graph.get("edges", [])
        target_nodes = target_graph.get("nodes", [])
        target_edges = target_graph.get("edges", [])
        
        if not source_nodes or not target_nodes:
            return 0.0, {}
            
        # Calculate node similarity matrix
        similarity_matrix = self._calculate_node_similarity_matrix(source_nodes, target_nodes)
        
        # Use Hungarian algorithm to find optimal node matching
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        # Calculate overall similarity score
        node_similarity = similarity_matrix[row_ind, col_ind].mean()
        
        # Calculate edge structure similarity if edges exist
        if source_edges and target_edges:
            edge_similarity = self._calculate_edge_similarity(
                source_edges, target_edges, dict(zip(row_ind, col_ind))
            )
        else:
            edge_similarity = 1.0 if not source_edges and not target_edges else 0.0
            
        # Combine node and edge similarity
        overall_similarity = 0.7 * node_similarity + 0.3 * edge_similarity
        
        # Create node mapping dictionary
        node_mapping = {source_nodes[i]["id"]: target_nodes[j]["id"] 
                       for i, j in zip(row_ind, col_ind) 
                       if similarity_matrix[i, j] >= self.similarity_threshold}
        
        return overall_similarity, node_mapping
    
    def find_subgraph_matches(
        self, 
        query_graph: Dict, 
        target_graph: Dict,
        min_similarity: float = 0.6,
    ) -> List[Dict]:
        """Find all subgraph matches of query_graph in target_graph.
        
        Args:
            query_graph: Query graph to find matches for
            target_graph: Target graph to search in
            min_similarity: Minimum similarity score to consider a match
            
        Returns:
            List of match dictionaries with similarity scores and node mappings
        """
        query_nodes = query_graph.get("nodes", [])
        target_nodes = target_graph.get("nodes", [])
        
        if not query_nodes or not target_nodes:
            return []
            
        # For each possible starting node in target graph
        matches = []
        for start_idx in range(len(target_nodes)):
            # Try to match query graph starting from this node
            similarity, mapping = self._match_from_node(
                query_graph, target_graph, start_idx
            )
            
            if similarity >= min_similarity:
                matches.append({
                    "similarity": similarity,
                    "mapping": mapping,
                    "start_node": target_nodes[start_idx]["id"]
                })
                
        # Sort matches by similarity score
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches
    
    def calculate_graph_similarity(
        self, 
        graph1: Dict, 
        graph2: Dict,
        normalize: bool = True,
    ) -> float:
        """Calculate overall similarity between two graphs.
        
        Args:
            graph1: First graph
            graph2: Second graph
            normalize: Whether to normalize the similarity score
            
        Returns:
            Similarity score between 0 and 1 if normalize=True
        """
        similarity, _ = self.match_graphs(graph1, graph2)
        
        if normalize:
            # Ensure similarity is between 0 and 1
            return max(0.0, min(1.0, similarity))
        
        return similarity
    
    def transform_state_to_graph(
        self, 
        state: State,
        reference_frame_id: Optional[str] = None,
    ) -> Dict:
        """Transform a State object to a graph representation.
        
        Args:
            state: State object to transform
            reference_frame_id: Optional reference frame to use
            
        Returns:
            Graph dictionary representation
        """
        # Get reference frame
        ref_frame = self.reference_frame
        if reference_frame_id is not None:
            if reference_frame_id in ABSTRACT_FRAME_REGISTRY:
                ref_frame = ABSTRACT_FRAME_REGISTRY[reference_frame_id]
            else:
                logging.warning(f"Unknown reference frame: {reference_frame_id}, using default")
        
        if ref_frame is None:
            logging.warning("No reference frame available for transformation")
            return self._default_state_to_graph(state)
            
        # Use reference frame to transform state
        return ref_frame.state_to_graph(state)
    
    def _default_state_to_graph(self, state: State) -> Dict:
        """Default implementation for transforming state to graph.
        
        Args:
            state: State object to transform
            
        Returns:
            Graph dictionary representation
        """
        graph = {"nodes": [], "edges": []}
        
        # Extract features from state
        features = state.get_features()
        
        # Create nodes for each feature
        for i, (feature_name, feature_value) in enumerate(features.items()):
            if isinstance(feature_value, (list, np.ndarray)) and len(feature_value) > 0:
                # Create a node for each element in array features
                for j, value in enumerate(feature_value):
                    node_id = f"{feature_name}_{i}_{j}"
                    graph["nodes"].append({
                        "id": node_id,
                        "type": feature_name,
                        "value": float(value) if isinstance(value, (int, float, np.number)) else str(value),
                        "position": [i, j, 0]  # Simple positional encoding
                    })
            else:
                # Create a single node for scalar features
                node_id = f"{feature_name}_{i}"
                graph["nodes"].append({
                    "id": node_id,
                    "type": feature_name,
                    "value": float(feature_value) if isinstance(feature_value, (int, float, np.number)) else str(feature_value),
                    "position": [i, 0, 0]  # Simple positional encoding
                })
        
        # Create basic edges between nodes
        for i in range(len(graph["nodes"])):
            for j in range(i+1, len(graph["nodes"])):
                graph["edges"].append({
                    "source": graph["nodes"][i]["id"],
                    "target": graph["nodes"][j]["id"],
                    "type": "related"
                })
        
        return graph
    
    def _calculate_node_similarity_matrix(
        self, 
        source_nodes: List[Dict], 
        target_nodes: List[Dict]
    ) -> np.ndarray:
        """Calculate similarity matrix between source and target nodes.
        
        Args:
            source_nodes: List of source nodes
            target_nodes: List of target nodes
            
        Returns:
            Similarity matrix of shape (len(source_nodes), len(target_nodes))
        """
        n_source = len(source_nodes)
        n_target = len(target_nodes)
        similarity_matrix = np.zeros((n_source, n_target))
        
        for i, source_node in enumerate(source_nodes):
            for j, target_node in enumerate(target_nodes):
                # Calculate feature similarity
                feature_sim = self._calculate_feature_similarity(source_node, target_node)
                
                # Calculate positional similarity if positions exist
                if "position" in source_node and "position" in target_node:
                    pos_sim = self._calculate_position_similarity(
                        source_node["position"], target_node["position"]
                    )
                    similarity_matrix[i, j] = 0.8 * feature_sim + 0.2 * pos_sim
                else:
                    similarity_matrix[i, j] = feature_sim
        
        return similarity_matrix
    
    def _calculate_feature_similarity(self, node1: Dict, node2: Dict) -> float:
        """Calculate similarity between node features.
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Similarity score between 0 and 1
        """
        # Check if nodes are of same type
        if node1.get("type") != node2.get("type"):
            return 0.3  # Different types have low similarity
        
        # If we have concept embeddings and semantic vectors, use them
        if (self.concept_embedding is not None and 
            "semantic_vector" in node1 and "semantic_vector" in node2):
            return self.concept_embedding.calculate_similarity(
                node1["semantic_vector"], node2["semantic_vector"]
            )
        
        # Otherwise, compare values directly
        if "value" in node1 and "value" in node2:
            if isinstance(node1["value"], (int, float)) and isinstance(node2["value"], (int, float)):
                # Numerical values - use normalized difference
                max_val = max(abs(node1["value"]), abs(node2["value"]))
                if max_val == 0:
                    return 1.0  # Both values are 0
                return 1.0 - min(1.0, abs(node1["value"] - node2["value"]) / max_val)
            else:
                # String values - use exact match
                return 1.0 if str(node1["value"]) == str(node2["value"]) else 0.0
        
        # Default similarity for nodes of same type
        return 0.5
    
    def _calculate_position_similarity(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate similarity between node positions.
        
        Args:
            pos1: Position of first node
            pos2: Position of second node
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to numpy arrays
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(pos1 - pos2)
        
        # Convert distance to similarity
        similarity = max(0.0, 1.0 - distance / self.max_graph_distance)
        
        return similarity
    
    def _calculate_edge_similarity(
        self, 
        source_edges: List[Dict], 
        target_edges: List[Dict],
        node_mapping: Dict
    ) -> float:
        """Calculate similarity between edge structures.
        
        Args:
            source_edges: List of source edges
            target_edges: List of target edges
            node_mapping: Mapping from source to target nodes
            
        Returns:
            Edge structure similarity score
        """
        # Count matched edges
        matched_edges = 0
        total_source_edges = len(source_edges)
        
        for source_edge in source_edges:
            source_start = source_edge["source"]
            source_end = source_edge["target"]
            
            # Skip if either node isn't in the mapping
            if source_start not in node_mapping or source_end not in node_mapping:
                continue
                
            target_start = node_mapping[source_start]
            target_end = node_mapping[source_end]
            
            # Check if corresponding edge exists in target
            for target_edge in target_edges:
                if ((target_edge["source"] == target_start and target_edge["target"] == target_end) or
                    (target_edge["source"] == target_end and target_edge["target"] == target_start)):
                    matched_edges += 1
                    break
        
        # Calculate edge similarity
        if total_source_edges == 0:
            return 1.0  # No edges to match
            
        return matched_edges / total_source_edges
    
    def _match_from_node(
        self, 
        query_graph: Dict, 
        target_graph: Dict, 
        start_idx: int
    ) -> Tuple[float, Dict]:
        """Match query graph to target graph starting from specific node.
        
        Args:
            query_graph: Query graph to match
            target_graph: Target graph to match to
            start_idx: Index of starting node in target graph
            
        Returns:
            Tuple of (similarity_score, node_mapping)
        """
        query_nodes = query_graph.get("nodes", [])
        target_nodes = target_graph.get("nodes", [])
        
        if not query_nodes:
            return 0.0, {}
            
        # Create subgraph from target centered around start_idx
        subgraph = self._extract_subgraph(target_graph, start_idx, len(query_nodes))
        
        # Match query to subgraph
        return self.match_graphs(query_graph, subgraph, transform_reference_frame=False)
    
    def _extract_subgraph(
        self, 
        graph: Dict, 
        center_idx: int, 
        size: int
    ) -> Dict:
        """Extract subgraph centered around specific node.
        
        Args:
            graph: Full graph to extract from
            center_idx: Index of center node
            size: Approximate number of nodes to include
            
        Returns:
            Subgraph dictionary
        """
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        if not nodes:
            return {"nodes": [], "edges": []}
            
        # Start with center node
        center_node = nodes[center_idx]
        included_nodes = [center_node]
        included_node_ids = {center_node["id"]}
        
        # Add connected nodes by BFS until we reach desired size
        queue = [center_node["id"]]
        while queue and len(included_nodes) < size:
            current_id = queue.pop(0)
            
            # Find all edges connected to current node
            for edge in edges:
                if edge["source"] == current_id:
                    target_id = edge["target"]
                    if target_id not in included_node_ids:
                        # Find the node with this ID
                        for node in nodes:
                            if node["id"] == target_id:
                                included_nodes.append(node)
                                included_node_ids.add(target_id)
                                queue.append(target_id)
                                break
                elif edge["target"] == current_id:
                    source_id = edge["source"]
                    if source_id not in included_node_ids:
                        # Find the node with this ID
                        for node in nodes:
                            if node["id"] == source_id:
                                included_nodes.append(node)
                                included_node_ids.add(source_id)
                                queue.append(source_id)
                                break
        
        # Extract relevant edges
        included_edges = [
            edge for edge in edges 
            if edge["source"] in included_node_ids and edge["target"] in included_node_ids
        ]
        
        return {
            "nodes": included_nodes,
            "edges": included_edges
        }


class AbstractGraphMemory(GraphMemory):
    """Specialized graph memory for abstract reasoning domains.
    
    Extends the base GraphMemory with capabilities for storing and retrieving
    abstract concept graphs.
    """
    
    def __init__(
        self,
        reference_frame_id: Optional[str] = None,
        concept_embedding_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize abstract graph memory.
        
        Args:
            reference_frame_id: ID of reference frame to use
            concept_embedding_id: ID of concept embedding to use
            **kwargs: Additional arguments for GraphMemory
        """
        super().__init__(**kwargs)
        
        # Set up reference frame
        self.reference_frame_id = reference_frame_id
        if reference_frame_id is not None and reference_frame_id in ABSTRACT_FRAME_REGISTRY:
            self.reference_frame = ABSTRACT_FRAME_REGISTRY[reference_frame_id]
        else:
            self.reference_frame = None
            
        # Set up concept embedding
        self.concept_embedding_id = concept_embedding_id
        if concept_embedding_id is not None and concept_embedding_id in CONCEPT_EMBEDDING_REGISTRY:
            self.concept_embedding = CONCEPT_EMBEDDING_REGISTRY[concept_embedding_id]
        else:
            self.concept_embedding = None
            
        # Initialize graph matcher
        self.graph_matcher = AbstractGraphMatcher(
            reference_frame=self.reference_frame,
            concept_embedding=self.concept_embedding
        )
        
        # Additional storage for abstract concept graphs
        self.concept_graphs = {}
        
    def add_concept_graph(self, concept_id: str, graph: Dict) -> None:
        """Add a concept graph to memory.
        
        Args:
            concept_id: Unique identifier for the concept
            graph: Graph representation of the concept
        """
        if self.reference_frame is not None:
            # Transform graph to standard reference frame
            graph = self.reference_frame.transform_graph(graph)
            
        self.concept_graphs[concept_id] = graph
        logging.info(f"Added concept graph '{concept_id}' to memory")
        
    def get_concept_graph(self, concept_id: str) -> Optional[Dict]:
        """Retrieve a concept graph from memory.
        
        Args:
            concept_id: ID of concept to retrieve
            
        Returns:
            Graph representation or None if not found
        """
        return self.concept_graphs.get(concept_id)
    
    def find_similar_concepts(
        self, 
        query_graph: Dict,
        threshold: float = 0.7,
        max_results: int = 5
    ) -> List[Tuple[str, float]]:
        """Find concepts similar to the query graph.
        
        Args:
            query_graph: Query graph to match
            threshold: Minimum similarity threshold
            max_results: Maximum number of results to return
            
        Returns:
            List of (concept_id, similarity) tuples
        """
        results = []
        
        for concept_id, graph in self.concept_graphs.items():
            similarity = self.graph_matcher.calculate_graph_similarity(query_graph, graph)
            if similarity >= threshold:
                results.append((concept_id, similarity))
                
        # Sort by similarity and limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def state_dict(self) -> Dict:
        """Get state dictionary for serialization.
        
        Returns:
            State dictionary
        """
        state = super().state_dict()
        state.update({
            "reference_frame_id": self.reference_frame_id,
            "concept_embedding_id": self.concept_embedding_id,
            "concept_graphs": self.concept_graphs
        })
        return state
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state from dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        super().load_state_dict(state_dict)
        
        # Load additional abstract reasoning components
        if "reference_frame_id" in state_dict:
            self.reference_frame_id = state_dict["reference_frame_id"]
            if self.reference_frame_id in ABSTRACT_FRAME_REGISTRY:
                self.reference_frame = ABSTRACT_FRAME_REGISTRY[self.reference_frame_id]
                
        if "concept_embedding_id" in state_dict:
            self.concept_embedding_id = state_dict["concept_embedding_id"]
            if self.concept_embedding_id in CONCEPT_EMBEDDING_REGISTRY:
                self.concept_embedding = CONCEPT_EMBEDDING_REGISTRY[self.concept_embedding_id]
                
        if "concept_graphs" in state_dict:
            self.concept_graphs = state_dict["concept_graphs"]
            
        # Reinitialize graph matcher with updated components
        self.graph_matcher = AbstractGraphMatcher(
            reference_frame=self.reference_frame,
            concept_embedding=self.concept_embedding
        )


class AbstractGraphLM(GraphLM):
    """Learning module for abstract graph matching.
    
    Extends the base GraphLM with specialized functionality for abstract reasoning
    domains, using the AbstractGraphMatcher and AbstractGraphMemory.
    """
    
    def __init__(
        self,
        learning_module_id: str,
        reasoning_domain: str,
        reference_frame_id: Optional[str] = None,
        concept_embedding_id: Optional[str] = None,
        similarity_threshold: float = 0.7,
        **kwargs
    ):
        """Initialize abstract graph learning module.
        
        Args:
            learning_module_id: Unique identifier for this learning module
            reasoning_domain: Abstract domain this module handles
            reference_frame_id: ID of reference frame to use
            concept_embedding_id: ID of concept embedding to use
            similarity_threshold: Threshold for considering graphs similar
            **kwargs: Additional arguments for GraphLM
        """
        super().__init__(initialize_base_modules=False, **kwargs)
        
        self.learning_module_id = learning_module_id
        self.reasoning_domain = reasoning_domain
        self.reference_frame_id = reference_frame_id
        self.concept_embedding_id = concept_embedding_id
        self.similarity_threshold = similarity_threshold
        
        # Initialize specialized memory
        self.graph_memory = AbstractGraphMemory(
            reference_frame_id=reference_frame_id,
            concept_embedding_id=concept_embedding_id,
            **kwargs.get("memory_args", {})
        )
        
        # Initialize graph matcher
        self.graph_matcher = AbstractGraphMatcher(
            reference_frame=self.graph_memory.reference_frame,
            concept_embedding=self.graph_memory.concept_embedding,
            similarity_threshold=similarity_threshold
        )
        
        # Track current state and matches
        self.current_state_graph = None
        self.current_matches = {}
        self.match_confidences = {}
        
    def matching_step(self, observations):
        """Update the possible matches given an observation.
        
        Args:
            observations: Observations to match against stored concepts
        """
        # Convert observations to state
        if isinstance(observations, State):
            state = observations
        else:
            # Create State from observations if needed
            state = State(observations)
            
        # Transform state to graph representation
        self.current_state_graph = self.graph_matcher.transform_state_to_graph(
            state, self.reference_frame_id
        )
        
        # Find matching concepts
        self._compute_possible_matches(self.current_state_graph)
        
        # Update buffer with processed observation
        self.buffer.add_observation(observations)
        self.add_lm_processing_to_buffer_stats(True)
        
    def _compute_possible_matches(self, query_graph, first_movement_detected=True):
        """Find concepts that match the query graph.
        
        Args:
            query_graph: Query graph to match
            first_movement_detected: Whether agent has moved (ignored in this implementation)
        """
        matches = {}
        confidences = {}
        
        # Match against each concept in memory
        for concept_id in self.graph_memory.concept_graphs:
            concept_graph = self.graph_memory.get_concept_graph(concept_id)
            
            # Calculate similarity and get node mapping
            similarity, mapping = self.graph_matcher.match_graphs(
                query_graph, concept_graph
            )
            
            if similarity >= self.similarity_threshold:
                matches[concept_id] = mapping
                confidences[concept_id] = similarity
                
        # Update current matches
        self.current_matches = matches
        self.match_confidences = confidences
        
        # Log results
        if matches:
            best_match = max(confidences.items(), key=lambda x: x[1])
            logging.info(f"Best match: {best_match[0]} with confidence {best_match[1]:.3f}")
        else:
            logging.info("No matching concepts found")
            
    def get_possible_matches(self):
        """Get list of current possible concept matches.
        
        Returns:
            List of concept IDs that match the current state
        """
        return list(self.current_matches.keys())
    
    def get_match_confidence(self, concept_id):
        """Get confidence score for a specific concept match.
        
        Args:
            concept_id: ID of concept to get confidence for
            
        Returns:
            Confidence score or 0.0 if not matched
        """
        return self.match_confidences.get(concept_id, 0.0)
    
    def get_best_match(self):
        """Get the best matching concept for current state.
        
        Returns:
            Tuple of (concept_id, confidence) or (None, 0.0) if no matches
        """
        if not self.match_confidences:
            return None, 0.0
            
        return max(self.match_confidences.items(), key=lambda x: x[1])
    
    def add_concept_from_state(self, state, concept_id):
        """Add a new concept to memory from a state.
        
        Args:
            state: State to create concept from
            concept_id: ID for the new concept
            
        Returns:
            True if concept was added successfully
        """
        # Transform state to graph
        graph = self.graph_matcher.transform_state_to_graph(
            state, self.reference_frame_id
        )
        
        # Add to memory
        self.graph_memory.add_concept_graph(concept_id, graph)
        return True
    
    def update_terminal_condition(self):
        """Check if we have reached a terminal condition.
        
        Updates the terminal_state attribute.
        """
        # Get best match and its confidence
        best_match, confidence = self.get_best_match()
        
        if best_match is None:
            self.terminal_state = "no_match"
        elif confidence >= self.similarity_threshold:
            self.detected_object = best_match
            self.terminal_state = "match"
        else:
            self.terminal_state = None
            
        return self.terminal_state
