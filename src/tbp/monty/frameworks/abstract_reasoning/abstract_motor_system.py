# abstract_motor_system.py

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.models.motor_policies import MotorPolicy
from tbp.monty.frameworks.models.motor_system import MotorSystem


class AbstractMotorAction(Action):
    """Base class for motor actions in abstract concept spaces.
    
    Represents movement/exploration actions in semantic/conceptual space,
    analogous to physical motor actions but operating on abstract concepts.
    """
    
    def __init__(self, action_type: str, parameters: Dict[str, Any]):
        """Initialize abstract motor action.
        
        Args:
            action_type: Type of abstract action (e.g., "semantic_step", "analogy_jump")
            parameters: Action-specific parameters
        """
        self.action_type = action_type
        self.parameters = parameters
        self.displacement = parameters.get('displacement', np.zeros(3))
        
    def __str__(self):
        return f"AbstractMotorAction({self.action_type}, {self.parameters})"


class SemanticStepAction(AbstractMotorAction):
    """Action representing a small step in semantic space."""
    
    def __init__(self, direction: np.ndarray, step_size: float = 0.1):
        """Initialize semantic step action.
        
        Args:
            direction: Direction vector in semantic space
            step_size: Size of the step
        """
        displacement = direction * step_size
        super().__init__(
            action_type="semantic_step",
            parameters={
                'direction': direction,
                'step_size': step_size,
                'displacement': displacement
            }
        )


class AnalogyJumpAction(AbstractMotorAction):
    """Action representing an analogical leap between concept domains."""
    
    def __init__(self, source_domain: str, target_domain: str, analogy_strength: float):
        """Initialize analogy jump action.
        
        Args:
            source_domain: Source conceptual domain
            target_domain: Target conceptual domain  
            analogy_strength: Strength of the analogical relationship
        """
        # Displacement based on analogy strength and domain difference
        displacement = np.random.normal(0, analogy_strength, 3)
        super().__init__(
            action_type="analogy_jump",
            parameters={
                'source_domain': source_domain,
                'target_domain': target_domain,
                'analogy_strength': analogy_strength,
                'displacement': displacement
            }
        )


class ConceptualExplorationAction(AbstractMotorAction):
    """Action for exploring around a concept to discover related concepts."""
    
    def __init__(self, exploration_radius: float = 0.5, exploration_angle: float = 0.0):
        """Initialize conceptual exploration action.
        
        Args:
            exploration_radius: How far to explore from current concept
            exploration_angle: Angle of exploration in concept space
        """
        # Convert polar coordinates to displacement
        displacement = np.array([
            exploration_radius * np.cos(exploration_angle),
            exploration_radius * np.sin(exploration_angle),
            0.0
        ])
        super().__init__(
            action_type="conceptual_exploration",
            parameters={
                'exploration_radius': exploration_radius,
                'exploration_angle': exploration_angle,
                'displacement': displacement
            }
        )


class AbstractMotorPolicy(MotorPolicy):
    """Motor policy for navigating abstract concept spaces.
    
    Implements sensorimotor learning principles for abstract reasoning,
    generating motor actions based on current conceptual state and goals.
    """
    
    def __init__(
        self,
        policy_id: str,
        exploration_probability: float = 0.3,
        step_size_range: Tuple[float, float] = (0.05, 0.2),
        max_analogy_distance: float = 2.0
    ):
        """Initialize abstract motor policy.
        
        Args:
            policy_id: Unique identifier for this policy
            exploration_probability: Probability of taking exploratory vs goal-directed action
            step_size_range: Range of step sizes for semantic steps
            max_analogy_distance: Maximum distance for analogy jumps
        """
        self.policy_id = policy_id
        self.exploration_probability = exploration_probability
        self.step_size_range = step_size_range
        self.max_analogy_distance = max_analogy_distance
        
        # Learning components
        self.action_history = []
        self.success_rates = {}  # action_type -> success_rate
        self.concept_transitions = {}  # (source, target) -> successful_actions
        
    def propose_action(
        self,
        current_concept: str,
        current_position: np.ndarray,
        goal_concept: Optional[str] = None,
        goal_position: Optional[np.ndarray] = None,
        nearby_concepts: List[Tuple[str, float]] = None
    ) -> AbstractMotorAction:
        """Propose next motor action based on current state and goals.
        
        Args:
            current_concept: Current concept identifier
            current_position: Current position in abstract space
            goal_concept: Target concept (if any)
            goal_position: Target position (if any)
            nearby_concepts: List of (concept_id, distance) tuples for nearby concepts
            
        Returns:
            Proposed abstract motor action
        """
        # Decide between exploration and goal-directed action
        if goal_concept is None or np.random.random() < self.exploration_probability:
            return self._propose_exploratory_action(current_concept, current_position, nearby_concepts)
        else:
            return self._propose_goal_directed_action(
                current_concept, current_position, goal_concept, goal_position
            )
    
    def _propose_exploratory_action(
        self,
        current_concept: str,
        current_position: np.ndarray,
        nearby_concepts: List[Tuple[str, float]]
    ) -> AbstractMotorAction:
        """Propose an exploratory action to discover new concepts."""
        action_type = np.random.choice(['semantic_step', 'conceptual_exploration'])
        
        if action_type == 'semantic_step':
            # Random direction with learned step size
            direction = np.random.normal(0, 1, 3)
            direction = direction / np.linalg.norm(direction)
            step_size = np.random.uniform(*self.step_size_range)
            return SemanticStepAction(direction, step_size)
        
        else:  # conceptual_exploration
            exploration_radius = np.random.uniform(0.1, 0.8)
            exploration_angle = np.random.uniform(0, 2 * np.pi)
            return ConceptualExplorationAction(exploration_radius, exploration_angle)
    
    def _propose_goal_directed_action(
        self,
        current_concept: str,
        current_position: np.ndarray,
        goal_concept: str,
        goal_position: np.ndarray
    ) -> AbstractMotorAction:
        """Propose action directed toward a specific goal."""
        # Calculate direction to goal
        direction_to_goal = goal_position - current_position
        distance_to_goal = np.linalg.norm(direction_to_goal)
        
        if distance_to_goal < 1e-6:
            # Already at goal, explore locally
            return ConceptualExplorationAction(0.1, np.random.uniform(0, 2 * np.pi))
        
        # Normalize direction
        direction_to_goal = direction_to_goal / distance_to_goal
        
        # Choose step size based on distance
        step_size = min(distance_to_goal * 0.5, self.step_size_range[1])
        step_size = max(step_size, self.step_size_range[0])
        
        return SemanticStepAction(direction_to_goal, step_size)
    
    def update_from_experience(
        self,
        action: AbstractMotorAction,
        success: bool,
        source_concept: str,
        target_concept: str
    ) -> None:
        """Update policy based on action outcome.
        
        Args:
            action: Action that was taken
            success: Whether the action was successful
            source_concept: Concept before action
            target_concept: Concept after action
        """
        # Record action
        self.action_history.append({
            'action': action,
            'success': success,
            'source': source_concept,
            'target': target_concept
        })
        
        # Update success rates
        action_type = action.action_type
        if action_type not in self.success_rates:
            self.success_rates[action_type] = {'successes': 0, 'total': 0}
        
        self.success_rates[action_type]['total'] += 1
        if success:
            self.success_rates[action_type]['successes'] += 1
        
        # Update concept transitions
        transition_key = (source_concept, target_concept)
        if transition_key not in self.concept_transitions:
            self.concept_transitions[transition_key] = []
        
        if success:
            self.concept_transitions[transition_key].append(action)
    
    def get_success_rate(self, action_type: str) -> float:
        """Get success rate for a specific action type."""
        if action_type not in self.success_rates:
            return 0.0
        
        stats = self.success_rates[action_type]
        if stats['total'] == 0:
            return 0.0
        
        return stats['successes'] / stats['total']


class AbstractMotorSystem(MotorSystem):
    """Motor system for abstract reasoning domains.
    
    Manages motor policies and coordinates movement in abstract concept spaces,
    implementing sensorimotor learning principles for abstract reasoning.
    """
    
    def __init__(self, motor_system_id: str = "abstract_motor_system"):
        """Initialize abstract motor system.
        
        Args:
            motor_system_id: Unique identifier for this motor system
        """
        self.motor_system_id = motor_system_id
        self.policies = {}  # policy_id -> AbstractMotorPolicy
        self.current_policy = None
        self.action_history = []
        
    def add_policy(self, policy: AbstractMotorPolicy) -> None:
        """Add a motor policy to the system."""
        self.policies[policy.policy_id] = policy
        if self.current_policy is None:
            self.current_policy = policy.policy_id
    
    def set_current_policy(self, policy_id: str) -> None:
        """Set the currently active policy."""
        if policy_id in self.policies:
            self.current_policy = policy_id
        else:
            raise ValueError(f"Policy {policy_id} not found")
    
    def propose_action(self, **kwargs) -> AbstractMotorAction:
        """Propose next motor action using current policy."""
        if self.current_policy is None:
            raise RuntimeError("No motor policy set")
        
        policy = self.policies[self.current_policy]
        action = policy.propose_action(**kwargs)
        self.action_history.append(action)
        return action
    
    def update_from_experience(self, action: AbstractMotorAction, success: bool, **kwargs) -> None:
        """Update motor system based on action outcome."""
        if self.current_policy is not None:
            policy = self.policies[self.current_policy]
            policy.update_from_experience(action, success, **kwargs)
