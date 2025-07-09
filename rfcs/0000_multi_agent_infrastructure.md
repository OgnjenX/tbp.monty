- Start Date: 2025-07-09
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Summary

This RFC proposes adding infrastructure to support multiple agents that move independently in Monty, while maintaining minimal changes to the existing codebase. The solution uses a compositional approach that treats the current single-agent Monty system as building blocks to construct multi-agent capabilities.

# Motivation

Currently, Monty's infrastructure only supports a single agent that moves around the scene, where that agent can be associated with multiple sensors and learning modules. We need support for multiple agents that move independently to enable:

1. **Hand-like surface agents** where each "finger" can move semi-independently
2. **Distant agents** that observe objects while saccading independently of surface agents
3. **Cross-modal coordination** where agents perceive the same location simultaneously for voting
4. **Biologically plausible multi-column architectures** with independent motor control

Example use cases include implementing cross-modal sensory guidance policies and enabling rich multi-agent sensorimotor experiences.

# Guide-level explanation

## Core Principle: Compositional Architecture

Instead of modifying the existing Monty architecture internally, we propose creating a **higher-level orchestrator** that manages multiple independent Monty instances, each controlling a single agent.

```
Current: MontyExperiment → Monty → (SensorModules, LearningModules, MotorSystem) → Environment
Proposed: MultiAgentExperiment → MultiAgentMonty → [AgentUnit1, AgentUnit2, ...] → SharedEnvironment
```

## Key Components

### 1. AgentUnit (Zero Changes to Existing Code)

Each "Agent Unit" is the existing Monty system unchanged:
- One Monty instance (existing class)
- One agent with multiple sensor modules
- Multiple learning modules
- One motor system for that agent

```python
class AgentUnit:
    def __init__(self, monty_instance, agent_id):
        self.monty = monty_instance  # Existing Monty class - unchanged
        self.agent_id = agent_id
        
    def step(self, observation):
        return self.monty.step(observation)  # Delegate to existing Monty
```

### 2. MultiAgentMonty (New Orchestrator)

```python
class MultiAgentMonty:
    """Orchestrates multiple independent Monty systems."""
    
    def __init__(self, agent_units: Dict[str, AgentUnit]):
        self.agent_units = agent_units
        self.spatial_transformer = SpatialTransformer()
        
    def step(self, multi_agent_observation):
        # 1. Extract observations for each agent
        agent_observations = self._split_observation(multi_agent_observation)
        
        # 2. Step each agent independently (existing Monty.step)
        agent_actions = {}
        for agent_id, agent_unit in self.agent_units.items():
            agent_unit.step(agent_observations[agent_id])
            agent_actions[agent_id] = agent_unit.monty.motor_system.last_action
            
        # 3. Cross-agent communication
        self._cross_agent_communication()
        
        return agent_actions
```

### 3. Cross-Agent Communication (Wrapper Pattern)

To enable cross-agent voting without modifying the existing CMP, we use a wrapper pattern:

```python
class CrossAgentLearningModule:
    """Wrapper around existing LearningModule for cross-agent communication."""
    
    def __init__(self, base_lm, agent_id, spatial_transformer):
        self.base_lm = base_lm  # Existing LM - unchanged!
        self.agent_id = agent_id
        self.spatial_transformer = spatial_transformer
        
    def receive_votes(self, votes, source_agent_id=None):
        if source_agent_id and source_agent_id != self.agent_id:
            # Transform cross-agent votes with spatial transformation
            votes = self.spatial_transformer.transform(votes, source_agent_id, self.agent_id)
        
        # Delegate to existing LM
        return self.base_lm.receive_votes(votes)  # EXISTING method!
        
    def send_out_vote(self):
        return self.base_lm.send_out_vote()  # EXISTING method!
        
    # All other methods delegate to base_lm
    def __getattr__(self, name):
        return getattr(self.base_lm, name)
```

### 4. MultiAgentEnvironment (Minimal Extension)

```python
class MultiAgentEnvironment:
    """Wraps existing environment to handle multiple agents."""
    
    def __init__(self, base_environment, agent_configs):
        self.env = base_environment  # Existing HabitatEnvironment
        self.agent_configs = agent_configs
        
    def step(self, multi_agent_actions):
        # Apply each agent's action to the environment
        combined_observation = {}
        for agent_id, action in multi_agent_actions.items():
            obs = self.env.step(action)  # Existing environment step!
            combined_observation[agent_id] = obs
        return combined_observation
```

## Configuration Example

Multi-agent configurations are simply multiple single-agent configurations:

```python
multi_agent_config = {
    "agents": {
        "surface_agent": {
            # Existing single-agent config format
            "monty_class": MontyForEvidenceGraphMatching,
            "sensor_module_configs": {...},
            "learning_module_configs": {...},
            "motor_system_config": {...},
            "sm_to_agent_dict": {"patch": "surface_agent"},
        },
        "distant_agent": {
            # Existing single-agent config format  
            "monty_class": MontyForEvidenceGraphMatching,
            "sensor_module_configs": {...},
            "learning_module_configs": {...},
            "motor_system_config": {...},
            "sm_to_agent_dict": {"patch": "distant_agent"},
        }
    }
}
```

# Reference-level explanation

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `MultiAgentMonty` orchestrator class
2. Create `CrossAgentLearningModule` wrapper for cross-agent communication
3. Create spatial transformation utilities for cross-agent voting
4. Extend environment to handle multi-agent actions
5. Create `MultiAgentMontyExperiment` wrapper

### Phase 2: Enhanced Coordination
1. Add sophisticated cross-agent coordination strategies
2. Add agent synchronization mechanisms
3. Add multi-agent specific logging and analysis tools

### Phase 3: Optimization
1. Parallel agent processing capabilities
2. Performance optimizations for large numbers of agents

# Drawbacks

1. **Memory Usage**: Higher memory usage due to multiple Monty instances
2. **Performance Overhead**: Slight orchestration overhead compared to internal approach
3. **Complexity**: Adds orchestration layer complexity

# Rationale and alternatives

## Alternative Approaches Considered

### Option 1: Internal Multi-Agent Architecture
Modify core Monty classes to handle multiple agents internally by replacing the single MotorSystem with a MultiAgentMotorSystem and enhancing learning modules for cross-agent voting.

**Pros:** More efficient, lower memory usage
**Cons:** Significant changes to core code, higher development risk, backward compatibility concerns

### Option 2: Pure Orchestration (Zero CMP Changes)
Handle all cross-agent communication purely in the orchestration layer without any wrapper pattern.

**Pros:** Absolutely zero changes to existing code
**Cons:** Learning modules unaware of vote sources, less efficient vote handling, spatial transformations outside CMP

### Option 3: Minimal CMP Extensions
Add optional parameters to existing LearningModule methods for cross-agent awareness.

**Pros:** More efficient, proper spatial handling
**Cons:** Requires modifications to existing LM interface

**Recommendation:** We propose the wrapper pattern (similar to Option 3) as it provides proper cross-agent communication without modifying existing CMP code.

## Why This Design is Proposed

1. **Minimal Code Changes**: ~95% of existing code unchanged
2. **Backward Compatibility**: Single-agent experiments work exactly as before
3. **Reuse Existing Infrastructure**: All existing sensor modules, learning modules, policies work as-is
4. **Gradual Migration**: Can implement incrementally
5. **Easy Testing**: Each agent unit can be tested independently using existing tests
6  **Flexibility**: Supports various agent configurations and coordination strategies

# Prior art and references

Multi-agent systems have been extensively studied in robotics and AI:

- **Multi-Robot Systems**: Coordination algorithms for robot swarms and multi-robot exploration
- **Distributed Sensing**: Sensor networks with multiple independent sensing nodes

The proposed approach draws inspiration from these fields while maintaining compatibility with Monty's existing architecture.

# Unresolved questions

1. Should we support dynamic addition/removal of agents during episodes?
2. How should we handle agent failures or disconnections?
3. What coordination strategies should be built-in vs. configurable?
4. Should we eventually migrate to the internal approach for performance?

The compositional approach also allows for future migration to more integrated architectures if performance requirements demand it, while preserving the investment in multi-agent coordination algorithms.
