"""
RSTM-CBG Algorithm for hmp2g-dca
Role Switching and Task Migration based on Cross-layer Bridging Game

HIERARCHICAL DESIGN:
- Upper Layer: RSTM-CBG optimization for role assignment
- Lower Layer: Role-based behavior rules for action generation

This matches the paper's approach: RSTM-CBG decides WHAT roles agents should have,
and behavior rules determine HOW they execute those roles.
"""
import numpy as np
from ALGORITHM.common.alg_base import AlgorithmBase
from config import GlobalConfig
from UTIL.colorful import *

class AlgorithmConfig:
    """
    Algorithm configuration class that can be overridden by JSONC settings
    """
    # RSTM-CBG optimization parameters
    enable_optimization = True
    optimization_interval = 10
    role_update_frequency = 20

    # Network parameters
    k_hop_neighborhood = 2  # k-hop neighborhood for Shapley value
    monte_carlo_samples = 50  # Number of Monte Carlo samples

    # Bridging-aware distance parameter
    bridging_preference_delta = 1.0

    # Suitability function weights
    omega_compatibility = 0.5
    omega_migration_cost = 0.3
    omega_load_balance = 0.2

class Role:
    """Role types for agents"""
    SCOUT = 'scout'
    ATTACKER = 'attacker'
    DEFENDER = 'defender'

    @staticmethod
    def get_behavior_params(role):
        """
        Get behavior parameters for a given role

        Returns:
            dict: Behavior parameters including:
                - engagement_distance: Preferred distance to center
                - movement_strategy: How to move (aggressive/conservative/patrol)
                - rotation_frequency: How often to rotate (0-1)
                - fire_threshold: When to fire
        """
        params = {
            Role.SCOUT: {
                'engagement_distance': 2.5,  # Closer, more aggressive scout
                'movement_strategy': 'aggressive_patrol',  # Aggressive patrol
                'rotation_frequency': 0.5,
                'fire_threshold': 0.3,  # Fire more readily
                'description': 'Scout: Aggressive patrol and engage'
            },
            Role.ATTACKER: {
                'engagement_distance': 0.3,  # Very close to center, highly aggressive
                'movement_strategy': 'hyper_aggressive',  # Super aggressive
                'rotation_frequency': 0.95,  # Rapid targeting
                'fire_threshold': 0.1,  # Always fire
                'description': 'Attacker: Hyper-aggressive engagement'
            },
            Role.DEFENDER: {
                'engagement_distance': 0.3,  # Close to center, defend aggressively
                'movement_strategy': 'active_defense',  # Active defense
                'rotation_frequency': 0.95,  # Rapid scanning
                'fire_threshold': 0.1,  # Always fire
                'description': 'Defender: Active defense and fire'
            }
        }
        return params.get(role, params[Role.SCOUT])

class MultiLayerNetwork:
    """
    Simplified multi-layer network model for RSTM-CBG

    In DCA context, we model 3 layers:
    - Layer 0: Communication layer (based on proximity)
    - Layer 1: Coordination layer (based on task proximity)
    - Layer 2: Fire Control layer (based on line-of-sight)
    """

    def __init__(self, num_agents, config):
        self.num_agents = num_agents
        self.config = config
        self.layers = {}
        self.agent_layers = {}  # Which layers each agent belongs to

    def build_network(self, agents_info):
        """
        Build multi-layer network based on agent information

        Args:
            agents_info: Dictionary {agent_id: info_dict}
        """
        import networkx as nx

        center_pos = np.array([0.0, 0.0])

        # Layer 0: Communication layer (proximity-based)
        comm_graph = nx.Graph()
        for i in agents_info:
            comm_graph.add_node(i)
            for j in agents_info:
                if i < j:
                    pos_i = agents_info[i]['pos']
                    pos_j = agents_info[j]['pos']
                    dist = np.linalg.norm(pos_i - pos_j)
                    # Connect if within communication range
                    if dist < 2.5:
                        comm_graph.add_edge(i, j, weight=dist)
        self.layers[0] = comm_graph

        # Layer 1: Coordination layer (task proximity)
        coord_graph = nx.Graph()
        for i in agents_info:
            coord_graph.add_node(i)
            for j in agents_info:
                if i < j:
                    pos_i = agents_info[i]['pos']
                    pos_j = agents_info[j]['pos']
                    dist_i = np.linalg.norm(pos_i - center_pos)
                    dist_j = np.linalg.norm(pos_j - center_pos)
                    # Connect if working on similar regions
                    if abs(dist_i - dist_j) < 1.5:
                        coord_graph.add_edge(i, j)
        self.layers[1] = coord_graph

        # Layer 2: Fire Control layer (line-of-sight to center)
        fire_graph = nx.Graph()
        for i in agents_info:
            fire_graph.add_node(i)
            for j in agents_info:
                if i < j:
                    # Connect if they have overlapping fields of fire
                    pos_i = agents_info[i]['pos']
                    pos_j = agents_info[j]['pos']
                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist < 2.0:  # Overlapping fire zones
                        fire_graph.add_edge(i, j)
        self.layers[2] = fire_graph

        # Determine which layers each agent belongs to
        for i in agents_info:
            self.agent_layers[i] = [0, 1, 2]  # All agents in all layers (simplified)

    def get_degree_centrality(self, agent_id, layer_id):
        """Get degree centrality of an agent in a specific layer"""
        if layer_id not in self.layers:
            return 0.0
        layer = self.layers[layer_id]
        if agent_id in layer.nodes():
            return layer.degree(agent_id)
        return 0.0

class RSTM_CBG_Algorithm(AlgorithmBase):
    """
    RSTM-CBG Algorithm implementation for hmp2g-dca

    HIERARCHICAL DESIGN:
    ┌─────────────────────────────────────────────────┐
    │         RSTM-CBG (上层规划层)                      │
    │  • 计算跨层桥接度（CBD）                           │
    │  • 基于Shapley值进行角色决策                        │
    │  输出：每个智能体的角色                             │
    └─────────────────┬───────────────────────────────┘
                      │ 角色分配
                      ▼
    ┌─────────────────────────────────────────────────┐
    │       下层执行规则 (角色行为控制器)                 │
    │  • Scout: 侦察行为模式                            │
    │  • Attacker: 进攻行为模式                         │
    │  • Defender: 防守行为模式                         │
    │  输出：具体动作                                    │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        super().__init__(n_agent, n_thread, space, mcv, team)

        # Get scenario config
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        self.n_actions = self.ScenarioConfig.n_actions  # Should be 7 for DCA

        # Algorithm parameters
        self.num_guards = self.ScenarioConfig.num_guards
        self.num_attackers = self.ScenarioConfig.num_attackers

        # Initialize role assignments (default: all scouts)
        self.current_roles = {i: Role.SCOUT for i in range(self.num_guards)}

        # Initialize multi-layer network
        self.network = MultiLayerNetwork(self.num_guards, AlgorithmConfig)

        # Statistics
        self.step_count = 0
        self.role_update_counter = 0

        print亮蓝(f"[RSTM-CBG] Initialized with {self.num_guards} guards, {self.num_attackers} attackers")
        print亮蓝(f"[RSTM-CBG] Hierarchical mode: Upper=RSTM-CBG optimization, Lower=Role-based rules")

    def interact_with_env(self, team_intel):
        """
        Main algorithm interface - called each step

        Args:
            team_intel: Dictionary containing:
                - 'Latest-Obs': Observations array of shape (n_thread, n_agent, obs_dim)
                - 'Test-Flag': Whether in test mode
                - 'Latest-Reward': Rewards
                - 'Latest-Done': Done flags

        Returns:
            (actions, team_intel) tuple
            - actions: numpy array of shape (n_agent, n_thread)
        """
        obs = team_intel['Latest-Obs']
        test_mode = team_intel.get('Test-Flag', True)

        # Initialize actions array: shape (n_thread, n_agent)
        actions = np.zeros(shape=(self.n_thread, self.n_agent), dtype=int) - 1

        # Process each thread's observations
        for thread_id in range(len(obs)):
            thread_obs = obs[thread_id]

            # Extract agent information from observations
            agents_info = self._extract_agents_info(thread_obs)

            # Build multi-layer network
            self.network.build_network(agents_info)

            # UPPER LAYER: Run RSTM-CBG role assignment optimization
            # Update roles periodically (every role_update_frequency steps)
            should_update = (self.step_count % AlgorithmConfig.role_update_frequency == 0 or
                             self.step_count == 0)

            if should_update:
                self.current_roles = self._run_rstm_optimization(agents_info)

            # LOWER LAYER: Generate actions based on roles using behavior rules
            for agent_id in range(self.num_guards):
                if agent_id in agents_info:
                    role = self.current_roles.get(agent_id, Role.SCOUT)
                    action = self._generate_action_by_role(agent_id, agents_info[agent_id], role)
                    if isinstance(action, np.ndarray):
                        actions[thread_id, agent_id] = int(action.item()) if action.ndim > 0 else int(action)
                    else:
                        actions[thread_id, agent_id] = int(action)

        self.step_count += 1

        # Swap axes: (n_thread, n_agent) -> (n_agent, n_thread)
        actions = np.swapaxes(actions, 0, 1)

        return actions, team_intel

    def _extract_agents_info(self, thread_obs):
        """Extract agent information from observation array"""
        agents_info = {}

        # thread_obs is (n_agent, obs_dim) array
        for i, agent_obs in enumerate(thread_obs):
            # Convert to numpy array if not already
            if not isinstance(agent_obs, np.ndarray):
                agent_obs = np.array(agent_obs)

            # Handle both flattened and nested observation formats
            if agent_obs.ndim == 1 and len(agent_obs) >= 8:
                agents_info[i] = {
                    'alive': bool(agent_obs[0].item() if hasattr(agent_obs[0], 'item') else agent_obs[0]),
                    'pos': agent_obs[1:3],
                    'atk_rad': float(agent_obs[3].item() if hasattr(agent_obs[3], 'item') else agent_obs[3]),
                    'vel': agent_obs[4:6],
                    'iden': int(agent_obs[6].item() if hasattr(agent_obs[6], 'item') else agent_obs[6]),
                    'terrain': float(agent_obs[7].item() if hasattr(agent_obs[7], 'item') else agent_obs[7])
                }

        return agents_info

    def _run_rstm_optimization(self, agents_info):
        """
        Run RSTM-CBG optimization for role assignment (UPPER LAYER)

        This implements:
        1. Cross-layer bridging node identification (CBD calculation)
        2. Role decision based on CBD scores

        Args:
            agents_info: Dictionary {agent_id: info_dict}

        Returns:
            Dictionary {agent_id: role}
        """
        role_assignment = {}

        # Step 1: Calculate CBD for each agent
        cbd_scores = {}
        for agent_id in agents_info:
            cbd = self._calculate_cbd(agent_id, agents_info)
            cbd_scores[agent_id] = cbd

        # Sort agents by CBD (descending) - prioritize high-CBD agents
        sorted_agents = sorted(cbd_scores.keys(), key=lambda x: cbd_scores[x], reverse=True)

        # Step 2: Assign roles based on CBD scores
        # Higher CBD -> more critical inner roles
        n_defenders = max(2, self.num_guards // 4)
        n_attackers = max(2, self.num_guards // 3)

        for i, agent_id in enumerate(sorted_agents):
            if i < n_defenders:
                # High CBD + central role -> Defender
                role_assignment[agent_id] = Role.DEFENDER
            elif i < n_defenders + n_attackers:
                # Medium CBD -> Attacker
                role_assignment[agent_id] = Role.ATTACKER
            else:
                # Lower CBD -> Scout
                role_assignment[agent_id] = Role.SCOUT

        # Print role distribution periodically
        self.role_update_counter += 1
        if self.role_update_counter <= 3 or self.role_update_counter % 20 == 0:
            role_counts = {}
            for role in role_assignment.values():
                role_counts[role] = role_counts.get(role, 0) + 1
            print亮蓝(f"[RSTM-CBG] Role update #{self.role_update_counter}: {role_counts}")
            if self.role_update_counter <= 3:
                print亮蓝(f"[RSTM-CBG] CBD scores (first 5): {[(k, f'{cbd_scores[k]:.2f}') for k in list(sorted_agents)[:5]]}")

        return role_assignment

    def _calculate_cbd(self, agent_id, agents_info):
        """
        Calculate Cross-layer Bridging Degree (CBD) for an agent

        CBD(v_i) = |L_i| × Σ(deg_l(v_i)/|V_l|) × (|R_i|/max|R_j|)

        Args:
            agent_id: ID of the agent
            agents_info: Dictionary of agent information

        Returns:
            float: CBD score
        """
        # Factor 1: Number of layers (simplified - always 3 in DCA)
        layer_count = 3

        # Factor 2: Connection strength in each layer (normalized)
        connection_strength = 0.0
        for layer_id in range(3):
            deg = self.network.get_degree_centrality(agent_id, layer_id)
            connection_strength += deg / max(1, self.num_guards)

        # Factor 3: Role flexibility (simplified - all agents have all roles)
        role_flexibility = 1.0

        cbd = layer_count * connection_strength * role_flexibility
        return cbd

    def _generate_action_by_role(self, agent_id, agent_info, role):
        """
        Generate action based on role using behavior rules (LOWER LAYER)
        HIERARCHICAL VERSION - Role-based tactical behaviors

        Args:
            agent_id: ID of the agent
            agent_info: Information dictionary for this agent
            role: Assigned role (Role.SCOUT/ATTACKER/DEFENDER)

        Returns:
            np.array: Action (0-6)
        """
        # Get behavior parameters for this role
        params = Role.get_behavior_params(role)

        # Extract agent state
        alive = agent_info.get('alive', True)
        pos = agent_info.get('pos', np.array([0.0, 0.0]))
        vel = agent_info.get('vel', np.array([0.0, 0.0]))

        # Default action: stay and fire (action 0 = fire)
        action = 0

        if not alive:
            return np.array([action])

        center_pos = np.array([0.0, 0.0])
        dist_to_center = np.linalg.norm(pos - center_pos)
        target_distance = params['engagement_distance']

        # Role-specific behavior
        if role == Role.SCOUT:
            # Scout: Aggressive patrol at medium distance
            if dist_to_center > target_distance + 0.5:
                # Move toward center
                if abs(pos[0]) > abs(pos[1]):
                    action = 1 if pos[0] > 0 else 3
                else:
                    action = 4 if pos[1] > 0 else 2
            elif dist_to_center < target_distance - 0.5:
                # Move away from center slightly
                if abs(pos[0]) > abs(pos[1]):
                    action = 3 if pos[0] > 0 else 1
                else:
                    action = 2 if pos[1] > 0 else 4
            else:
                # At patrol distance - scan and fire
                if np.random.random() < params['fire_threshold']:
                    action = 0  # Fire
                else:
                    action = np.random.choice([5, 6])  # Rotate to scan

        elif role == Role.ATTACKER:
            # Attacker: Hyper-aggressive, move to center and engage
            if dist_to_center > 0.3:
                # Aggressively move to center
                if abs(pos[0]) > abs(pos[1]):
                    action = 1 if pos[0] > 0 else 3
                else:
                    action = 4 if pos[1] > 0 else 2
            else:
                # At center - rapid scan and fire
                if np.random.random() < params['fire_threshold']:
                    action = 0  # Fire
                else:
                    action = np.random.choice([5, 6])  # Quick scan

        elif role == Role.DEFENDER:
            # Defender: Active defense, hold position near center
            if dist_to_center > 0.5:
                # Move toward center to defend
                if abs(pos[0]) > abs(pos[1]):
                    action = 1 if pos[0] > 0 else 3
                else:
                    action = 4 if pos[1] > 0 else 2
            else:
                # Hold position and scan/fire
                if np.random.random() < params['fire_threshold']:
                    action = 0  # Fire
                else:
                    action = np.random.choice([5, 6])  # Scan

        return np.array([action])

    def _get_role_color(self, role):
        """Get color for role-based visualization"""
        colors = {
            'scout': 'cyan',
            'attacker': 'blue',
            'defender': 'green'
        }
        return colors.get(role, 'blue')
