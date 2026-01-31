"""
RTM-RPF Algorithm for hmp2g-dca
Resilient Task Migration based on Risk Potential Field

ALGORITHM DESIGN:
- Phase 1: GD-RER (Greedy Decision based on Replenishment Efficiency Ratio)
- Phase 2: RPF-PP (Risk Potential Field based Path Planning)
- Phase 3: TA-RF (Task Allocation based on Resilient Fitness)

This implementation adapts the RTM-RPF algorithm to the DCA environment,
using the three-role system (Scout/Attacker/Defender) from RSTM-CBG.
"""
import numpy as np
from ALGORITHM.common.alg_base import AlgorithmBase
from config import GlobalConfig
from UTIL.colorful import *
import heapq
from collections import defaultdict

class AlgorithmConfig:
    """
    Algorithm configuration class that can be overridden by JSONC settings
    """
    # Failure model parameters
    alpha_risk = 0.8  # Maximum risk upper limit coefficient
    eta = 1.5  # Risk growth exponent
    gamma = 2.0  # Risk sensitivity exponent
    kappa_task = 0.5  # Task impact weight coefficient
    epsilon = 1e-6  # Numerical stability constant

    # Replenishment mechanism parameters
    eta_rer = 0.1  # Replenishment efficiency ratio threshold
    kappa_link = 0.1  # Unit link establishment cost coefficient
    L_max = 10.0  # Maximum load

    # Task allocation parameters (resilient fitness weights)
    alpha1 = 0.35  # Role-task fitness weight
    alpha2 = 0.25  # Survival rate weight
    alpha3 = 0.2  # Path risk weight
    alpha4 = 0.2  # Load weight

    # Update frequency
    role_update_frequency = 20
    failure_check_frequency = 10

    # Network parameters
    communication_range = 3.0  # Communication range for network topology
    coordination_range = 4.0  # Coordination range

class Role:
    """Role types for agents (same as RSTM-CBG)"""
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
                'engagement_distance': 2.5,
                'movement_strategy': 'aggressive_patrol',
                'rotation_frequency': 0.5,
                'fire_threshold': 0.3,
                'description': 'Scout: Aggressive patrol and engage'
            },
            Role.ATTACKER: {
                'engagement_distance': 0.3,
                'movement_strategy': 'hyper_aggressive',
                'rotation_frequency': 0.95,
                'fire_threshold': 0.1,
                'description': 'Attacker: Hyper-aggressive engagement'
            },
            Role.DEFENDER: {
                'engagement_distance': 0.3,
                'movement_strategy': 'active_defense',
                'rotation_frequency': 0.95,
                'fire_threshold': 0.1,
                'description': 'Defender: Active defense and fire'
            }
        }
        return params.get(role, params[Role.SCOUT])

class RTM_RPF_Algorithm(AlgorithmBase):
    """
    RTM-RPF Algorithm implementation for DCA environment

    Implements three core sub-algorithms:
    1. GD-RER: Greedy Decision based on Replenishment Efficiency Ratio
    2. RPF-PP: Risk Potential Field based Path Planning
    3. TA-RF: Task Allocation based on Resilient Fitness
    """

    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        super().__init__(n_agent, n_thread, space, mcv, team)

        # Get scenario configuration
        self.num_guards = self.ScenarioConfig.num_guards
        self.num_attackers = self.ScenarioConfig.num_attackers

        # Initialize role assignments (all start as scouts)
        self.current_roles = {i: Role.SCOUT for i in range(self.num_guards)}

        # Initialize failure tracking
        self.failed_agents = set()
        self.isolated_agents = set()
        self.agent_loads = {i: 0.0 for i in range(self.num_guards)}
        self.failure_probabilities = {i: 0.0 for i in range(self.num_guards)}

        # Initialize network topology
        self.network_adjacency = {}
        self.network_distances = {}

        # Initialize replenishment plan
        self.replenishment_plan = {}

        # Initialize RPD matrix (Risk Potential Distance)
        self.rpd_matrix = {}
        self.rpd_max = 1.0

        # Statistics
        self.step_count = 0
        self.role_update_counter = 0

        print亮蓝(f"[RTM-RPF] Initialized with {self.num_guards} guards, {self.num_attackers} attackers")
        print亮蓝(f"[RTM-RPF] Algorithm: GD-RER + RPF-PP + TA-RF")

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

            # Step 1: Extract agent information from observations
            agents_info = self._extract_agents_info(thread_obs)

            # Step 2: Build network topology
            self._build_network_topology(agents_info)

            # Step 3: Calculate failure probabilities
            self._calculate_failure_probabilities(agents_info)

            # Step 4: Execute failure detection (periodically)
            if self.step_count % AlgorithmConfig.failure_check_frequency == 0:
                self._execute_failure_detection(agents_info)

            # Step 5: Identify cascade failures
            self._identify_cascade_failures(agents_info)

            # Step 6: Execute GD-RER replenishment (if needed)
            if len(self.failed_agents) > 0:
                self._execute_gd_rer(agents_info)

            # Step 7: Execute RPF-PP path planning
            self._execute_rpf_pp(agents_info)

            # Step 8: Execute TA-RF task allocation (role assignment)
            if self.step_count % AlgorithmConfig.role_update_frequency == 0 or self.step_count == 0:
                self._execute_ta_rf(agents_info)

            # Step 9: Generate actions based on roles
            for agent_id in range(self.num_guards):
                if agent_id in agents_info and agents_info[agent_id]['alive']:
                    role = self.current_roles.get(agent_id, Role.SCOUT)
                    action = self._generate_action_by_role(agent_id, agents_info[agent_id], role, agents_info)
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

    def _build_network_topology(self, agents_info):
        """
        Build network topology based on agent distances
        Creates adjacency list and distance matrix
        """
        self.network_adjacency = {i: [] for i in agents_info.keys()}
        self.network_distances = {}

        agent_ids = list(agents_info.keys())
        for i in agent_ids:
            if not agents_info[i]['alive']:
                continue
            for j in agent_ids:
                if i >= j or not agents_info[j]['alive']:
                    continue

                # Calculate distance
                pos_i = agents_info[i]['pos']
                pos_j = agents_info[j]['pos']
                dist = np.linalg.norm(pos_i - pos_j)

                # Add edge if within communication range
                if dist <= AlgorithmConfig.communication_range:
                    self.network_adjacency[i].append(j)
                    self.network_adjacency[j].append(i)
                    self.network_distances[(i, j)] = dist
                    self.network_distances[(j, i)] = dist

    def _calculate_failure_probabilities(self, agents_info):
        """
        Calculate failure probability for each agent
        Formula: p_fail = 1 - π_i × (1 - μ_i^M)
        where:
        - π_i: Normal working probability (based on load, layer load)
        - μ_i^M: Multi-degree centrality exposure risk
        """
        for agent_id, info in agents_info.items():
            if not info['alive']:
                self.failure_probabilities[agent_id] = 1.0
                continue

            # Calculate load factor (normalized)
            load = self.agent_loads.get(agent_id, 0.0)
            load_factor = 1.0 - min(load / AlgorithmConfig.L_max, 1.0)

            # Calculate exposure risk based on network centrality
            degree = len(self.network_adjacency.get(agent_id, []))
            max_degree = max(len(self.network_adjacency.get(i, [])) for i in agents_info.keys())
            exposure_risk = degree / max(max_degree, 1.0)

            # Calculate terrain risk (higher terrain = higher risk)
            terrain_risk = info['terrain']

            # Calculate normal working probability π_i
            pi_i = load_factor * (1.0 - 0.3 * terrain_risk)

            # Calculate comprehensive failure probability
            p_fail = 1.0 - pi_i * (1.0 - exposure_risk)

            # Apply risk growth with exponent
            p_fail = min(AlgorithmConfig.alpha_risk * (p_fail ** AlgorithmConfig.eta), 1.0)

            self.failure_probabilities[agent_id] = p_fail

    def _execute_failure_detection(self, agents_info):
        """
        Execute Monte Carlo failure detection
        Agents with high failure probability may be marked as failed
        """
        for agent_id, info in agents_info.items():
            if not info['alive']:
                self.failed_agents.add(agent_id)
                continue

            # Monte Carlo sampling
            p_fail = self.failure_probabilities.get(agent_id, 0.0)
            if np.random.random() < p_fail * 0.1:  # Scale down for simulation
                self.failed_agents.add(agent_id)

    def _identify_cascade_failures(self, agents_info):
        """
        Identify cascade failures and isolated agents
        Uses BFS to find connected components
        """
        # Find connected components
        visited = set()
        components = []

        for agent_id in agents_info.keys():
            if agent_id in visited or agent_id in self.failed_agents:
                continue
            if not agents_info[agent_id]['alive']:
                continue

            # BFS to find component
            component = set()
            queue = [agent_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)

                # Add neighbors
                for neighbor in self.network_adjacency.get(current, []):
                    if neighbor not in visited and neighbor not in self.failed_agents:
                        if agents_info[neighbor]['alive']:
                            queue.append(neighbor)

            components.append(component)

        # Identify isolated agents (not in largest component)
        if components:
            largest_component = max(components, key=len)
            self.isolated_agents = set()
            for component in components:
                if component != largest_component:
                    self.isolated_agents.update(component)

    def _execute_gd_rer(self, agents_info):
        """
        Algorithm 3-1: Greedy Decision based on Replenishment Efficiency Ratio (GD-RER)

        Steps:
        1. Calculate failure impact degree (FID) for each failed agent
        2. Sort failed agents by FID in descending order
        3. For each failed agent, select optimal replenisher based on RER
        4. Execute replenishment (role switching)
        """
        # Calculate FID for failed agents
        fid_scores = {}
        for failed_id in self.failed_agents:
            # Count isolated agents caused by this failure
            isolated_count = len([a for a in self.isolated_agents
                                 if self._is_isolated_due_to(a, failed_id, agents_info)])

            # Count interrupted tasks (use load as proxy)
            interrupted_tasks = self.agent_loads.get(failed_id, 0.0)
            total_tasks = sum(self.agent_loads.values())

            # Calculate FID: |C_iso| × (1 + κ_task × |T_int|/|T|)
            fid = isolated_count * (1.0 + AlgorithmConfig.kappa_task *
                                   (interrupted_tasks / max(total_tasks, 1.0)))
            fid_scores[failed_id] = fid

        # Sort by FID (descending)
        sorted_failed = sorted(fid_scores.keys(), key=lambda x: fid_scores[x], reverse=True)

        # For each failed agent, find optimal replenisher
        available_agents = set(agents_info.keys()) - self.failed_agents
        available_agents = {a for a in available_agents if agents_info[a]['alive']}

        self.replenishment_plan = {}

        for failed_id in sorted_failed:
            if not available_agents:
                break

            best_replenisher = None
            best_rer = -float('inf')

            # Evaluate each candidate
            for candidate_id in available_agents:
                # Calculate replenishment cost (distance-based)
                if (failed_id, candidate_id) in self.network_distances:
                    c_rep = self.network_distances[(failed_id, candidate_id)]
                else:
                    pos_f = agents_info.get(failed_id, {}).get('pos', np.zeros(2))
                    pos_c = agents_info[candidate_id]['pos']
                    c_rep = np.linalg.norm(pos_f - pos_c)

                c_rep = max(c_rep * AlgorithmConfig.kappa_link, 0.1)

                # Calculate RER: [|C_iso| × (1 - L_j/L_max)] / [c_rep × (1 + α_risk × μ_j^M)]
                isolated_count = fid_scores[failed_id] / (1.0 + AlgorithmConfig.kappa_task)
                load_margin = 1.0 - (self.agent_loads.get(candidate_id, 0.0) / AlgorithmConfig.L_max)
                exposure_risk = self.failure_probabilities.get(candidate_id, 0.0)

                numerator = isolated_count * load_margin
                denominator = c_rep * (1.0 + AlgorithmConfig.alpha_risk * exposure_risk)

                rer = numerator / max(denominator, 0.01)

                if rer > best_rer:
                    best_rer = rer
                    best_replenisher = candidate_id

            # Execute replenishment if RER exceeds threshold
            if best_replenisher is not None and best_rer >= AlgorithmConfig.eta_rer:
                self.replenishment_plan[failed_id] = best_replenisher
                available_agents.remove(best_replenisher)

                # Transfer role and load
                if failed_id in self.current_roles:
                    self.current_roles[best_replenisher] = self.current_roles[failed_id]
                self.agent_loads[best_replenisher] += self.agent_loads.get(failed_id, 0.0)

    def _is_isolated_due_to(self, agent_id, failed_id, agents_info):
        """Check if agent is isolated due to specific failed agent"""
        # Simplified: check if failed agent was on path to main component
        return agent_id in self.isolated_agents

    def _execute_rpf_pp(self, agents_info):
        """
        Algorithm 3-2: Risk Potential Field based Path Planning (RPF-PP)

        Steps:
        1. Calculate risk potential U_i for each agent
        2. Build risk potential weight matrix w_ij^RPD
        3. Calculate risk potential distance (RPD) using Dijkstra
        """
        # Step 1: Calculate risk potential for each agent
        risk_potentials = {}
        for agent_id, info in agents_info.items():
            if not info['alive']:
                risk_potentials[agent_id] = float('inf')
                continue

            p_fail = self.failure_probabilities.get(agent_id, 0.0)
            # U_i = 1 / (max(ε, 1 - p_fail))^γ
            survival_prob = max(AlgorithmConfig.epsilon, 1.0 - p_fail)
            U_i = 1.0 / (survival_prob ** AlgorithmConfig.gamma)
            risk_potentials[agent_id] = U_i

        # Step 2: Build risk potential weight matrix
        rpd_weights = {}
        for (i, j), dist in self.network_distances.items():
            # w_ij^RPD = ω_ij × U_j
            U_j = risk_potentials.get(j, float('inf'))
            rpd_weights[(i, j)] = dist * U_j

        # Step 3: Calculate RPD using Dijkstra for all source agents
        self.rpd_matrix = {}
        alive_agents = [a for a in agents_info.keys() if agents_info[a]['alive']]

        for source in alive_agents:
            distances = self._dijkstra(source, alive_agents, rpd_weights)
            for target, dist in distances.items():
                self.rpd_matrix[(source, target)] = dist

        # Calculate RPD_max for normalization
        if self.rpd_matrix:
            finite_rpds = [d for d in self.rpd_matrix.values() if d < float('inf')]
            self.rpd_max = max(finite_rpds) if finite_rpds else 1.0
        else:
            self.rpd_max = 1.0

    def _dijkstra(self, source, nodes, weights):
        """
        Dijkstra's algorithm for shortest path

        Args:
            source: Source node
            nodes: List of all nodes
            weights: Dictionary of edge weights {(i,j): weight}

        Returns:
            Dictionary of distances from source to all nodes
        """
        distances = {node: float('inf') for node in nodes}
        distances[source] = 0.0

        # Priority queue: (distance, node)
        pq = [(0.0, source)]
        visited = set()

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current in visited:
                continue
            visited.add(current)

            # Check neighbors
            for neighbor in self.network_adjacency.get(current, []):
                if neighbor not in nodes:
                    continue

                edge_weight = weights.get((current, neighbor), float('inf'))
                new_dist = current_dist + edge_weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

        return distances

    def _execute_ta_rf(self, agents_info):
        """
        Algorithm 3-3: Task Allocation based on Resilient Fitness (TA-RF)

        In DCA context, "task allocation" means role assignment based on
        resilient fitness considering survival rate, path risk, and load balance.

        Steps:
        1. Calculate resilient fitness for each agent
        2. Assign roles based on fitness scores
        """
        # Calculate resilient fitness for each agent
        fitness_scores = {}

        for agent_id, info in agents_info.items():
            if not info['alive'] or agent_id in self.failed_agents:
                continue

            # Calculate fitness for each potential role
            role_fitness = {}
            for role in [Role.SCOUT, Role.ATTACKER, Role.DEFENDER]:
                fitness = self._compute_resilient_fitness(agent_id, role, agents_info)
                role_fitness[role] = fitness

            # Select best role
            best_role = max(role_fitness.keys(), key=lambda r: role_fitness[r])
            fitness_scores[agent_id] = (best_role, role_fitness[best_role])

        # Assign roles based on fitness
        # Ensure balanced distribution: 40% scouts, 30% attackers, 30% defenders
        sorted_agents = sorted(fitness_scores.keys(),
                              key=lambda x: fitness_scores[x][1], reverse=True)

        n_agents = len(sorted_agents)
        n_scouts = int(n_agents * 0.4)
        n_attackers = int(n_agents * 0.3)

        for i, agent_id in enumerate(sorted_agents):
            if i < n_scouts:
                self.current_roles[agent_id] = Role.SCOUT
            elif i < n_scouts + n_attackers:
                self.current_roles[agent_id] = Role.ATTACKER
            else:
                self.current_roles[agent_id] = Role.DEFENDER

    def _compute_resilient_fitness(self, agent_id, role, agents_info):
        """
        Compute resilient fitness function F_{i,k}^res

        Formula: F = α1×Φ + α2×(1-p_fail) - α3×RPD/RPD_max - α4×(L+q)/L_max

        Args:
            agent_id: Agent ID
            role: Candidate role
            agents_info: All agents information

        Returns:
            Resilient fitness score
        """
        # α1: Role-task fitness (role compatibility)
        # For DCA, we use position-based fitness
        pos = agents_info[agent_id]['pos']
        center = np.array([0.0, 0.0])  # Assume center is origin
        dist_to_center = np.linalg.norm(pos - center)

        role_params = Role.get_behavior_params(role)
        target_dist = role_params['engagement_distance']

        # Fitness based on distance match
        phi = 1.0 - min(abs(dist_to_center - target_dist) / 5.0, 1.0)

        # α2: Survival rate (1 - p_fail)
        p_fail = self.failure_probabilities.get(agent_id, 0.0)
        survival = 1.0 - p_fail

        # α3: Normalized path risk (RPD)
        # Use average RPD to other agents
        rpd_sum = 0.0
        rpd_count = 0
        for other_id in agents_info.keys():
            if other_id != agent_id and agents_info[other_id]['alive']:
                rpd = self.rpd_matrix.get((agent_id, other_id), self.rpd_max)
                if rpd < float('inf'):
                    rpd_sum += rpd
                    rpd_count += 1

        avg_rpd = rpd_sum / max(rpd_count, 1)
        normalized_rpd = avg_rpd / max(self.rpd_max, 1.0)

        # α4: Predicted load rate
        current_load = self.agent_loads.get(agent_id, 0.0)
        # Assume role change adds 1.0 load
        predicted_load = (current_load + 1.0) / AlgorithmConfig.L_max

        # Calculate resilient fitness
        F_res = (AlgorithmConfig.alpha1 * phi +
                AlgorithmConfig.alpha2 * survival -
                AlgorithmConfig.alpha3 * normalized_rpd -
                AlgorithmConfig.alpha4 * predicted_load)

        return F_res

    def _generate_action_by_role(self, agent_id, agent_info, role, agents_info):
        """
        Generate action based on role and risk potential field

        Args:
            agent_id: Agent ID
            agent_info: Agent information
            role: Assigned role
            agents_info: All agents information

        Returns:
            Action index
        """
        # Get role behavior parameters
        params = Role.get_behavior_params(role)

        # Get agent position
        pos = agent_info['pos']

        # Calculate target position based on role and risk field
        center = np.array([0.0, 0.0])
        target_dist = params['engagement_distance']

        # Direction to center
        to_center = center - pos
        dist_to_center = np.linalg.norm(to_center)

        if dist_to_center > 0.01:
            direction = to_center / dist_to_center
        else:
            direction = np.array([1.0, 0.0])

        # Adjust target based on risk potential field
        # Move away from high-risk areas
        risk_gradient = self._calculate_risk_gradient(agent_id, pos, agents_info)

        # Combine role target with risk avoidance
        if params['movement_strategy'] == 'hyper_aggressive':
            # Attackers prioritize center, less risk-averse
            target_direction = 0.8 * direction - 0.2 * risk_gradient
        elif params['movement_strategy'] == 'aggressive_patrol':
            # Scouts balance between patrol and risk avoidance
            target_direction = 0.5 * direction - 0.5 * risk_gradient
        else:  # active_defense
            # Defenders prioritize risk avoidance
            target_direction = 0.3 * direction - 0.7 * risk_gradient

        # Normalize
        if np.linalg.norm(target_direction) > 0.01:
            target_direction = target_direction / np.linalg.norm(target_direction)

        # Convert to action (simplified: 8 directions + stay)
        # Actions: 0-7 for 8 directions, 8 for stay
        angle = np.arctan2(target_direction[1], target_direction[0])
        action_idx = int((angle + np.pi) / (2 * np.pi) * 8) % 8

        return action_idx

    def _calculate_risk_gradient(self, agent_id, pos, agents_info):
        """
        Calculate risk gradient at agent position
        Gradient points away from high-risk areas

        Args:
            agent_id: Agent ID
            pos: Agent position
            agents_info: All agents information

        Returns:
            Risk gradient vector (2D)
        """
        gradient = np.zeros(2)

        for other_id, other_info in agents_info.items():
            if other_id == agent_id or not other_info['alive']:
                continue

            other_pos = other_info['pos']
            diff = pos - other_pos
            dist = np.linalg.norm(diff)

            if dist < 0.1:
                continue

            # Risk potential of other agent
            p_fail = self.failure_probabilities.get(other_id, 0.0)
            U_other = 1.0 / (max(AlgorithmConfig.epsilon, 1.0 - p_fail) ** AlgorithmConfig.gamma)

            # Gradient contribution (inverse square law)
            gradient += U_other * diff / (dist ** 3)

        # Normalize
        if np.linalg.norm(gradient) > 0.01:
            gradient = gradient / np.linalg.norm(gradient)

        return gradient
