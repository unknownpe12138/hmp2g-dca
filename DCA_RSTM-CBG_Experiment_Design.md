# DCA框架实现RSTM-CBG算法实验设计

## 一、研究背景与目标

### 1.1 论文核心内容

**论文题目：** Role Switching and Task Migration Optimization in Adversarial Multi-layer Network Environments (对抗性多层网络环境中的角色切换和任务迁移优化)

**核心问题：**
- 在对抗性场景中，自主无人集群需要动态切换智能体角色以适应任务需求
- 角色切换触发网络拓扑重构，与任务分配形成深度耦合
- 需要联合优化角色切换、网络结构和任务分配策略

**提出算法：** RSTM-CBG (基于跨层桥接博弈的角色切换和任务迁移联合优化方法)

### 1.2 DCA框架适配性

DCA (Decentralized Collective Assault) 是一个多智能体对抗仿真环境，非常适合验证RSTM-CBG算法：

| DCA特性 | 论文需求 | 匹配度 |
|---------|---------|--------|
| 多智能体协同 (50v50) | 多智能体集群 | ✓ 完全匹配 |
| 对抗性环境 | 对抗性场景 | ✓ 完全匹配 |
| 2D平面运动 | 网络拓扑结构 | ✓ 可适配 |
| 激光射击机制 | 任务执行 | ✓ 可映射 |
| PPO强化学习 | 算法集成 | ✓ 可集成 |

### 1.3 实验目标

1. 验证RSTM-CBG算法在DCA环境中的有效性
2. 对比角色切换策略与固定角色策略的性能差异
3. 分析不同场景配置下的算法表现
4. 评估跨层桥接节点识别对任务迁移的影响

---

## 二、系统映射设计

### 2.1 智能体模型映射

#### 2.1.1 角色定义

将DCA中的智能体设计为多角色系统：

```python
# 角色类型定义
ROLE_TYPES = {
    'scout': {
        'name': '侦察兵',
        'capability': ['high_vision', 'high_speed', 'stealth'],
        'function': ['situational_awareness', 'target_reconnaissance'],
        'risk_exposure': 0.3,  # 低暴露风险
        'attack_power': 0.3,
        'vision_range': 2.0,
        'max_speed': 1.2
    },
    'attacker': {
        'name': '突击兵',
        'capability': ['high_attack', 'medium_speed', 'armor'],
        'function': ['target_strike', 'fire_coordination'],
        'risk_exposure': 0.8,  # 高暴露风险
        'attack_power': 1.0,
        'vision_range': 1.0,
        'max_speed': 1.0
    },
    'defender': {
        'name': '防守兵',
        'capability': ['high_defense', 'medium_attack', 'support'],
        'function': ['area_defense', 'teammate_protection'],
        'risk_exposure': 0.5,  # 中等暴露风险
        'attack_power': 0.6,
        'vision_range': 1.2,
        'max_speed': 0.9
    },
    'relay': {
        'name': '通信中继',
        'capability': ['high_communication', 'information_relay'],
        'function': ['information_relay', 'coordination_bridge'],
        'risk_exposure': 0.4,
        'attack_power': 0.2,
        'vision_range': 1.5,
        'max_speed': 1.0
    }
}
```

#### 2.1.2 角色状态表示

```python
class AgentRole:
    def __init__(self, role_type):
        self.role_type = role_type
        self.capability_vector = self._get_capability_vector()
        self.function_vector = self._get_function_vector()
        self.state_vector = self._get_state_vector()

    def _get_capability_vector(self):
        """能力特征向量 c_i(t)"""
        role = ROLE_TYPES[self.role_type]
        return np.array([
            role['attack_power'],
            role['vision_range'],
            role['max_speed'],
            1.0 if 'stealth' in role['capability'] else 0.0,
            1.0 if 'armor' in role['capability'] else 0.0
        ])

    def _get_function_vector(self):
        """功能属性向量 f_i(t)"""
        role = ROLE_TYPES[self.role_type]
        return np.array([
            1.0 if 'situational_awareness' in role['function'] else 0.0,
            1.0 if 'target_strike' in role['function'] else 0.0,
            1.0 if 'area_defense' in role['function'] else 0.0,
            1.0 if 'information_relay' in role['function'] else 0.0
        ])
```

#### 2.1.3 角色切换机制

```python
class RoleSwitchingMechanism:
    def __init__(self):
        self.switching_cost_matrix = self._build_cost_matrix()
        self.feasible_roles = {
            'agent_0': ['scout', 'attacker'],
            'agent_1': ['attacker', 'defender'],
            # ... 每个智能体的可行角色集
        }

    def _build_cost_matrix(self):
        """构建角色切换成本矩阵"""
        roles = list(ROLE_TYPES.keys())
        cost_matrix = np.zeros((len(roles), len(roles)))

        for i, r1 in enumerate(roles):
            for j, r2 in enumerate(roles):
                if i == j:
                    cost_matrix[i][j] = 0
                else:
                    # 基于角色差异计算切换成本
                    cost_matrix[i][j] = self._calculate_switch_cost(r1, r2)

        return cost_matrix

    def _calculate_switch_cost(self, r1, r2):
        """计算从角色r1切换到r2的成本"""
        # 时间延迟成本
        delay_cost = abs(ROLE_TYPES[r1]['max_speed'] - ROLE_TYPES[r2]['max_speed'])

        # 资源消耗成本
        resource_cost = 0.5  # 固定切换消耗

        # 能力重建成本
        capability_diff = np.linalg.norm(
            ROLE_TYPES[r1]['attack_power'] - ROLE_TYPES[r2]['attack_power']
        )

        return delay_cost + resource_cost + capability_diff
```

### 2.2 多层网络模型映射

#### 2.2.1 网络层定义

在DCA环境中构建3层网络结构：

```python
class MultiLayerNetwork:
    def __init__(self, num_agents):
        self.layers = {
            'communication': self._init_communication_layer(num_agents),
            'coordination': self._init_coordination_layer(num_agents),
            'fire_control': self._init_fire_control_layer(num_agents)
        }

    def _init_communication_layer(self, num_agents):
        """通信层：所有智能体都参与，基于通信距离建边"""
        return NetworkLayer(
            agents=list(range(num_agents)),
            edge_type='distance_based',
            max_distance=1.5
        )

    def _init_coordination_layer(self, num_agents):
        """协调层：侦察兵、通信中继和部分防守兵参与"""
        return NetworkLayer(
            agents=[i for i in range(num_agents) if self._has_coordination_role(i)],
            edge_type='hierarchical'
        )

    def _init_fire_control_layer(self, num_agents):
        """火力层：突击兵和防守兵参与"""
        return NetworkLayer(
            agents=[i for i in range(num_agents) if self._has_fire_role(i)],
            edge_type='formation_based'
        )
```

#### 2.2.2 角色-网络层映射

```python
def get_network_affiliations(role):
    """根据角色确定智能体所属的网络层"""
    role_layer_mapping = {
        'scout': ['communication', 'coordination'],
        'attacker': ['communication', 'fire_control'],
        'defender': ['communication', 'coordination', 'fire_control'],
        'relay': ['communication', 'coordination']
    }
    return role_layer_mapping.get(role, ['communication'])
```

### 2.3 任务模型映射

#### 2.3.1 任务类型定义

```python
TASK_TYPES = {
    'reconnaissance': {
        'capability_requirements': ['high_vision', 'high_speed'],
        'priority': 0.7,
        'duration': 100
    },
    'assault': {
        'capability_requirements': ['high_attack', 'armor'],
        'priority': 0.9,
        'duration': 150
    },
    'defense': {
        'capability_requirements': ['high_defense', 'medium_attack'],
        'priority': 0.8,
        'duration': 200
    },
    'relay_support': {
        'capability_requirements': ['high_communication'],
        'priority': 0.5,
        'duration': 80
    }
}
```

#### 2.3.2 角色-任务兼容性函数

```python
def calculate_role_task_compatibility(role, task):
    """
    计算角色与任务的兼容度 φ(r_i, τ_j)
    """
    role_info = ROLE_TYPES[role]
    task_info = TASK_TYPES[task['type']]

    # 计算能力满足率
    role_capabilities = set(role_info['capability'])
    task_requirements = set(task_info['capability_requirements'])

    capability_satisfaction = len(
        role_capabilities & task_requirements
    ) / len(task_requirements)

    # 考虑角色状态 (健康度、能量等)
    state_factor = 0.8  # 简化处理，实际应从state_vector计算

    # 综合映射
    compatibility = min(1.0, capability_satisfaction * state_factor + 0.2)

    return compatibility
```

---

## 三、RSTM-CBG算法实现

### 3.1 算法总体框架

```python
class RSTM_CBG:
    """
    基于跨层桥接博弈的角色切换和任务迁移联合优化方法
    """
    def __init__(self, dca_env, config):
        self.env = dca_env
        self.config = config

        # 算法组件
        self.bridging_node_identifier = BridgingNodeIdentifier()
        self.role_decision_engine = RoleDecisionEngine()
        self.task_allocator = TaskAllocator()

        # 系统状态
        self.agent_roles = {}  # 每个智能体当前的角色
        self.network_topology = None
        self.task_assignments = {}

    def run(self, obs):
        """算法主循环"""
        # 阶段1: 跨层博弈角色决策
        optimal_roles = self._cross_layer_role_decision(obs)

        # 阶段2: 结构感知任务分配
        task_allocation = self._structure_aware_task_allocation(
            optimal_roles, obs
        )

        # 返回动作
        return self._generate_actions(task_allocation)

    def _cross_layer_role_decision(self, obs):
        """阶段1: 跨层博弈角色决策"""
        # 步骤1: 识别跨层桥接节点
        bridging_nodes = self.bridging_node_identifier.identify(
            self.env.get_network_topology(),
            self.env.get_agents()
        )

        # 步骤2: 基于Shapley值进行角色决策
        optimal_roles = self.role_decision_engine.decide_roles(
            agents=self.env.get_agents(),
            bridging_nodes=bridging_nodes,
            current_obs=obs
        )

        return optimal_roles

    def _structure_aware_task_allocation(self, roles, obs):
        """阶段2: 结构感知任务分配"""
        # 更新网络拓扑
        updated_topology = self._update_topology_after_role_switch(roles)

        # 计算桥接感知距离
        bridging_aware_distances = self._calculate_bridging_aware_distances(
            updated_topology
        )

        # 分配任务
        task_allocation = self.task_allocator.allocate(
            tasks=self._identify_tasks(obs),
            agents=self.env.get_agents(),
            roles=roles,
            distances=bridging_aware_distances
        )

        return task_allocation
```

### 3.2 跨层桥接节点识别

```python
class BridgingNodeIdentifier:
    """
    跨层桥接节点识别器
    """
    def identify(self, network_topology, agents):
        """
        识别跨层桥接节点

        返回: 按CBD值排序的智能体优先级序列
        """
        cbd_scores = {}

        for agent_id, agent in agents.items():
            cbd = self._calculate_cross_layer_bridging_degree(
                agent, network_topology
            )
            cbd_scores[agent_id] = cbd

        # 按CBD降序排序
        priority_sequence = sorted(
            cbd_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return priority_sequence

    def _calculate_cross_layer_bridging_degree(self, agent, topology):
        """
        计算跨层桥接度 CBD(v_i)

        CBD(v_i) = |L_i| * Σ(deg_l(v_i) / |V_l|) * (|R_i| / max|R_j|)
        """
        # 1. 计算所属网络层数 |L_i|
        layer_affiliations = get_network_affiliations(agent.current_role)
        num_layers = len(layer_affiliations)

        # 2. 计算各层连接强度
        connection_strength = 0
        for layer in layer_affiliations:
            layer_agents = topology.get_layer_agents(layer)
            degree = topology.get_degree_in_layer(agent.id, layer)
            connection_strength += degree / len(layer_agents)

        # 3. 计算角色灵活性
        feasible_roles = agent.get_feasible_roles()
        role_flexibility = len(feasible_roles) / MAX_ROLES_PER_AGENT

        # 综合计算CBD
        cbd = num_layers * connection_strength * role_flexibility

        return cbd
```

### 3.3 基于Shapley值的角色决策

```python
class RoleDecisionEngine:
    """
    基于Shapley值的角色决策引擎
    """
    def decide_roles(self, agents, bridging_nodes, current_obs):
        """
        为所有智能体确定最优角色配置
        """
        optimal_roles = {}

        # 按CBD优先级顺序处理
        for agent_id, cbd_value in bridging_nodes:
            agent = agents[agent_id]

            # 构建k-hop邻域
            neighborhood = self._construct_k_hop_neighborhood(
                agent, agents, k=2
            )

            # 计算每个可行角色的Shapley值
            role_shapley_values = {}
            for role in agent.get_feasible_roles():
                shapley_value = self._calculate_neighborhood_shapley(
                    agent, role, neighborhood, current_obs
                )
                role_shapley_values[role] = shapley_value

            # 计算角色切换净收益
            current_role = agent.current_role
            best_role = None
            max_net_benefit = -float('inf')

            for role, shapley_value in role_shapley_values.items():
                # 净收益 = Shapley值增量 - 切换成本
                net_benefit = (
                    shapley_value -
                    role_shapley_values[current_role] -
                    self._get_switching_cost(current_role, role)
                )

                if net_benefit > max_net_benefit:
                    max_net_benefit = net_benefit
                    best_role = role

            # 仅当净收益为正时才切换角色
            if max_net_benefit > 0:
                optimal_roles[agent_id] = best_role
            else:
                optimal_roles[agent_id] = current_role

        return optimal_roles

    def _calculate_neighborhood_shapley(self, agent, role, neighborhood, obs):
        """
        计算邻域Shapley值（使用蒙特卡洛采样）
        """
        num_samples = 100  # 采样次数
        total_marginal_contribution = 0

        for _ in range(num_samples):
            # 随机排列邻域智能体
            permuted_agents = self._random_permutation(neighborhood)

            # 计算边际贡献
            marginal_contribution = self._calculate_marginal_contribution(
                agent, role, permuted_agents, obs
            )
            total_marginal_contribution += marginal_contribution

        shapley_value = total_marginal_contribution / num_samples
        return shapley_value

    def _calculate_marginal_contribution(self, agent, role, permuted_agents, obs):
        """
        计算智能体以指定角色加入联盟时的边际贡献
        """
        # 模拟：使用该角色时，联盟能完成的任务价值
        # 简化实现：计算该角色对邻域整体兼容性的提升

        base_value = 0  # 无该智能体时的联盟价值

        # 计算加入该智能体（指定角色）后的联盟价值
        improved_value = 0
        for other_agent in permuted_agents:
            # 计算协同效应
            compatibility = calculate_role_task_compatibility(role, other_agent.task)
            improved_value += compatibility

        marginal_contribution = improved_value - base_value
        return marginal_contribution
```

### 3.4 结构感知任务分配

```python
class TaskAllocator:
    """
    结构感知任务分配器
    """
    def allocate(self, tasks, agents, roles, distances):
        """
        使用桥接感知距离进行任务分配
        """
        # 按任务紧急程度排序
        sorted_tasks = sorted(
            tasks,
            key=lambda t: t['priority'],
            reverse=True
        )

        task_allocation = {}

        for task in sorted_tasks:
            best_agent = None
            max_suitability = -float('inf')

            for agent_id, agent in agents.items():
                # 计算适配度
                suitability = self._calculate_suitability(
                    task, agent, roles[agent_id], distances
                )

                if suitability > max_suitability:
                    max_suitability = suitability
                    best_agent = agent_id

            # 分配任务
            task_allocation[task['id']] = {
                'agent_id': best_agent,
                'task': task,
                'suitability': max_suitability
            }

            # 更新智能体负载
            agents[best_agent].load += task['duration']

        return task_allocation

    def _calculate_suitability(self, task, agent, role, distances):
        """
        计算任务分配适配度

        Φ(τ_j, v_i) = ω1·φ(r_i, τ_j) - ω2·d_BA/d_max - ω3·load(v_i)/load_max
        """
        # 1. 角色-任务兼容性
        compatibility = calculate_role_task_compatibility(role, task)

        # 2. 桥接感知迁移距离
        task_location = task.get('current_location', agent.position)
        bridging_aware_distance = distances.get(
            (task_location, agent.id),
            float('inf')
        )
        distance_cost = bridging_aware_distance / MAX_DISTANCE

        # 3. 负载均衡
        load_cost = agent.load / MAX_LOAD

        # 综合适配度（权重可调）
        suitability = (
            0.5 * compatibility -
            0.3 * distance_cost -
            0.2 * load_cost
        )

        return suitability
```

### 3.5 桥接感知距离度量

```python
def calculate_bridging_aware_distances(topology, cbd_scores, delta=0.5):
    """
    计算所有智能体对之间的桥接感知距离

    d_BA(v_i, v_j) = min_{p∈P_ij} Σ_{e∈p} c_e / (1 + δ·CBD(v_mid))
    """
    import itertools

    distances = {}

    for i, j in itertools.combinations(topology.all_agents, 2):
        # 找到所有可行路径
        paths = topology.find_all_paths(i, j)

        min_distance = float('inf')
        best_path = None

        for path in paths:
            path_cost = 0
            for edge in path:
                # 获取中间节点的CBD值
                mid_node = edge.mid_node
                cbd = cbd_scores.get(mid_node, 0)

                # 桥接感知权重
                edge_cost = edge.base_cost / (1 + delta * cbd)
                path_cost += edge_cost

            if path_cost < min_distance:
                min_distance = path_cost
                best_path = path

        distances[(i, j)] = min_distance
        distances[(j, i)] = min_distance  # 对称

    return distances
```

---

## 四、DCA环境集成方案

### 4.1 环境修改

#### 4.1.1 修改核心文件

需要在DCA环境中修改以下文件：

**1. `MISSION/dca/collective_assault_parallel_run.py`**

添加角色系统支持：

```python
class ScenarioConfig:
    # ... 现有配置 ...

    # 新增：角色系统配置
    enable_role_switching = True
    role_types = ['scout', 'attacker', 'defender', 'relay']
    num_roles_per_agent = 2  # 每个智能体可切换的角色数

    # 新增：多层网络配置
    enable_multi_layer_network = True
    network_layers = ['communication', 'coordination', 'fire_control']

    # 新增：任务系统配置
    enable_task_system = True
    task_types = ['reconnaissance', 'assault', 'defense', 'relay_support']
```

**2. 创建 `MISSION/dca/role_system.py`**

```python
"""
DCA环境中的角色系统实现
"""

class DCA_Agent_Role:
    """DCA智能体角色类"""
    def __init__(self, agent_id, initial_role='attacker'):
        self.agent_id = agent_id
        self.current_role = initial_role
        self.role_history = [initial_role]
        self.switching_count = 0

        # 应用角色属性
        self._apply_role_attributes()

    def switch_role(self, new_role, cost=0):
        """切换角色"""
        if new_role != self.current_role:
            self.current_role = new_role
            self.role_history.append(new_role)
            self.switching_count += 1
            self._apply_role_attributes()
            return cost
        return 0

    def _apply_role_attributes(self):
        """应用当前角色的属性"""
        role_attrs = ROLE_TYPES[self.current_role]

        # 修改智能体属性
        self.attack_power = role_attrs['attack_power']
        self.vision_range = role_attrs['vision_range']
        self.max_speed = role_attrs['max_speed']
        self.risk_exposure = role_attrs['risk_exposure']


class Role_Switching_Manager:
    """角色切换管理器"""
    def __init__(self, num_agents):
        self.agent_roles = {}
        self.switching_costs = RoleSwitchingMechanism()

        # 初始化每个智能体的角色
        for i in range(num_agents):
            initial_role = self._assign_initial_role(i)
            self.agent_roles[i] = DCA_Agent_Role(i, initial_role)

    def _assign_initial_role(self, agent_id):
        """分配初始角色"""
        # 基于智能体ID分配初始角色
        roles = list(ROLE_TYPES.keys())
        return roles[agent_id % len(roles)]

    def switch_agent_role(self, agent_id, new_role):
        """切换指定智能体的角色"""
        if agent_id in self.agent_roles:
            old_role = self.agent_roles[agent_id].current_role
            cost = self.switching_costs.get_cost(old_role, new_role)
            self.agent_roles[agent_id].switch_role(new_role, cost)
            return cost
        return 0

    def get_agent_role(self, agent_id):
        """获取智能体当前角色"""
        if agent_id in self.agent_roles:
            return self.agent_roles[agent_id].current_role
        return 'attacker'  # 默认角色
```

**3. 创建 `MISSION/dca/multi_layer_network.py`**

```python
"""
DCA环境中的多层网络实现
"""

class MultiLayerNetwork_DCA:
    """DCA多层网络"""

    LAYER_COMMUNICATION = 0
    LAYER_COORDINATION = 1
    LAYER_FIRE_CONTROL = 2

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.layers = {
            self.LAYER_COMMUNICATION: self._create_communication_layer(),
            self.LAYER_COORDINATION: self._create_coordination_layer(),
            self.LAYER_FIRE_CONTROL: self._create_fire_control_layer()
        }
        self.agent_layer_affiliation = {}

    def _create_communication_layer(self):
        """通信层：所有智能体都参与"""
        return {
            'agents': list(range(self.num_agents)),
            'edges': set(),
            'type': 'distance_based'
        }

    def _create_coordination_layer(self):
        """协调层：侦察兵和通信中继参与"""
        return {
            'agents': [],  # 动态更新
            'edges': set(),
            'type': 'hierarchical'
        }

    def _create_fire_control_layer(self):
        """火力层：突击兵和防守兵参与"""
        return {
            'agents': [],  # 动态更新
            'edges': set(),
            'type': 'formation_based'
        }

    def update_layer_affiliation(self, agent_id, role):
        """根据角色更新智能体的网络层归属"""
        layers = get_network_affiliations(role)

        # 更新协调层
        if self.LAYER_COORDINATION in layers:
            if agent_id not in self.layers[self.LAYER_COORDINATION]['agents']:
                self.layers[self.LAYER_COORDINATION]['agents'].append(agent_id)
        else:
            if agent_id in self.layers[self.LAYER_COORDINATION]['agents']:
                self.layers[self.LAYER_COORDINATION]['agents'].remove(agent_id)

        # 更新火力层
        if self.LAYER_FIRE_CONTROL in layers:
            if agent_id not in self.layers[self.LAYER_FIRE_CONTROL]['agents']:
                self.layers[self.LAYER_FIRE_CONTROL]['agents'].append(agent_id)
        else:
            if agent_id in self.layers[self.LAYER_FIRE_CONTROL]['agents']:
                self.layers[self.LAYER_FIRE_CONTROL]['agents'].remove(agent_id)

    def get_layer_agents(self, layer_id):
        """获取指定层的所有智能体"""
        return self.layers[layer_id]['agents']

    def get_agent_layers(self, agent_id):
        """获取智能体所属的所有层"""
        agent_layers = []
        for layer_id, layer_data in self.layers.items():
            if agent_id in layer_data['agents']:
                agent_layers.append(layer_id)
        return agent_layers
```

### 4.2 集成到DCA环境

**修改 `MISSION/dca/core.py` 中的智能体类：**

```python
class Agent:
    def __init__(self, uid, side, pos):
        # ... 现有初始化 ...

        # 新增：角色系统
        self.role = 'attacker'  # 默认角色
        self.feasible_roles = self._assign_feasible_roles()
        self.role_attributes = {}

    def _assign_feasible_roles(self):
        """分配可行角色集"""
        # 基于智能体ID分配不同的可行角色
        role_options = [
            ['scout', 'attacker'],
            ['attacker', 'defender'],
            ['defender', 'relay'],
            ['scout', 'relay'],
            ['attacker', 'relay']
        ]
        return role_options[self.uid % len(role_options)]

    def apply_role_attributes(self):
        """应用当前角色的属性"""
        role_info = ROLE_TYPES[self.role]

        # 覆盖基础属性
        self.attack_range = 0.4 * role_info['attack_power']
        self.view_range = 0.6 * role_info['vision_range']
        self.speed上限 = 0.03 * role_info['max_speed']

        # 调整激光参数
        if self.role == 'scout':
            self.laser_damage = 0.3
            self.laser_cooldown = 15
        elif self.role == 'attacker':
            self.laser_damage = 1.0
            self.laser_cooldown = 10
        elif self.role == 'defender':
            self.laser_damage = 0.6
            self.laser_cooldown = 12
        elif self.role == 'relay':
            self.laser_damage = 0.2
            self.laser_cooldown = 20

    def switch_role(self, new_role):
        """切换角色"""
        if new_role in self.feasible_roles and new_role != self.role:
            self.role = new_role
            self.apply_role_attributes()
            return True
        return False
```

---

## 五、实验配置

### 5.1 基础配置

```json
{
    "config.py->GlobalConfig": {
        "note": "RSTM-CBG算法验证实验",
        "env_name": "dca",
        "num_threads": 8,
        "device": "cuda",
        "n_parallel_frame": 10000,
        "max_n_episode": 100,
        "seed": 42
    },
    "MISSION.dca.collective_assault_parallel_run.py->ScenarioConfig": {
        "num_guards": 50,
        "num_attackers": 50,
        "agent_max_speed": 1.0,
        "agent_shoot_base_radius": 0.4,
        "MaxEpisodeStep": 190,
        "render": "False",

        // 新增：角色系统配置
        "enable_role_switching": true,
        "role_switch_interval": 20,
        "num_feasible_roles_per_agent": 2,

        // 新增：多层网络配置
        "enable_multi_layer_network": true,
        "network_update_interval": 10,

        // 新增：任务系统配置
        "enable_task_system": true,
        "task_generation_rate": 0.1
    },
    "ALGORITHM.conc_4hist.foundation->ReinforceAlgorithmFoundation": {
        "algorithm": "RSTM-CBG-PPO",
        "n_rollout_threads": 8,
        "n_training_threads": 1,
        "use_role_switching": true,
        "role_decision_interval": 20
    }
}
```

### 5.2 实验组设置

#### 实验组1：对比不同角色切换策略

| 组别 | 算法 | 角色切换策略 | 说明 |
|------|------|-------------|------|
| A1 | RSTM-CBG | 跨层博弈+Shapley值 | 本文算法 |
| A2 | Random | 随机切换 | 随机选择可行角色 |
| A3 | Fixed | 固定角色 | 不切换角色 |
| A4 | Greedy | 贪婪切换 | 选择当前最优角色 |
| A5 | Q-Learning | Q学习切换 | 基于Q值的角色切换 |

#### 实验组2：不同任务负载测试

| 组别 | 任务数量 | 智能体数量 | 角色类型数 |
|------|---------|-----------|-----------|
| B1 | 10 | 25 | 3 |
| B2 | 20 | 25 | 3 |
| B3 | 30 | 25 | 3 |
| B4 | 40 | 25 | 3 |
| B5 | 50 | 25 | 3 |

#### 实验组3：不同智能体规模测试

| 组别 | 任务数量 | 智能体数量 | 说明 |
|------|---------|-----------|------|
| C1 | 40 | 15 | 小规模 |
| C2 | 40 | 25 | 中规模 |
| C3 | 40 | 35 | 大规模 |
| C4 | 40 | 50 | 超大规模 |

#### 实验组4：不同角色灵活性测试

| 组别 | 每智能体可行角色数 | 说明 |
|------|------------------|------|
| D1 | 1 | 固定角色 |
| D2 | 2 | 低灵活性 |
| D3 | 3 | 中灵活性 |
| D4 | 4 | 高灵活性 |

---

## 六、评估指标

### 6.1 主要指标

```python
class ExperimentMetrics:
    """实验评估指标"""

    @staticmethod
    def calculate_objective_function_value(results):
        """
        目标函数值 (Utility)
        U = R_achieve - J(ξ, β, α)
        """
        # 任务期望完成率
        achievement_ratio = results['achievement_ratio']

        # 总成本
        total_cost = (
            0.2 * results['fulfillment_cost'] +
            0.2 * results['migration_cost'] +
            0.6 * results['reconfiguration_cost']
        )

        utility = achievement_ratio - total_cost
        return utility

    @staticmethod
    def calculate_task_completion_cost(results):
        """
        任务完成成本 (Total Cost)
        包括：履行成本、迁移成本、角色切换成本
        """
        return (
            results['fulfillment_cost'] +
            results['migration_cost'] +
            results['reconfiguration_cost']
        )

    @staticmethod
    def calculate_task_expected_completion_ratio(results):
        """
        任务期望完成率
        考虑角色-任务兼容性和智能体生存概率
        """
        completed_tasks = results['completed_tasks']
        total_tasks = results['total_tasks']

        # 加权完成率（考虑任务优先级）
        weighted_completed = sum(
            t['priority'] for t in completed_tasks
        )
        weighted_total = sum(
            t['priority'] for t in results['all_tasks']
        )

        return weighted_completed / weighted_total if weighted_total > 0 else 0

    @staticmethod
    def calculate_algorithm_runtime(start_time, end_time):
        """
        算法执行时间 (Runtime)
        """
        return end_time - start_time

    @staticmethod
    def calculate_role_switching_efficiency(results):
        """
        角色切换效率
        切换次数 / 完成任务数
        """
        return results['num_switches'] / results['completed_tasks']

    @staticmethod
    def calculate_network_connectivity(results):
        """
        网络连通性
        各层网络的平均连通度
        """
        return results['avg_connectivity']
```

### 6.2 辅助分析指标

```python
class AuxiliaryMetrics:
    """辅助分析指标"""

    @staticmethod
    def calculate_cbd_distribution(agents):
        """跨层桥接度分布"""
        cbd_values = [agent.cbd for agent in agents]
        return {
            'mean': np.mean(cbd_values),
            'std': np.std(cbd_values),
            'max': np.max(cbd_values),
            'min': np.min(cbd_values)
        }

    @staticmethod
    def calculate_role_distribution(agents):
        """角色分布"""
        role_counts = {}
        for agent in agents:
            role = agent.current_role
            role_counts[role] = role_counts.get(role, 0) + 1
        return role_counts

    @staticmethod
    def calculate_shapley_convergence(shapley_history):
        """Shapley值收敛性"""
        return {
            'iterations': len(shapley_history),
            'final_value': shapley_history[-1],
            'convergence_rate': shapley_history[-1] / shapley_history[0]
        }

    @staticmethod
    def calculate_task_completion_time(tasks):
        """任务完成时间分布"""
        completion_times = [t['completion_time'] for t in tasks if t['completed']]
        return {
            'mean': np.mean(completion_times),
            'std': np.std(completion_times),
            'median': np.median(completion_times)
        }
```

---

## 七、实验流程

### 7.1 完整实验流程

```python
def run_rstm_cbg_experiment(config):
    """运行RSTM-CBG实验"""

    # 1. 环境初始化
    env = DCARoleSwitchingEnv(config)

    # 2. 算法初始化
    algorithm = RSTM_CBG(env, config)

    # 3. 指标收集器
    metrics = ExperimentMetricsCollector()

    for episode in range(config.max_episodes):
        obs = env.reset()
        episode_results = {
            'steps': [],
            'role_switches': [],
            'task_completions': [],
            'network_states': []
        }

        for step in range(config.max_steps):
            # 执行算法
            start_time = time.time()
            actions = algorithm.run(obs)
            end_time = time.time()

            # 环境步进
            next_obs, rewards, dones, info = env.step(actions)

            # 收集数据
            episode_results['steps'].append({
                'obs': obs,
                'actions': actions,
                'rewards': rewards,
                'runtime': end_time - start_time
            })

            # 记录角色切换
            if 'role_switches' in info:
                episode_results['role_switches'].extend(info['role_switches'])

            # 记录任务完成
            if 'task_completions' in info:
                episode_results['task_completions'].extend(info['task_completions'])

            # 记录网络状态
            if 'network_state' in info:
                episode_results['network_states'].append(info['network_state'])

            obs = next_obs

            if dones:
                break

        # 计算本回合指标
        episode_metrics = metrics.calculate_episode_metrics(episode_results)

        # 记录
        metrics.record_episode(episode, episode_metrics)

        # 定期输出
        if (episode + 1) % 10 == 0:
            metrics.print_summary()

    # 生成最终报告
    return metrics.generate_final_report()
```

### 7.2 对比实验流程

```python
def run_comparative_experiments():
    """运行对比实验"""

    algorithms = {
        'RSTM-CBG': RSTM_CBG,
        'Random': RandomRoleSwitching,
        'Fixed': FixedRole,
        'Greedy': GreedyRoleSwitching,
        'Q-Learning': QLearningRoleSwitching
    }

    results = {}

    for alg_name, alg_class in algorithms.items():
        print(f"Running {alg_name}...")

        # 运行多次取平均
        alg_results = []
        for seed in range(10):
            config = get_experiment_config(seed)
            result = run_experiment(alg_class, config)
            alg_results.append(result)

        # 统计分析
        results[alg_name] = {
            'mean': np.mean(alg_results),
            'std': np.std(alg_results),
            'raw': alg_results
        }

    # 生成对比报告
    generate_comparison_report(results)
```

---

## 八、预期结果与分析

### 8.1 预期结果

1. **目标函数值**
   - RSTM-CBG应显著高于其他对比算法
   - 随着任务规模增加，优势更明显

2. **任务完成成本**
   - RSTM-CBG应能有效降低总成本
   - 桥接感知距离应减少迁移成本

3. **任务完成率**
   - RSTM-CBG应保持较高完成率
   - 角色切换应提升任务适应性

4. **算法执行时间**
   - RSTM-CBG应在可接受范围内
   - 远低于精确算法（如CPLEX）

### 8.2 消融分析

| 变体 | 说明 | 预期影响 |
|------|------|---------|
| RSTM-CBG-full | 完整算法 | 基准性能 |
| RSTM-CBG-no-bridging | 不使用桥接感知 | 成本增加 |
| RSTM-CBG-no-shapley | 不使用Shapley值 | 性能下降 |
| RSTM-CBG-no-role-switch | 固定角色 | 完成率降低 |

---

## 九、实验可视化

### 9.1 可视化组件

```python
class ExperimentVisualizer:
    """实验结果可视化"""

    @staticmethod
    def plot_comparison_results(results):
        """绘制对比实验结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 目标函数值对比
        axes[0, 0].bar(results.keys(), [r['mean']['utility'] for r in results.values()])
        axes[0, 0].set_title('Objective Function Value')

        # 任务完成成本对比
        axes[0, 1].bar(results.keys(), [r['mean']['cost'] for r in results.values()])
        axes[0, 1].set_title('Task Completion Cost')

        # 任务完成率对比
        axes[1, 0].bar(results.keys(), [r['mean']['achievement_ratio'] for r in results.values()])
        axes[1, 0].set_title('Task Achievement Ratio')

        # 执行时间对比
        axes[1, 1].bar(results.keys(), [r['mean']['runtime'] for r in results.values()])
        axes[1, 1].set_title('Algorithm Runtime')

        plt.tight_layout()
        plt.savefig('comparison_results.png')

    @staticmethod
    def plot_convergence_analysis(shapley_history):
        """绘制Shapley值收敛分析"""
        plt.figure(figsize=(10, 6))
        for agent_id, history in shapley_history.items():
            plt.plot(history, label=f'Agent {agent_id}')
        plt.xlabel('Iteration')
        plt.ylabel('Shapley Value')
        plt.title('Shapley Value Convergence')
        plt.legend()
        plt.savefig('convergence_analysis.png')

    @staticmethod
    def plot_role_switching_dynamics(role_history):
        """绘制角色切换动态"""
        fig, ax = plt.subplots(figsize=(12, 6))

        for agent_id, roles in role_history.items():
            ax.step(range(len(roles)), roles, where='post', label=f'Agent {agent_id}')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Role')
        ax.set_title('Role Switching Dynamics')
        ax.legend()
        plt.savefig('role_switching_dynamics.png')

    @staticmethod
    def plot_network_topology_evolution(network_states):
        """绘制网络拓扑演化"""
        # 使用networkx绘制多层网络演化
        fig, axes = plt.subplots(1, len(network_states), figsize=(15, 5))

        for i, state in enumerate(network_states):
            # 绘制通信层
            draw_network_layer(axes[i], state['communication_layer'], 'Communication')

        plt.tight_layout()
        plt.savefig('network_evolution.png')
```

### 9.2 DCA可视化集成

利用DCA现有的VHMAP可视化系统：

```python
# 在VISUALIZE/mcom.py中添加角色切换可视化

def visualize_role_switching(mcom, agents, roles):
    """可视化角色切换"""
    role_colors = {
        'scout': 'blue',
        'attacker': 'red',
        'defender': 'green',
        'relay': 'yellow'
    }

    for agent_id, role in roles.items():
        agent_pos = agents[agent_id].position
        mcom.draw_circle(
            center=agent_pos,
            radius=0.1,
            color=role_colors[role],
            alpha=0.5
        )

def visualize_multi_layer_network(mcom, network_topology):
    """可视化多层网络"""
    # 通信层 - 蓝色边
    for edge in network_topology['communication']['edges']:
        mcom.draw_line(edge[0], edge[1], color='blue', width=1)

    # 协调层 - 绿色边
    for edge in network_topology['coordination']['edges']:
        mcom.draw_line(edge[0], edge[1], color='green', width=2)

    # 火力层 - 红色边
    for edge in network_topology['fire_control']['edges']:
        mcom.draw_line(edge[0], edge[1], color='red', width=1)

def visualize_cbd_highlight(mcom, high_cbd_agents):
    """高亮显示高CBD智能体"""
    for agent_id in high_cbd_agents:
        # 绘制光晕效果
        mcom.draw_glow(agent_id, color='gold', intensity=0.7)
```

---

## 十、实施时间表

### 第一阶段：环境准备（1-2周）

- [ ] 修改DCA环境核心文件
- [ ] 实现角色系统
- [ ] 实现多层网络
- [ ] 实现任务系统

### 第二阶段：算法实现（2-3周）

- [ ] 实现跨层桥接节点识别
- [ ] 实现基于Shapley值的角色决策
- [ ] 实现结构感知任务分配
- [ ] 实现桥接感知距离计算

### 第三阶段：集成测试（1周）

- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能测试

### 第四阶段：实验执行（2-3周）

- [ ] 运行对比实验
- [ ] 收集数据
- [ ] 分析结果

### 第五阶段：论文撰写（2-3周）

- [ ] 整理实验数据
- [ ] 生成图表
- [ ] 撰写实验章节

---

## 十一、关键技术点

### 11.1 性能优化

```python
# 使用numba加速Shapley值计算
from numba import jit

@jit(nopython=True)
def calculate_marginal_contribution_fast(agent_value, coalition_values):
    """加速的边际贡献计算"""
    return coalition_values + agent_value - coalition_values

# 缓存网络拓扑计算
class TopologyCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size

    def get(self, topology_key):
        if topology_key in self.cache:
            return self.cache[topology_key]
        return None

    def set(self, topology_key, value):
        if len(self.cache) >= self.max_size:
            # LRU淘汰
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[topology_key] = value
```

### 11.2 并行化处理

```python
from multiprocessing import Pool

def parallel_role_decision(agents, topology, num_processes=4):
    """并行化角色决策"""
    with Pool(num_processes) as pool:
        results = pool.starmap(
            calculate_agent_optimal_role,
            [(agent, topology) for agent in agents]
        )
    return results
```

---

## 十二、风险评估与应对

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|---------|
| DCA环境修改困难 | 中 | 高 | 使用装饰器模式，最小化侵入性修改 |
| 算法性能不足 | 中 | 中 | 使用Cython/Numba加速关键路径 |
| 实验结果不显著 | 低 | 高 | 调整超参数，增加实验规模 |
| 可视化效果不佳 | 低 | 低 | 使用多种可视化工具互补 |

---

## 总结

本实验设计方案将RSTM-CBG算法与DCA多智能体对抗环境深度结合，通过：

1. **完整的系统映射**：将论文中的概念精确映射到DCA环境
2. **模块化实现**：算法组件独立实现，便于测试和优化
3. **全面的评估体系**：多维度指标评估算法性能
4. **丰富的对比实验**：验证算法的有效性和优越性

该方案既保持了论文算法的核心思想，又充分利用了DCA环境的仿真能力，预期可以有效验证RSTM-CBG算法在对抗性场景中的性能。
