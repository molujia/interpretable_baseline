
import networkx as nx
import numpy as np
import pyAgrum.lib.image as gumimage
import pyAgrum as gum
import utils as u
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
import random

def generate_random_dag(n, min_degree=0, max_degree=3):
    """
    生成一个随机有向无环图（DAG）。
    
    参数：
        n (int): 节点数量。
        min_degree (int): 每个节点的最小出度（默认 0，允许孤立节点）。
        max_degree (int): 每个节点的最大出度（默认 3，限制连通性）。
    
    返回：
        G (nx.DiGraph): 生成的 DAG 图对象。
    """
    # 初始化空的有向图
    G = nx.DiGraph()
    
    # 添加 n 个节点，编号 0 到 n-1
    for i in range(n):
        G.add_node(i)
    
    # 生成随机排列，确保拓扑序
    perm = np.random.permutation(n)
    
    # 为每个节点生成随机的出边
    for i in range(n-1):  # 最后一个节点不生成出边（避免无可用目标）
        current_node = perm[i]
        # 随机选择出度，限制不超过剩余节点数
        available_nodes = n - i - 1
        max_possible_degree = min(max_degree, available_nodes)
        r = np.random.randint(min_degree, max_possible_degree + 1)
        
        # 从后续节点中随机选择 r 个非连续的父母节点
        if r > 0:  # 只有当 r > 0 时选择目标节点
            possible_parents = perm[i+1:]  # 后续节点作为候选父母
            parents = np.random.choice(
                possible_parents,
                size=r,
                replace=False  # 不重复选择
            )
            # 添加边
            for p in parents:
                G.add_edge(current_node, p)
    
    # 验证是否为 DAG（防御性检查）
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Generated graph is not a DAG!")
    
    return G.edges()

# 获得贝叶斯（有概率表）和图本身
def get_random_dag(n, states):
    edges = generate_random_dag(n, 0, int(n/2))

    bn = gum.BayesNet('BN')
    g_graph = GeneralGraph([])
    for node in range(n):
        _node = u.get_node_name(node)
        g_graph.add_node(GraphNode(_node))
        bn.add(gum.RangeVariable(_node, str(node), 0, states - 1)) 
        # 节点名，显示名称，变量取值范围的最小值，变量取值范围最大值
        # 用于给定一个节点可以有几种状态

    for e in edges:
        bn.addArc(u.get_node_name(e[0]), u.get_node_name(e[1]))
        g_graph.add_directed_edge(GraphNode(u.get_node_name(e[0])),
                                  GraphNode(u.get_node_name(e[1])))

    bn.generateCPTs() #随机生成CPT
    return bn, g_graph

def inject_failure(bn, a_nodes):
    """
    为指定节点注入故障，随机选择以下三种故障模式：
    1. 节点恒为某个状态，不受父母影响。
    2. 节点出现新状态（模拟异常状态）。
    3. 节点的 CPT 随机变化（扰动现有分布）。

    参数：
        bn (gum.BayesNet): 贝叶斯网络。
        a_nodes (list): 要注入故障的节点列表（节点标识符，如 "node_0"）。
    
    返回：
        None（直接修改 bn 的 CPT）。
    """
    # 定义故障模式
    failure_modes = ['fixed_state']  #, 'new_state', 'cpt_change'
    
    for node in a_nodes:
        # 随机选择一种故障模式
        mode = np.random.choice(failure_modes)
        print(f"Injecting failure to node {node}: mode = {mode}")
        
        # 获取节点的状态数
        states = bn.variable(node).domainSize()  # 当前状态数（如 2 或 3）
        # print(states)

        if mode == 'fixed_state':
            # 故障模式 1：节点恒为某个状态
            fixed_state = np.random.randint(0, states)  # 随机选择一个状态
            cpt = bn.cpt(node)
            # 将 CPT 每列设置为固定状态概率为 1
            cpt.fillWith(0)  # 先清零
            for config in cpt.loopIn():  # 遍历父母状态组合
                cpt[config] = fixed_state  # 固定状态概率
            
        elif mode == 'new_state':
            # 故障模式 2：节点增加新状态
            # 注意：pyAgrum 不支持动态修改状态数，需重新定义变量
            # 这里我们模拟新状态的效果：将 CPT 调整为包含“异常”状态的概率
            cpt = bn.cpt(node)
            new_state_prob = 0.3  # 新状态的概率（可调整）
            cpt.fillWith(0)  # 清零
            for config in cpt.loopIn():
                # 为现有状态分配剩余概率（均匀分布）
                normal_probs = np.random.dirichlet([1] * states) * (1 - new_state_prob)
                cpt[config] = normal_probs
                # 模拟新状态：将部分概率转移到某个状态（例如最后一个状态）
                cpt[config][states-1] += new_state_prob  # 最后一个状态作为“异常”
        
        else:  # mode == 'cpt_change'
            # 故障模式 3：CPT 随机扰动
            cpt = bn.cpt(node)
            for config in cpt.loopIn():
                # 生成新的概率分布（Dirichlet 确保和为 1）
                new_probs = np.random.dirichlet([1] * states)
                cpt[config] = new_probs
    
    return bn

# 示例测试代码
if __name__ == "__main__":

    # 生成随机 DAG 和贝叶斯网络
    np.random.seed(42)
    n = 5
    states = 2
    bn, G = get_random_dag(n, states)
    
    # 注入故障
    valid_nodes = [bn.variable(node).name() for node in bn.nodes()]
    # print(valid_nodes)
    a_nodes = []
    a_nodes.append(random.choice(valid_nodes))
    # print(a_nodes)
    bn = inject_failure(bn, a_nodes)
    
    # 打印 CPT 检查结果
    for node in a_nodes:
        print(f"CPT for node {node} after failure:")
        print(bn.cpt(node))