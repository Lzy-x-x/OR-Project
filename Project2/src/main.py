import networkx as nx
import random
import heapq
import time
import math

# pip install pulp
import pulp

def calculate_p(n, epsilon=0.1):
    return ((1 + epsilon) * math.log(n)) / n

def generate_connected_ER_graph(n, p):
    """
    生成一个连通的ER图，如果生成的图不连通就继续尝试。
    :param n: 节点数
    :param p: 每条边出现的概率
    :return: 生成的连通图 G（networkx Graph 对象）
    """
    while True:
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            return G


def assign_random_weights(G, low=1, high=10):
    """
    随机给图的边赋权重
    :param G: networkx Graph
    :param low: 权重下界（含）
    :param high: 权重上界（含）
    :return: 无返回，直接对 G 的属性进行修改
    """
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(low, high)


def has_negative_edge(G):
    """
    检查图中是否有负权边
    :param G: networkx Graph
    :return: bool
    """
    for (u, v) in G.edges():
        if G[u][v]['weight'] < 0:
            return True
    return False


def dijkstra_heapq(G, source, target):
    """
    使用 heapq 实现的 Dijkstra 算法，返回从 source 到 target 的最短距离。
    若无法到达，则返回 None。
    :param G: networkx Graph（无负权）
    :param source: 源点
    :param target: 目标点
    :return: 最短路径长度(或 None)
    """
    # 初始化距离
    dist = {node: math.inf for node in G.nodes}
    dist[source] = 0

    # 最小堆 (距离, 节点)
    heap = [(0, source)]

    visited = set()

    while heap:
        current_dist, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        if u == target:
            return current_dist

        # 遍历邻接节点
        for v in G[u]:
            w = G[u][v]['weight']
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))

    # 若无法到达
    return None


def shortest_path_lp(G, source, target, solver_name='PULP_CBC_CMD'):
    """
    使用线性规划方式求最短路：
      min sum_{(i,j) in E} c(i,j) * x(i,j)
      s.t.
          流量平衡:
            - 出源点 s 有1单位流, 进源点流 = 0
            - 对除 s,t 外点 k, 流入量=流出量
            - 入汇点 t 有1单位流, 出汇点流 = 0
          x(i,j) >= 0
    :param G: networkx Graph
    :param source: 源点
    :param target: 目标点
    :param solver_name: 求解器名称
    :return: 最短距离(若可行)，否则 None
    """
    # 1. 创建模型
    prob = pulp.LpProblem("ShortestPath", pulp.LpMinimize)

    # 2. 为每条边创建变量 x(i,j)
    x_vars = {}
    for (i, j) in G.edges():
        # 注意网络中的边是无向的话，需要稍作处理，这里假设是无向则拆成两个有向边
        # 如果是有向图，可以直接用 (i,j)
        x_vars[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0)
        x_vars[(j, i)] = pulp.LpVariable(f"x_{j}_{i}", lowBound=0)

    # 3. 目标函数：min \sum c(i,j) * x(i,j)
    prob += pulp.lpSum([
                           G[u][v]['weight'] * x_vars[(u, v)]
                           for (u, v) in G.edges()
                       ] + [
                           G[u][v]['weight'] * x_vars[(v, u)]
                           for (u, v) in G.edges()
                       ])

    # 4. 约束：流量平衡
    # 4.1 对于源点 s，流出 - 流入 = 1
    prob += (
            pulp.lpSum([x_vars[(source, j)] for j in G[source]]) -
            pulp.lpSum([x_vars[(j, source)] for j in G[source]])
            == 1
    ), "flow_conservation_source"

    # 4.2 对于汇点 t，流出 - 流入 = -1
    prob += (
            pulp.lpSum([x_vars[(target, j)] for j in G[target]]) -
            pulp.lpSum([x_vars[(j, target)] for j in G[target]])
            == -1
    ), "flow_conservation_target"

    # 4.3 对于其他节点 k，流出 - 流入 = 0
    for k in G.nodes():
        if k != source and k != target:
            prob += (
                    pulp.lpSum([x_vars[(k, j)] for j in G[k]]) -
                    pulp.lpSum([x_vars[(j, k)] for j in G[k]])
                    == 0
            ), f"flow_conservation_{k}"

    # 5. 调用求解器
    solver = pulp.getSolver(solver_name, timeLimit=60)  # 可设置一些 solver 参数
    prob.solve(solver)

    # 6. 查看结果
    if pulp.LpStatus[prob.status] == 'Optimal':
        return pulp.value(prob.objective)
    else:
        return None


def test_compare_methods(n, p):
    """
    生成随机连通图，并比较Dijkstra与LP方法在同一个图上的运行时间以及结果。
    """
    # 1. 生成随机连通图
    G = generate_connected_ER_graph(n, p)
    # 2. 随机赋权重
    assign_random_weights(G, 1, 50)

    # 检查负边
    if has_negative_edge(G):
        print("图中含有负权重边，Dijkstra算法不适用！")
        return

    # 随机指定 source, target
    nodes_list = list(G.nodes())
    source = nodes_list[0]
    target = nodes_list[-1]  # 简单地取第一个和最后一个

    print(f"图节点数: {n}, 源点: {source}, 汇点: {target}, 边数: {G.number_of_edges()}")

    # ============ Dijkstra ============
    start_time = time.time()
    dist_dij = dijkstra_heapq(G, source, target)
    end_time = time.time()
    dij_time = end_time - start_time
    print(f"Dijkstra结果: {dist_dij}, 用时: {dij_time:.4f}秒")

    # ============ LP ============
    start_time = time.time()
    dist_lp = shortest_path_lp(G, source, target)
    end_time = time.time()
    lp_time = end_time - start_time
    print(f"LP求解结果: {dist_lp}, 用时: {lp_time:.4f}秒")

    # ============ 比较 ============
    if dist_dij is not None and dist_lp is not None:
        print(f"二者最短距离结果是否一致？ {abs(dist_dij - dist_lp) < 1e-5}")
    print("-" * 50)


if __name__ == "__main__":
    node_counts = [1000, 5000, 10000]
    epsilon = 0.1
    for n in node_counts:
        p = calculate_p(n, epsilon)
        test_compare_methods(n, p)