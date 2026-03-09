import json
from typing import Dict, List

class Task:
    """
    Represents a computational task in the DAG.
    表示 DAG 中的一个计算任务
    """
    def __init__(self, task_id: str):
        self.task_id = task_id
        # Execution time matrix (W): processor_id -> cost
        # 执行时间矩阵: w_{i,j} => processor_id -> cost
        self.comp_costs: Dict[str, float] = {}
        # Dependencies: predecessors & successors -> comm_cost
        # 依赖关系: 前驱 & 后继 => neighbor_task_id -> comm_cost
        self.predecessors: Dict[str, float] = {}
        self.successors: Dict[str, float] = {}

        # Optimistic Cost Table (OCT) matrix: processor_id -> oct value
        # OCT 表单矩阵: processor_id -> oct value
        self.oct: Dict[str, float] = {}
        # Priority: calculated based on the OCT
        # 优先级: 基于 OCT 计算得出
        self.rank_oct: float = 0.0

        # Allocation result
        # 分配结果
        self.assigned_processor: str = None
        self.ast: float = 0.0  # Actual Start Time (实际开始时间)
        self.aft: float = 0.0  # Actual Finish Time (实际完成时间)

    def __repr__(self):
        return f"Task({self.task_id}, Rank: {self.rank_oct:.2f}, Assigned: {self.assigned_processor})"


class Processor:
    """
    Represents a processor/computing node in a heterogeneous computing environment.
    表示异构计算环境中的一个处理器/计算节点
    """
    def __init__(self, proc_id: str):
        self.proc_id = proc_id
        # Tracks the current available time of this processor (T_Available)
        # 用于记录该处理器的当前可用时间 T_Available
        self.available_time: float = 0.0


class PEFT:
    """
    Predictive Earliest Finish Time (PEFT) scheduling algorithm.
    PEFT (Predictive Earliest Finish Time) 调度算法
    """
    def __init__(self, tasks_data: dict, processors_list: List[str]):
        """
        Initialize DAG tasks and heterogeneous environment nodes.
        初始化 DAG 任务与异构环境节点
        :param tasks_data: dict with execution times and dependencies (包含任务执行时间和依赖关系的字典)
        :param processors_list: list of processor IDs (处理器的ID列表)
        """
        self.processors = [Processor(p) for p in processors_list]
        self.processor_ids = processors_list
        self.tasks: Dict[str, Task] = {}

        # 1. Parse tasks and computation costs (w_{i,j})
        # 1. 解析任务及其在各处理器的计算开销 (w_{i,j})
        for t_id, t_info in tasks_data.items():
            task = Task(t_id)
            task.comp_costs = t_info.get("comp_costs", {})
            self.tasks[t_id] = task

        # 2. Parse dependencies and communication costs (c_{m, i})
        # 2. 解析依赖和通信开销 (c_{m, i})
        for t_id, t_info in tasks_data.items():
            deps = t_info.get("dependencies", {})
            for pred_id, comm_cost in deps.items():
                self.tasks[t_id].predecessors[pred_id] = comm_cost
                if pred_id in self.tasks:
                    self.tasks[pred_id].successors[t_id] = comm_cost

        self.scheduling_order: List[Task] = []

    def topological_sort(self) -> List[str]:
        """
        Kahn's algorithm for topological sorting, supporting backward OCT calculation.
        使用 Kahn 算法进行拓扑排序，用于支持反向遍历计算 OCT
        """
        in_degree = {t_id: 0 for t_id in self.tasks}
        for task in self.tasks.values():
            for succ_id in task.successors:
                in_degree[succ_id] += 1

        queue = [t_id for t_id, deg in in_degree.items() if deg == 0]
        topo_order = []

        while queue:
            curr = queue.pop(0)
            topo_order.append(curr)
            for succ_id in self.tasks[curr].successors:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)

        return topo_order

    def phase1_task_prioritizing(self):
        """
        Phase 1: Calculate OCT (Optimistic Cost Table) backward and evaluate task priorities.
        阶段 1: 反向计算 OCT 矩阵，并评估任务优先级
        """
        # Reverse topological sort: Compute backward from the exit node.
        # 逆向拓扑排序：从出口节点往前递归计算
        reversed_tasks = self.topological_sort()[::-1]

        for t_id in reversed_tasks:
            task = self.tasks[t_id]
            # Exit Node: OCT(t_exit, p_k) = 0
            # 出口节点：OCT(t_exit, p_k) = 0
            if not task.successors:
                for p_k in self.processor_ids:
                    task.oct[p_k] = 0.0
            else:
                # Other nodes backward recursive calculation (PEFT Standard Formula: max over successor, min over processors)
                # 其他节点逆向递归计算 (PEFT 规范公式: 先关于后继节点取 max，内部对不同计算节点取 min)
                # OCT(t_i, p_k) = max_{t_m in succ(t_i)} [ min_{p_w} (OCT(t_m, p_w) + w_{m,w} + c_{i,m}) ]
                for p_k in self.processor_ids:
                    max_val = 0.0
                    for succ_id, comm_cost in task.successors.items():
                        succ_task = self.tasks[succ_id]
                        min_pw_val = float('inf')
                        # Find the best subsequent step p_w for each successor
                        # 为每个后继节点寻找独立的最优后续跳步 p_w
                        for p_w in self.processor_ids: 
                            # Zero out communication cost if data is transferred within the same node
                            # 数据跨节点传输需考虑通信开销，若是同一节点则设0 (关键路径预判逻辑中的通信消减)
                            actual_comm_cost = comm_cost if p_k != p_w else 0.0
                            val = succ_task.oct[p_w] + succ_task.comp_costs[p_w] + actual_comm_cost
                            if val < min_pw_val:
                                min_pw_val = val
                        
                        if min_pw_val > max_val:
                            max_val = min_pw_val
                    
                    task.oct[p_k] = max_val

        # Calculate rank_oct = Average over processors
        # 计算优先级 rank_oct = 处理器开销平均值
        P = len(self.processor_ids)
        for task in self.tasks.values():
            task.rank_oct = sum(task.oct[p_k] for p_k in self.processor_ids) / P

        # Sort descending to get scheduling order
        # 按照 rank_oct 降序排列得到调度顺序
        self.scheduling_order = sorted(list(self.tasks.values()), key=lambda t: t.rank_oct, reverse=True)

    def print_oct_table(self):
        """
        Output the full OCT matrix and rank_oct
        输出完整的 OCT 矩阵及 rank_oct
        """
        print("\n" + "="*50)
        print(f"{'OCT Table (Optimistic Cost Table)':^50}")
        print("="*50)
        header = f"{'Task':<6} | " + " | ".join([f"{p:>6}" for p in self.processor_ids]) + " | rank_oct"
        print(header)
        print("-" * 50)
        # Sort output by numeric task identity
        # 按照 Task 数字身份排序输出
        sorted_tasks = sorted(self.tasks.values(), key=lambda t: int(t.task_id.replace('T', '')))
        for task in sorted_tasks:
            octs = " | ".join([f"{task.oct[p]:>6.1f}" for p in self.processor_ids])
            print(f"{task.task_id:<6} | {octs} | {task.rank_oct:>6.1f}")
        print("="*50 + "\n")

    def calculate_cp_min_nodes(self) -> List[str]:
        """
        Calculate nodes on the Critical Path (CP_min) based on minimum computation costs and communication costs.
        计算基于最小执行时间和通信开销的关键路径 (CP_min)
        """
        topo = self.topological_sort()
        dp = {} # t_id -> (max_length, [path_nodes])
        
        for t_id in topo:
            task = self.tasks[t_id]
            node_min_w = min(task.comp_costs.values())
            
            if not task.predecessors:
                dp[t_id] = (node_min_w, [t_id])
            else:
                max_len = 0
                best_path = []
                for pred_id, comm_cost in task.predecessors.items():
                    pred_len, pred_path = dp[pred_id]
                    # path length = previous path length + communication cost + node min computation
                    path_len = pred_len + comm_cost + node_min_w
                    if path_len > max_len:
                        max_len = path_len
                        best_path = pred_path + [t_id]
                dp[t_id] = (max_len, best_path)
                
        # Find the exit node that has the maximum path length
        exit_nodes = [t_id for t_id, t in self.tasks.items() if not t.successors]
        global_max_len = 0
        cp_nodes = []
        for en in exit_nodes:
            if dp[en][0] > global_max_len:
                global_max_len = dp[en][0]
                cp_nodes = dp[en][1]
                
        return cp_nodes

    def print_performance_metrics(self, makespan: float):
        """
        Calculate and output Speedup and SLR (Schedule Length Ratio).
        计算并输出加速比 (Speedup) 和 调度长度比 (SLR)
        """
        # 1. Speedup: The ratio of sequential execution time (on the fastest node) to the parallel makespan
        seq_times = {p_id: sum(task.comp_costs[p_id] for task in self.tasks.values()) 
                     for p_id in self.processor_ids}
        min_seq_time = min(seq_times.values())
        best_seq_proc = min(seq_times, key=seq_times.get)
        speedup = min_seq_time / makespan
        
        # 2. SLR: The ratio of the makespan to the sum of min computation costs on the critical path (CP_min)
        cp_nodes = self.calculate_cp_min_nodes()
        cp_min_cost = sum(min(self.tasks[n].comp_costs.values()) for n in cp_nodes)
        slr = makespan / cp_min_cost
        
        print("\n" + "="*50)
        print(f"{'Performance Metrics (性能指标)':^50}")
        print("="*50)
        print(f"1. Minimum Sequential Time: {min_seq_time:.2f} (on Node {best_seq_proc})")
        print(f"2. Parallel Makespan      : {makespan:.2f}")
        print(f"-> Speedup ratio          : {speedup:.4f}")
        print("-" * 50)
        print(f"3. Critical Path Nodes    : {' -> '.join(cp_nodes)}")
        print(f"4. CP_min Computation Cost: {cp_min_cost:.2f}")
        print(f"-> SLR (Sch. Length Ratio): {slr:.4f}")
        print("="*50 + "\n")

    def phase2_processor_selection(self):
        """
        Phase 2: Forward simulation, assign tasks sequentially to minimize O_EFT.
        阶段 2: 前向模拟，按排序依次处理任务，选择令综合预期(O_EFT)最优的处理器
        """
        for task in self.scheduling_order:
            best_processor = None
            min_oeft = float('inf')
            best_aft = 0.0
            best_ast = 0.0

            for p in self.processors:
                p_j = p.proc_id

                # 1. EST(t_i, p_j) = max(T_Available(p_j), max_{n_m in pred(t_i)} (AFT(n_m) + c_{m,i}))
                max_pred_time = 0.0
                for pred_id, comm_cost in task.predecessors.items():
                    pred_task = self.tasks[pred_id]
                    # No communication cost if predecessor is on the same processor p_j
                    # 若前驱和自己均在 p_j 处理器，则通信开销为 0
                    actual_comm_cost = comm_cost if pred_task.assigned_processor != p_j else 0.0
                    pred_finish_time = pred_task.aft + actual_comm_cost
                    if pred_finish_time > max_pred_time:
                        max_pred_time = pred_finish_time

                est = max(p.available_time, max_pred_time)
                
                # 2. EFT(t_i, p_j) = EST(t_i, p_j) + w_{i,j}
                eft = est + task.comp_costs[p_j]

                # 3. O_EFT(t_i, p_j) = EFT(t_i, p_j) + OCT(t_i, p_j)
                o_eft = eft + task.oct[p_j]

                # Record minimum O_EFT situation
                # 记录使得 O_EFT 最小的处理情况
                if o_eft < min_oeft:
                    min_oeft = o_eft
                    best_processor = p
                    best_aft = eft
                    best_ast = est

            # Assign and advance local timeline
            # 执行最终调度记录
            task.assigned_processor = best_processor.proc_id
            task.ast = best_ast
            task.aft = best_aft
            best_processor.available_time = best_aft

    def schedule(self):
        """
        Run overall algorithm flow, return tasks allocation tracking and aggregate Makespan.
        运行整体算法，返回所有任务的执行记录和整体的 Makespan
        """
        self.phase1_task_prioritizing()
        self.print_oct_table()  # Print OCT to terminal (输出 OCT 以供对照)
        self.phase2_processor_selection()

        # Final Makespan is the maximum actual finish time crossing all tasks
        # Final Makespan = max { AFT(n_exit) }
        makespan = max(task.aft for task in self.tasks.values())
        
        results = []
        for task in self.tasks.values():
            results.append({
                "Task ID": task.task_id,
                "Assigned Processor": task.assigned_processor,
                "Start Time": round(task.ast, 2),
                "End Time": round(task.aft, 2),
                "Rank OCT": round(task.rank_oct, 2)
            })

        results.sort(key=lambda x: x["Start Time"])
        return results, makespan

    def visualize_dag(self):
        """
        Draw DAG representing dependency tree using NetworkX and Matplotlib.
        使用 NetworkX 和 Matplotlib 绘制分层结构的依赖图 (DAG)
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Warning] Please install networkx and matplotlib for DAG visualizing (pip install networkx matplotlib).")
            return

        G = nx.DiGraph()
        for t_id, task in self.tasks.items():
            G.add_node(t_id)
            for succ_id, cost in task.successors.items():
                G.add_edge(t_id, succ_id, weight=cost)
        
        # Calculate topological depth for layered hierarchical view
        # 计算图的分层深度，用于层次化布局
        for node in nx.topological_sort(G):
            layer = 0
            for pred in G.predecessors(node):
                layer = max(layer, G.nodes[pred].get('layer', 0) + 1)
            G.nodes[node]['layer'] = layer
            
        # Use multipartite layout aligned by 'layer'
        # 使用 multipartite_layout 进行分层布局 (按 layer 对齐)
        pos = nx.multipartite_layout(G, subset_key='layer')
        
        # Default multipartite is horizontal; flip X and Y axes for top-down tree view
        # 默认 multipartite 是从左到右水平拉长的，我们通过翻转坐标轴把它变成从上到下的树状图
        pos_top_down = {node: (coords[1], -coords[0]) for node, coords in pos.items()}
        
        plt.figure(figsize=(10, 8))
        
        # Color intensity correlates with rank priority
        # 节点颜色基于优先级 rank_oct (颜色深=优先级高)
        ranks = [self.tasks[node].rank_oct for node in G.nodes()]
        
        # Draw nodes and edges (绘制节点和边)
        nx.draw(G, pos_top_down, with_labels=True, node_color=ranks, cmap=plt.cm.Blues, 
                node_size=2500, font_size=12, font_weight='bold', arrows=True,
                arrowsize=20, edgecolors='black')
        
        # Draw edge labels adjusting positions slightly to avoid overlay
        # 绘制权重标签，移位避免重叠
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos_top_down, edge_labels=edge_labels, label_pos=0.3, font_size=10)
        
        plt.title("DAG Dependency Visualization (Hierarchical Layout)")
        plt.show(block=False)  # <-- Changed to non-blocking so the Gantt chart opens at the same time

    def visualize_gantt_chart(self, tasks_schedule, final_makespan):
        """
        Draw Gantt chart of execution schedule using Matplotlib.
        使用 Matplotlib 绘制调度结果的甘特图
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            print("[Warning] Please install matplotlib for Gantt visualizing (pip install matplotlib).")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        proc_y = {p.proc_id: i for i, p in enumerate(self.processors)}
        
        for i, res in enumerate(tasks_schedule):
            task_id = res['Task ID']
            p_id = res['Assigned Processor']
            start = res['Start Time']
            end = res['End Time']
            duration = end - start
            
            y = proc_y[p_id]
            ax.barh(y, duration, left=start, height=0.4, align='center', 
                    color=colors[i % len(colors)], edgecolor='black', alpha=0.8)
            ax.text(start + duration / 2, y, task_id, ha='center', va='center', 
                    color='white', fontweight='bold')
        
        ax.set_yticks(list(proc_y.values()))
        ax.set_yticklabels(list(proc_y.keys()))
        ax.set_xlabel("Time")
        ax.set_ylabel("Processor Nodes")
        ax.set_title(f"PEFT Scheduling Gantt Chart (Makespan: {final_makespan:.2f})")
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()


# ============== 本地测试 / JSON 数据示例 ==============
if __name__ == "__main__":
    # 解析来自云端或者其他入口的一组 JSON 描述：包含 Node ID, Dependencies, Duration / Costs 等
    sample_dag_json = {
        "processors": ["P1", "P2", "P3"],
        "tasks": {
            "T1": {"comp_costs": {"P1": 22, "P2": 21, "P3": 36}, "dependencies": {}},
            "T2": {"comp_costs": {"P1": 22, "P2": 18, "P3": 18}, "dependencies": {"T1": 17}},
            "T3": {"comp_costs": {"P1": 32, "P2": 27, "P3": 43}, "dependencies": {"T1": 31}},
            "T4": {"comp_costs": {"P1": 7,  "P2": 10, "P3": 4},  "dependencies": {"T1": 29}},
            "T5": {"comp_costs": {"P1": 29, "P2": 27, "P3": 35}, "dependencies": {"T1": 13}},
            "T6": {"comp_costs": {"P1": 26, "P2": 17, "P3": 24}, "dependencies": {"T1": 7}},
            "T7": {"comp_costs": {"P1": 14, "P2": 25, "P3": 30}, "dependencies": {"T3": 16}},
            "T8": {"comp_costs": {"P1": 29, "P2": 23, "P3": 36}, "dependencies": {"T2": 3, "T4": 11, "T6": 5}},
            "T9": {"comp_costs": {"P1": 15, "P2": 21, "P3": 8},  "dependencies": {"T2": 30, "T4": 7, "T5": 57}},
            "T10": {"comp_costs": {"P1": 13, "P2": 16, "P3": 33}, "dependencies": {"T7": 9, "T8": 42, "T9": 7}}
        }
    }

    scheduler = PEFT(
        tasks_data=sample_dag_json["tasks"],
        processors_list=sample_dag_json["processors"]
    )
    
    tasks_schedule, final_makespan = scheduler.schedule()
    
    print("====== PEFT Algorithm Scheduling Results ======")
    for res in tasks_schedule:
        print(f"Task: {res['Task ID']:<4} | Node: {res['Assigned Processor']:<4} | "
              f"Execution Window: [{res['Start Time']:>5.2f} - {res['End Time']:>5.2f}] | Priority(Rank): {res['Rank OCT']}")
    print("-" * 55)
    print(f"Total Makespan(Max AFT): {final_makespan:.2f}")

    # ===== Metrics Calculation =====
    scheduler.print_performance_metrics(final_makespan)

    # ===== Visualization =====
    scheduler.visualize_dag()
    scheduler.visualize_gantt_chart(tasks_schedule, final_makespan)
