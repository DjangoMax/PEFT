import json
from typing import Dict, List

class Task:
    """
    表示 DAG 中的一个计算任务
    """
    def __init__(self, task_id: str):
        self.task_id = task_id
        # 执行时间矩阵: w_{i,j} => processor_id -> cost
        self.comp_costs: Dict[str, float] = {}
        # 依赖关系: predecessors & successors => neighbor_task_id -> comm_cost
        self.predecessors: Dict[str, float] = {}
        self.successors: Dict[str, float] = {}

        # OCT (Optimistic Cost Table) 矩阵: processor_id -> oct value
        self.oct: Dict[str, float] = {}
        # 优先级: 基于 OCT 计算得出
        self.rank_oct: float = 0.0

        # 分配结果
        self.assigned_processor: str = None
        self.ast: float = 0.0  # 实际开始时间 (Actual Start Time)
        self.aft: float = 0.0  # 实际完成时间 (Actual Finish Time)

    def __repr__(self):
        return f"Task({self.task_id}, Rank: {self.rank_oct:.2f}, Assigned: {self.assigned_processor})"


class Processor:
    """
    表示异构计算环境中的一个处理器/计算节点
    """
    def __init__(self, proc_id: str):
        self.proc_id = proc_id
        # 用于记录该处理器的当前可用时间 T_Available
        self.available_time: float = 0.0


class PEFT:
    """
    PEFT (Predictive Earliest Finish Time) 调度算法
    """
    def __init__(self, tasks_data: dict, processors_list: List[str]):
        """
        :param tasks_data: 包含任务执行时间和依赖关系的字典
        :param processors_list: 处理器的ID列表
        """
        self.processors = [Processor(p) for p in processors_list]
        self.processor_ids = processors_list
        self.tasks: Dict[str, Task] = {}

        # 1. 解析任务及其在各处理器的计算开销 (w_{i,j})
        for t_id, t_info in tasks_data.items():
            task = Task(t_id)
            task.comp_costs = t_info.get("comp_costs", {})
            self.tasks[t_id] = task

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
        阶段 1: 反向计算 OCT (Optimistic Cost Table) 矩阵，并评估任务优先级
        """
        # 逆向拓扑排序：从出口节点往前递归计算
        reversed_tasks = self.topological_sort()[::-1]

        for t_id in reversed_tasks:
            task = self.tasks[t_id]
            # 出口节点（Exit Node）：OCT(t_exit, p_k) = 0
            if not task.successors:
                for p_k in self.processor_ids:
                    task.oct[p_k] = 0.0
            else:
                # 其他节点逆向递归计算：
                # OCT(t_i, p_k) = min_{p_w} [ max_{t_m in succ(t_i)} (OCT(t_m, p_w) + w_{m,w} + c_{i,m}) ]
                for p_k in self.processor_ids:
                    min_val = float('inf')
                    for p_w in self.processor_ids: # p_w 后继节点的执行处理器
                        max_val = 0.0
                        for succ_id, comm_cost in task.successors.items():
                            succ_task = self.tasks[succ_id]
                            # 数据跨节点传输需考虑通信开销，若是同一节点则设0 (关键路径预判逻辑中的通信消减)
                            actual_comm_cost = comm_cost if p_k != p_w else 0.0
                            
                            val = succ_task.oct[p_w] + succ_task.comp_costs[p_w] + actual_comm_cost
                            if val > max_val:
                                max_val = val
                        
                        if max_val < min_val:
                            min_val = max_val
                    
                    task.oct[p_k] = min_val

        # 计算优先级 rank_oct = 平均值
        P = len(self.processor_ids)
        for task in self.tasks.values():
            task.rank_oct = sum(task.oct[p_k] for p_k in self.processor_ids) / P

        # 按照 rank_oct 降序排列得到调度顺序
        self.scheduling_order = sorted(list(self.tasks.values()), key=lambda t: t.rank_oct, reverse=True)

    def phase2_processor_selection(self):
        """
        阶段 2: 前向模拟，按排序依次处理任务，选择令综合预期(O_EFT)最优的处理器
        """
        for task in self.scheduling_order:
            best_processor = None
            min_oeft = float('inf')
            best_aft = 0.0
            best_ast = 0.0

            for p in self.processors:
                p_j = p.proc_id

                # EST(t_i, p_j) = max(T_Available(p_j), max_{n_m in pred(t_i)} (AFT(n_m) + c_{m,i}))
                max_pred_time = 0.0
                for pred_id, comm_cost in task.predecessors.items():
                    pred_task = self.tasks[pred_id]
                    # 若前驱和自己均在 p_j 处理器，则通信开销为 0
                    actual_comm_cost = comm_cost if pred_task.assigned_processor != p_j else 0.0
                    pred_finish_time = pred_task.aft + actual_comm_cost
                    if pred_finish_time > max_pred_time:
                        max_pred_time = pred_finish_time

                est = max(p.available_time, max_pred_time)
                
                # EFT(t_i, p_j) = EST(t_i, p_j) + w_{i,j}
                eft = est + task.comp_costs[p_j]

                # O_EFT(t_i, p_j) = EFT(t_i, p_j) + OCT(t_i, p_j)
                o_eft = eft + task.oct[p_j]

                # 记录使得 O_EFT 最小的处理情况
                if o_eft < min_oeft:
                    min_oeft = o_eft
                    best_processor = p
                    best_aft = eft
                    best_ast = est

            # 执行最终调度记录
            task.assigned_processor = best_processor.proc_id
            task.ast = best_ast
            task.aft = best_aft
            best_processor.available_time = best_aft

    def schedule(self):
        """
        运行整体算法，返回所有任务的执行记录和整体的 Makespan
        """
        self.phase1_task_prioritizing()
        self.phase2_processor_selection()

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


# ============== 本地测试 / JSON 数据示例 ==============
if __name__ == "__main__":
    # 解析来自云端或者其他入口的一组 JSON 描述：包含 Node ID, Dependencies, Duration / Costs 等
    sample_dag_json = {
        "processors": ["P1", "P2", "P3"],
        "tasks": {
            "T1": {
                "comp_costs": {"P1": 14, "P2": 16, "P3": 9},
                "dependencies": {}
            },
            "T2": {
                "comp_costs": {"P1": 13, "P2": 19, "P3": 18},
                "dependencies": {"T1": 18}
            },
            "T3": {
                "comp_costs": {"P1": 11, "P2": 13, "P3": 19},
                "dependencies": {"T1": 12}
            },
            "T4": {
                "comp_costs": {"P1": 13, "P2": 8, "P3": 17},
                "dependencies": {"T2": 27, "T3": 31}
            }
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
