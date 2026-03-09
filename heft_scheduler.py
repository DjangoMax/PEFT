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
        self.comp_costs: Dict[str, float] = {}
        # Dependencies: predecessors & successors -> comm_cost
        self.predecessors: Dict[str, float] = {}
        self.successors: Dict[str, float] = {}

        # Upward Rank for HEFT
        # HEFT 的向上秩 (Upward Rank)
        self.rank_u: float = 0.0

        # Allocation result
        self.assigned_processor: str = None
        self.ast: float = 0.0  # Actual Start Time
        self.aft: float = 0.0  # Actual Finish Time

    def __repr__(self):
        return f"Task({self.task_id}, Rank_U: {self.rank_u:.2f}, Assigned: {self.assigned_processor})"


class Processor:
    """
    Represents a processor/computing node.
    表示计算节点
    """
    def __init__(self, proc_id: str):
        self.proc_id = proc_id
        # Tracks the current available time of this processor
        self.available_time: float = 0.0


class HEFT:
    """
    Heterogeneous Earliest Finish Time (HEFT) scheduling algorithm.
    HEFT 调度算法
    """
    def __init__(self, tasks_data: dict, processors_list: List[str]):
        self.processors = [Processor(p) for p in processors_list]
        self.processor_ids = processors_list
        self.tasks: Dict[str, Task] = {}

        # Parse tasks
        for t_id, t_info in tasks_data.items():
            task = Task(t_id)
            task.comp_costs = t_info.get("comp_costs", {})
            self.tasks[t_id] = task

        # Parse dependencies
        for t_id, t_info in tasks_data.items():
            deps = t_info.get("dependencies", {})
            for pred_id, comm_cost in deps.items():
                self.tasks[t_id].predecessors[pred_id] = comm_cost
                if pred_id in self.tasks:
                    self.tasks[pred_id].successors[t_id] = comm_cost

        self.scheduling_order: List[Task] = []

    def topological_sort(self) -> List[str]:
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
        Phase 1: Calculate Upward Rank (rank_u) backward.
        阶段 1: 反向计算 rank_u
        """
        reversed_tasks = self.topological_sort()[::-1]
        P = len(self.processor_ids)

        for t_id in reversed_tasks:
            task = self.tasks[t_id]
            # Average computation cost
            avg_comp_cost = sum(task.comp_costs[p] for p in self.processor_ids) / P
            
            if not task.successors:
                # Exit node
                task.rank_u = avg_comp_cost
            else:
                max_succ_val = 0.0
                for succ_id, comm_cost in task.successors.items():
                    succ_task = self.tasks[succ_id]
                    # HEFT considers average communication cost + successor's rank_u
                    val = comm_cost + succ_task.rank_u
                    if val > max_succ_val:
                        max_succ_val = val
                
                task.rank_u = avg_comp_cost + max_succ_val

        # Sort descending by rank_u
        self.scheduling_order = sorted(list(self.tasks.values()), key=lambda t: t.rank_u, reverse=True)

    def print_rank_table(self):
        print("\n" + "="*40)
        print(f"{'HEFT Rank_u Table':^40}")
        print("="*40)
        print(f"{'Task':<6} | rank_u")
        print("-" * 40)
        sorted_tasks = sorted(self.tasks.values(), key=lambda t: int(t.task_id.replace('T', '')))
        for task in sorted_tasks:
            print(f"{task.task_id:<6} | {task.rank_u:>6.1f}")
        print("="*40 + "\n")

    def phase2_processor_selection(self):
        """
        Phase 2: Forward simulation, assign tasks sequentially to minimize EFT.
        阶段 2: 前向模拟，选择令 EFT 最小的计算节点
        """
        for task in self.scheduling_order:
            best_processor = None
            min_eft = float('inf')
            best_ast = 0.0

            for p in self.processors:
                p_j = p.proc_id

                max_pred_time = 0.0
                for pred_id, comm_cost in task.predecessors.items():
                    pred_task = self.tasks[pred_id]
                    # No comm cost if on same processor
                    actual_comm_cost = comm_cost if pred_task.assigned_processor != p_j else 0.0
                    pred_finish_time = pred_task.aft + actual_comm_cost
                    if pred_finish_time > max_pred_time:
                        max_pred_time = pred_finish_time

                # Non-insertion based EST calculation (similar to PEFT implementation)
                est = max(p.available_time, max_pred_time)
                
                # EFT(t_i, p_j)
                eft = est + task.comp_costs[p_j]

                # In HEFT, we simply pick the processor that minimizes EFT
                if eft < min_eft:
                    min_eft = eft
                    best_processor = p
                    best_ast = est

            # Assign
            task.assigned_processor = best_processor.proc_id
            task.ast = best_ast
            task.aft = min_eft
            best_processor.available_time = min_eft

    def schedule(self):
        self.phase1_task_prioritizing()
        self.print_rank_table()
        self.phase2_processor_selection()

        makespan = max(task.aft for task in self.tasks.values())
        
        results = []
        for task in self.tasks.values():
            results.append({
                "Task ID": task.task_id,
                "Assigned Processor": task.assigned_processor,
                "Start Time": round(task.ast, 2),
                "End Time": round(task.aft, 2),
                "Rank U": round(task.rank_u, 2)
            })

        results.sort(key=lambda x: x["Start Time"])
        return results, makespan

    def visualize_gantt_chart(self, tasks_schedule, final_makespan):
        """
        Visualize Gantt chart for HEFT
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
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
        ax.set_title(f"HEFT Scheduling Gantt Chart (Makespan: {final_makespan:.2f})")
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
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

    scheduler = HEFT(
        tasks_data=sample_dag_json["tasks"],
        processors_list=sample_dag_json["processors"]
    )
    
    tasks_schedule, final_makespan = scheduler.schedule()
    
    print("====== HEFT Algorithm Scheduling Results ======")
    for res in tasks_schedule:
        print(f"Task: {res['Task ID']:<4} | Node: {res['Assigned Processor']:<4} | "
              f"Execution Window: [{res['Start Time']:>5.2f} - {res['End Time']:>5.2f}] | Priority(Rank_U): {res['Rank U']}")
    print("-" * 55)
    print(f"Total Makespan(Max AFT): {final_makespan:.2f}")

    scheduler.visualize_gantt_chart(tasks_schedule, final_makespan)
