# PEFT Scheduling Algorithm

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A Predictive Earliest Finish Time (PEFT) scheduling algorithm implementation for computing nodes.*

[English](#english) | [中文说明](#chinese)

---

<h2 id="english">English Description</h2>

### Introduction
This repository contains a Python implementation of the **Predictive Earliest Finish Time (PEFT)** scheduling algorithm. It is designed to optimize the execution of Directed Acyclic Graphs (DAGs) in heterogeneous computing environments (such as cloud architectures like Azure/AWS) by minimizing the overall Makespan (total execution time).

### Features
* **Phased Execution**:
    1. **Backward Calculation (Task Prioritizing)**: Computes the Optimistic Cost Table (OCT) starting from the exit node to evaluate task priorities.
    2. **Forward Simulation (Processor Selection)**: Schedules tasks sequentially by predicting the best processor node offering the minimal Optimistic Expected Finish Time (O_EFT).
* **Network Cost Reduction**: Smartly eliminates transmission costs if dependent tasks are scheduled on the same computing node.
* **Extensibility**: Easy to plug in your custom JSON DAG descriptions.

### Quick Start
To run the included sample calculation:

```bash
# Clone the repository
git clone https://github.com/DjangoMax/PEFT.git

# Enter the directory
cd PEFT

# Run the algorithm script
python peft_scheduler.py
```

---

<h2 id="chinese">中文说明</h2>

### 简介
本项目是一个 **PEFT (Predictive Earliest Finish Time)** 调度算法的 Python 实现。该算法主要用于在异构计算环境（如 Azure/AWS 等云基础设施）中优化有向无环图（DAG）形式的任务执行，其核心目标是最小化总执行时间，即 Makespan。

### 核心特性
* **两阶段调度过程**：
    1. **反向计算评估（任务优先级划分）**：从出口节点反向遍历，计算 OCT 表单 (Optimistic Cost Table) 并据此得出所有任务的优先级 rank。
    2. **前向模拟分配（处理器选择）**：按优先级依次处理任务，通过预判综合预期完成时间最优解 (O_EFT)，为每个任务分配最佳计算节点。
* **网络通讯开销消减**：如果在当前分配计划中，存在依赖关系的上下游任务被分配到了同一计算节点上，实际的跨节点传输开销将被自动免除。
* **高扩展性**：支持以 JSON 格式快速加载自定义的任务依赖和开销数据。

### 快速开始
运行项目自带的 DAG 样例：

```bash
# 克隆仓库到本地电脑
git clone https://github.com/DjangoMax/PEFT.git

# 进入项目目录
cd PEFT

# 运行调度脚本
python peft_scheduler.py
```
