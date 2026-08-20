# M²Harness — Architecture v0.2

> **A Process-Aware Harness for Mathematical Modeling**  
> 面向数学建模竞赛与开放式建模任务的进程感知 AI Harness

**项目代号：** `M2Harness`  
**版本：** v0.2  
**核心定位：** Process-Aware Harness + Modeling Agent + Coding Agent + Persistent Reports + Skills

---

## 1. 项目目标

M²Harness 的目标不是构建一个固定的“建模 Agent → 代码 Agent → 论文 Agent”流水线，而是构建一个能够：

- 感知数学建模任务当前进程；
- 管理问题之间的依赖关系；
- 对复杂问题进行多视角建模探索；
- 将建模设计与代码执行隔离；
- 根据真实运行结果进行返修；
- 将关键过程持久化为 Report / Artifact；
- 最终由拥有最高上下文权限的 Main Harness 完成论文综合；

的 **Mathematical Modeling Harness**。

系统面向高教社杯、MCM/ICM 等数学建模竞赛，同时希望能够扩展至一般开放式数学建模任务。

---

# 2. 核心设计判断

M²Harness 不采用传统的：

```text
Modeling Agent
    ↓
Coding Agent
    ↓
Paper Agent
```

这种固定角色流水线。

系统真正的核心是：

```text
Main Harness
    │
    ├── DAG / TODO
    ├── Project State
    ├── Context Management
    ├── Artifact / Report Index
    ├── Subagent Dispatch
    ├── Progress Awareness
    └── Final Paper Synthesis
```

而建模与代码执行分别作为高上下文、高专业度的隔离任务，由专用 Subagent 承担：

```text
Model Agent
Coding Agent
```

它们的主要价值不仅是“专业分工”，更重要的是：

> **Context Isolation**

---

# 3. 总体架构

```text
                         ┌────────────────────────────┐
                         │        MAIN HARNESS        │
                         │                            │
                         │ observe / plan / dispatch  │
                         │ track / rollback / replan  │
                         │ final paper synthesis      │
                         └─────────────┬──────────────┘
                                       │
                       ┌───────────────┼───────────────┐
                       │               │               │
                       ▼               ▼               ▼
                  DAG / TODO      Project State      Skills
                       │               │               │
                       │          Report Index         ├─ paper
                       │          Artifact Index       ├─ plotting
                       │          Decision Log         ├─ LaTeX/Typst
                       │          Risk / Issues        ├─ QA
                       │                               └─ utilities
                       │
                       ▼
                  Current Problem
                       │
                       ▼
                 MODEL AGENT
                       │
            modeling exploration
                       │
            Unified Modeling Report
                       │
                       ▼
                 CODING AGENT
                       │
            implementation / run
                       │
                Coding Report
                       │
                       ▼
                 MODEL AGENT
                       │
                approve / revise
                   /         \
                  /           \
             approve         revise
                │               │
                │               └────→ Coding Agent
                ▼
          Final Question Report
                │
                ▼
             Main Harness
                │
           update DAG/state
                │
                ▼
             Next Problem
```

---

# 4. Main Harness

Main Harness 是全局唯一的 **Process Owner**。

它不承担具体的数学建模推导，也不应持续参与具体代码实现。

它主要回答：

> **What should happen next?**

其职责包括：

1. 读取当前 Project State；
2. 管理任务 DAG；
3. 判断当前问题是否已经完成；
4. 决定下一步调用 Model Agent、Coding Agent 或某个 Skill；
5. 管理 Report 与 Artifact；
6. 管理跨问题依赖；
7. 发现需要 rollback / replan 的情况；
8. 控制上下文披露；
9. 最终完成论文综合。

## 4.1 Main Harness 的核心状态

建议长期维护：

```text
.project/
├── project_state.yaml
├── dag.yaml
├── report_index.yaml
├── artifact_index.yaml
├── decision_log.jsonl
├── issue_log.yaml
└── risk_registry.yaml
```

Main Harness 默认只加载高价值状态，而不是整个项目历史。

---

# 5. DAG：问题完成意味着“完整结果闭环完成”

M²Harness 不采用：

```text
Q1 model finished
    ↓
Q2 model finished
    ↓
Q3 model finished
    ↓
统一 coding
```

而采用：

```text
Q1
 ↓
Modeling
 ↓
Coding
 ↓
Result
 ↓
Model Review
 ↓
Final Q1 Report
 ↓
Q1 DONE
 ↓
unlock dependent nodes
```

因此，一个问题的 DAG 节点只有在：

> **最终建模方案已经被真实实现、得到结果、经过 Model Agent 审核，并形成完整问题报告**

后才视为完成。

## 5.1 DAG 不只是 Todo List

例如：

```text
Q1
├────→ Q2
└────→ Q3
         ↓
        Q4
```

DAG 还应知道下游依赖什么。

例如：

```yaml
edge:
  from: Q1
  to: Q3
  requires:
    - key_parameters
    - processed_dataset
    - final_conclusion
```

因此下游问题默认读取的是 Q1 的最终成果，而不是 Q1 的全部历史上下文。

---

# 6. Model Agent

Model Agent 是整个系统中真正负责：

> **Mathematical Modeling Decision**

的主体。

它不是简单 Planner，也不是仅仅从知识库里返回一个模型名称。

它需要拥有：

- 数学建模知识；
- HMML / 建模知识库检索；
- Deep Research；
- 文献检索；
- 数据初步读取；
- 基础统计；
- 轻量代码执行；
- 简单聚类 / 拟合 / 验证；
- 数学推导；
- 候选方案探索；
- 建模方案比较；
- 建模风险判断。

## 6.1 Model Agent 必须具备初步验证能力

Model Agent 不应该只进行文本推理。

对于真实数据任务，它应能够进行必要的轻量探索，例如：

```text
descriptive statistics
correlation analysis
distribution inspection
K-Means
PCA
simple regression
quick optimization prototype
basic visualization
```

这些操作的目的不是替代 Coding Agent，而是：

> **让 Model Agent 在正式交付方案之前，至少有真实数据证据支撑自己的判断。**

例如：

```text
题目
 ↓
初步统计
 ↓
发现明显分群
 ↓
K-Means quick check
 ↓
确认存在结构
 ↓
正式设计分群 + 后续优化方案
```

---

# 7. Modeling Report

Model Agent 的核心交付物不是高度离散化的 `model_spec.yaml`，而是：

> **Modeling Report**

结构化是为了方便传递和审计，但不能把开放式数学建模压缩成配置文件。

建议包含：

```text
# Modeling Report

## 1. Problem Understanding
## 2. Initial Data Analysis
## 3. Key Observations
## 4. Assumptions
## 5. Candidate Modeling Ideas
## 6. Preliminary Validation
## 7. Recommended Main Scheme
## 8. Mathematical Formulation
## 9. Expected Data Processing
## 10. Expected Computational Procedure
## 11. Required Validation
## 12. Error Analysis Requirements
## 13. Sensitivity / Robustness Requirements
## 14. Expected Results
## 15. Expected Figures
## 16. Required Outputs for Downstream Problems
## 17. Risks / Uncertainties
```

---

# 8. Model Agent 必须给出“主方案 + 预期输出”

Model Agent 不能只说：

> “可以考虑 XGBoost、GAM 或随机森林。”

最终必须生成一个可执行的 **主方案**。

例如：

```text
Main Scheme

1. 对原始数据进行异常值和缺失值处理；
2. 对变量进行标准化；
3. 使用聚类识别样本群体；
4. 分群后建立回归模型；
5. 使用 baseline 进行对比；
6. 对关键参数进行敏感性分析；
7. 对主要误差来源进行统计；
8. 输出最终预测与决策指标。
```

同时还需要说明预期。

### Expected Processing

```text
缺失值处理
异常值诊断
标准化
特征构造
聚类
模型拟合
```

### Expected Results

例如：

```text
预计能够识别 3~4 个具有明显差异的群体；
预计模型应显著优于简单线性 baseline；
若不存在明显提升，应重新评估分群建模假设。
```

### Expected Figures

例如：

```text
- 数据分布图
- 聚类二维投影图
- baseline vs main model 对比图
- 实际值 / 预测值图
- 残差分布图
- 参数敏感性图
```

因此：

> **Model Agent 决定需要证明什么，以及期望形成什么证据。**

---

# 9. Coding Agent

Coding Agent 是：

> **Computational Realization Owner**

它的核心职责是：

> 根据 Modeling Report，高质量地将主方案实现、运行、调试、可视化，并忠实记录真实计算中发生的一切。

它主要回答：

> **How should this modeling plan be realized computationally?**

## 9.1 Coding Agent 的职责

包括：

```text
读取 Modeling Report
        ↓
理解主方案
        ↓
实现数据处理
        ↓
实现模型 / 求解过程
        ↓
运行与调试
        ↓
完成指定分析
        ↓
生成结果
        ↓
生成高质量图表
        ↓
形成 Coding Report
```

## 9.2 Coding Agent 的自由度

Coding Agent 具有实现层面的自主权。

例如 Model Agent 要求：

> 对参数 λ 进行敏感性分析。

Coding Agent 可以自主决定：

- λ 的合理搜索范围；
- 采样密度；
- 采用折线图、heatmap 还是 contour；
- 是否补充置信区间；
- 是否增加一个有价值的诊断图；
- 如何组织代码；
- 如何优化运行效率。

但它不应擅自删除 Model Agent 明确要求的关键验证。

---

# 10. 分析需求由 Model Agent 提出

以下内容原则上属于 **Modeling Decision**：

- 是否需要误差分析；
- 重点分析哪些误差；
- 哪些变量需要敏感性分析；
- 是否需要稳定性验证；
- 哪些 baseline 是必要的；
- 哪些假设必须验证；
- 什么结果会否定当前建模方案。

因此应由 Model Agent 在 Modeling Report 中提出。

Coding Agent 的任务是：

> **选择合适的计算和可视化方式将这些需求落实。**

---

# 11. 图像：Model Agent 定义目的，Coding Agent 负责实现与表达

图表设计采用双层职责。

## Model Agent

定义：

> 为什么需要这张图？

例如：

```text
需要展示模型误差是否随目标值增大而系统性上升。
```

或：

```text
需要展示参数 λ 对目标函数和最终决策的敏感性。
```

并可以给出预期图像类型。

## Coding Agent

负责：

> 怎样最好地画出来？

它可以决定：

- scatter；
- line plot；
- heatmap；
- contour；
- confidence band；
- boxplot；
- residual plot；
- 多面板组合图；

以及：

- layout；
- 轴标签；
- annotation；
- 图例；
- 输出格式；
- publication-quality styling。

因此 Coding Agent 同时承担：

> **turn computation into evidence**

的职责。

---

# 12. Every Run Leaves a Report

M²Harness 的硬原则之一：

> **No meaningful execution without a durable report.**

每次重要 Coding 执行，都必须生成一个 Coding Report。

例如：

```text
Q1/coding/run_001/
├── coding_report.md
├── source/
├── outputs/
├── figures/
└── logs/
```

---

# 13. Coding Report

Coding Report 应至少包含：

```text
# Coding Report

## 1. Implementation Summary
## 2. Data Processing
## 3. Deviations from Modeling Report
## 4. Execution
## 5. Results
## 6. Figures
## 7. Required Analyses
## 8. Additional Findings
## 9. Problems / Limitations
## 10. Recommendation
```

其中需要明确记录：

- 实际实现了什么；
- 数据如何处理；
- 哪些地方与 Modeling Report 不同；
- 运行了哪些程序与实验；
- 主要数值结果；
- 生成了哪些图；
- 误差、敏感性、稳健性等要求是否完成；
- Coding Agent 自主发现的有价值现象；
- 实际计算限制；
- 是否建议交回 Model Agent 审核。

---

# 14. Coding Agent 可以主动发现问题

Coding Agent 不是机械 executor。

例如，它可以返回：

```text
原方案使用精确整数规划。
按照当前问题规模，估计求解时间超过 12 小时。
```

或者：

```text
预期聚类结构没有出现，Silhouette Score 很低。
```

或者：

```text
敏感性分析显示模型对参数 α 极度不稳定。
```

这些都属于重要反馈。

但 Coding Agent 不负责最终决定：

> 是否应该放弃模型。

这个判断返回给 Model Agent。

---

# 15. Model ↔ Coding Review Loop

一个问题内部的核心闭环是：

```text
Modeling Report
       ↓
Coding Agent
       ↓
Coding Report + Results
       ↓
Model Agent
       ↓
Review
   /        \
Approve    Revise
  │           │
  │           └────→ Coding Agent
  ▼
Final Question Report
```

## 15.1 Model Agent Review

Model Agent 收到：

- Coding Report；
- 主要结果；
- 图表；
- 实现偏差；
- 额外发现；

后进行建模层审核。

它判断：

```text
APPROVE
```

或者：

```text
REVISION REQUIRED
```

返修可以包括：

- 修改实现；
- 补实验；
- 补误差分析；
- 调整参数；
- 修改数据处理；
- 改变模型部分结构；
- 必要时重新进行建模。

---

# 16. Final Question Report

当 Model Agent 最终批准结果后，需要将：

1. 原始建模分析；
2. 最终 Modeling Report；
3. Coding Agent 的真实运行结果；
4. 图表；
5. 误差 / 敏感性 / 稳健性分析；
6. 最终代码索引；

汇总成：

> **Final Question Report**

建议结构：

```text
# Q1 Final Report

## Problem Analysis
## Data Analysis
## Assumptions
## Modeling Rationale
## Mathematical Formulation
## Computational Method
## Results
## Error Analysis
## Sensitivity / Robustness
## Figures and Interpretation
## Limitations
## Final Conclusions
## Downstream Outputs
## Code / Artifact Index
```

这个报告是交给 Main Harness 的主要上下文。

---

# 17. 双层闭环

M²Harness 存在两个层次的闭环。

## 17.1 问题内部闭环

```text
Model Agent
    ↓
Coding Agent
    ↓
Result
    ↓
Model Review
    ↓
Revision
```

负责：

> **把当前问题真正做好。**

## 17.2 全局 Harness 闭环

```text
Q1 Final Report
       ↓
Main Harness
       ↓
Update DAG / Project State
       ↓
Unlock Q2
       ↓
Dispatch Q2
```

负责：

> **把整个 long-horizon 建模任务推进完成。**

---

# 18. 多路建模探索

对于开放性较强的问题，单一路径 Model Agent 容易产生：

> **Premature Convergence**

即过早锁定一种数学建模方案。

因此 M²Harness 允许 Model Agent 在建模初期进行多路隔离探索。

例如：

```text
               Current Problem
                     │
                Model Agent
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Route A    Route B    Route C
          │          │          │
          ▼          ▼          ▼
    Preliminary Preliminary Preliminary
      Report A    Report B    Report C
          │          │          │
          └──────────┼──────────┘
                     ▼
                  Synthesis
                     │
                     ▼
          Unified Modeling Report
                     │
                     ▼
                Coding Agent
```

---

# 19. 多路探索的目的不是投票

三个探索分支并不是为了：

```text
A: XGBoost
B: XGBoost
C: Random Forest

2 : 1
→ XGBoost wins
```

M²Harness 不采用这种 majority-vote 思路。

多路探索主要用于：

> **提升建模全面性，而不是简单提升“答案正确率”。**

不同分支可以分别发现：

- 不同数学抽象；
- 不同数据结构；
- 不同约束；
- 不同 baseline；
- 不同风险；
- 不同解释角度；
- 不同竞赛叙事价值。

最终通过 **Synthesis** 形成一个统一的主方案。

---

# 20. Preliminary Modeling Report

并行探索阶段产生的是：

> **Preliminary Modeling Report**

而不是三个完整解决方案。

建议每一路只关注：

```text
Problem Interpretation
Potential Modeling Route
Quick Data Evidence
Key Assumptions
Strengths
Weaknesses
Expected Outputs
What Should Be Tested
```

它们的目标是扩大思考空间，而不是独立跑完整工作流。

---

# 21. Unified Modeling Report

多个 Preliminary Report 最终由 Model Agent 汇总为：

> **唯一的 Unified Modeling Report**

例如：

```text
Route A
提供经典可解释 baseline

Route B
发现时间结构，应作为主模型

Route C
提出重要约束与敏感性变量

            ↓

Unified Scheme

主体采用 Route B
加入 Route A 作为 baseline
吸收 Route C 的约束与敏感性设计
```

之后只将该统一方案交给 Coding Agent。

---

# 22. 为什么不让三路完整执行到底

不采用：

```text
A → code → result → final A
B → code → result → final B
C → code → result → final C
               ↓
           aggregator
```

作为默认策略。

原因包括：

1. 成本高；
2. 状态分叉严重；
3. 会产生多套预处理；
4. 多套代码难以统一；
5. 多套结果难以融合；
6. 最终 Aggregator 很难形成一致叙事；
7. 多路探索的核心目标只是避免建模视角过窄。

因此默认采用：

> **parallel exploration → synthesis → single coding track**

---

# 23. 多路探索不是强制步骤

多路探索应视为 Model Agent 的一种推理能力，而不是固定阶段。

例如：

```text
single_explore
```

适用于：

- 问题结构清晰；
- 方法高度确定；
- 数据规模简单。

而：

```text
multi_explore(k=3)
```

适用于：

- 开放问题；
- 多种数学抽象都合理；
- 初始方案选择对后续影响很大；
- 创新性要求高；
- Model Agent 置信度较低。

---

# 24. Report Hierarchy

每个问题建议维护：

```text
Q1/
│
├── modeling/
│   ├── preliminary_A.md
│   ├── preliminary_B.md
│   ├── preliminary_C.md
│   └── modeling_report.md
│
├── coding/
│   ├── run_001/
│   │   ├── coding_report.md
│   │   ├── source/
│   │   ├── outputs/
│   │   └── figures/
│   │
│   └── run_002/
│
└── final/
    └── solution_report.md
```

---

# 25. Progressive Disclosure

Main Harness 默认只读取：

```text
solution_report.md
```

如果需要进一步调查，则逐级展开：

```text
solution_report
      ↓
modeling_report
      ↓
coding_report
      ↓
preliminary reports / code / outputs / logs
```

因此报告体系本身也是：

> **Context Compression Hierarchy**

---

# 26. 最终论文

最终论文不设置独立 Paper Agent。

原因是 Main Harness：

- 拥有最高上下文访问权限；
- 知道整个 DAG；
- 知道各问题之间的关系；
- 知道全部 Final Question Reports；
- 知道最终选择与历史决策。

因此：

```text
Main Harness
    +
Paper Skill
    +
Scientific Plotting Skill
    +
LaTeX / Typst Skill
    +
Citation Skill
    +
QA Skill
```

负责最终论文生成。

---

# 27. Main Harness 的“最高上下文权限”

最高上下文权限不意味着：

> 永远加载所有内容。

而是：

> **有权按需访问全部 Artifact。**

默认上下文可以只包括：

```text
System
Project State
DAG
Current Problem
Accepted Results
Critical Issues
Relevant Report Index
```

需要写论文或处理依赖问题时，再按需加载对应报告。

---

# 28. 核心 Agent 定义

## Main Harness

> **Owns the process.**

决定：

- 当前做到哪里；
- 下一步做什么；
- 调用谁；
- 哪些上下文应该被加载；
- 是否 rollback / replan；
- 如何推进整场比赛。

## Model Agent

> **Owns the modeling decision.**

决定：

- 问题如何理解；
- 模型怎么设计；
- 为什么这样设计；
- 需要什么数据处理；
- 需要验证什么；
- 需要做什么误差 / 敏感性分析；
- 预期得到什么结果；
- 预期需要哪些图；
- Coding 结果是否足以支持模型；
- 是否批准当前方案。

## Coding Agent

> **Owns the computational realization.**

决定：

- 如何高质量实现；
- 如何组织代码；
- 如何调试；
- 如何高效计算；
- 如何完成 Model Agent 指定的分析；
- 如何生成高质量图表；
- 是否需要添加有价值的补充结果；
- 如何忠实报告实现中发生的问题。

---

# 29. 核心交互语义

可以将 M²Harness 的核心执行语义概括成：

```text
Main Harness
      ↓
   Problem
      ↓
Model Agent
      ↓
[Optional Parallel Exploration]
      ↓
Unified Modeling Report
      ↓
Coding Agent
      ↓
Coding Report + Results
      ↓
Model Agent Review
   /             \
revise          approve
  │                │
  └── Coding ──────┘
                   ↓
          Final Question Report
                   ↓
              Main Harness
                   ↓
              DAG Progress
```

---

# 30. 当前 MVP

第一版只需要实现：

## Main Harness

- Project State
- DAG / TODO
- Report Index
- Artifact Index
- Subagent Dispatch
- Context Loading
- Progress Update

## Model Agent

- HMML / Modeling Knowledge
- Deep Research
- Lightweight Data Analysis
- Preliminary Validation
- Optional Multi-Explore
- Modeling Report
- Coding Review
- Final Question Report

## Coding Agent

- Python / MATLAB 等代码生成
- Command Execution
- Debug
- Data Processing
- Model Implementation
- Required Analysis
- Scientific Plotting
- Coding Report

## Skills

- modeling knowledge retrieval
- web / paper research
- data inspection
- scientific plotting
- paper writing
- LaTeX / Typst
- final QA

---

# 31. 暂不优先实现

当前不需要：

- 大量永久 Agent 角色；
- 多 Agent 持续互聊；
- learned router；
- RL orchestrator；
- 每个候选方案端到端并行执行；
- 固定九阶段工作流；
- 独立 Paper Agent；
- 复杂前端 UI。

---

# 32. 当前最重要的三个 Artifact

第一阶段优先做好：

```text
modeling_report.md
coding_report.md
solution_report.md
```

其中：

```text
modeling_report.md
```

代表：

> 当前问题应该怎么做。

```text
coding_report.md
```

代表：

> 真实计算中发生了什么。

```text
solution_report.md
```

代表：

> 经 Model Agent 审核后，该问题最终成立的完整结论是什么。

这三个 Report 构成问题级 Context Engineering 的核心。

---

# 33. 项目差异化

M²Harness 的主要差异不在于“用了几个 Agent”。

而在于：

1. **Main Harness 拥有显式 Process Awareness。**
2. **问题只有在 Modeling–Coding–Review 闭环完成后才算 DONE。**
3. **Model Agent 是真正的建模主体，并拥有轻量验证能力。**
4. **Coding Agent 是计算实现主体，而不是建模决策者。**
5. **所有重要运行都有 Durable Report。**
6. **多路推理用于提升建模全面性，并在 Coding 前合并为唯一主方案。**
7. **Report Hierarchy 天然形成 Progressive Disclosure Context。**
8. **最终论文由拥有全局权限的 Main Harness 综合生成。**

---

# 34. 一句话定义

> **M²Harness is a process-aware mathematical modeling harness where modeling decisions and computational realization are separated into isolated specialist agents, coordinated through persistent reports and iterative model–code review.**

中文：

> **M²Harness 是一个面向开放式数学建模任务的进程感知 Harness，通过隔离的 Model Agent 与 Coding Agent，将建模决策和计算实现分离，并依靠持久化报告与 Model–Code 迭代审核完成长程建模任务。**

---

# 35. 当前架构决策

```text
M²Harness
│
├── Main Harness
│   ├── Process Awareness
│   ├── DAG / TODO
│   ├── Project State
│   ├── Context Management
│   ├── Report / Artifact Index
│   ├── Dispatch
│   └── Final Paper
│
├── Model Agent
│   ├── HMML / Knowledge
│   ├── Deep Research
│   ├── Lightweight Validation
│   ├── Optional Multi-Explore
│   ├── Unified Modeling Report
│   ├── Coding Review
│   └── Final Question Report
│
├── Coding Agent
│   ├── Implementation
│   ├── Execution
│   ├── Debug
│   ├── Required Analysis
│   ├── Scientific Plotting
│   └── Coding Report
│
└── Skills
    ├── Research
    ├── Data Analysis
    ├── Plotting
    ├── Paper Writing
    ├── LaTeX / Typst
    └── QA
```

---

## Next Step

下一阶段最值得定义的不是更多 Agent，而是：

1. `project_state.yaml`
2. `dag.yaml`
3. `modeling_report.md` 模板
4. `coding_report.md` 模板
5. `solution_report.md` 模板
6. Model Agent ↔ Coding Agent 的 review / revision protocol

这些内容确定后，M²Harness 的核心执行语义基本完成。
