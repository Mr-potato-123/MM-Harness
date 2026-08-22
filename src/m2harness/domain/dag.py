"""Validated task-table contracts for the Main Agent Harness.

The planner may choose a serial chain, a dependency DAG, or a TODO-like graph,
but the Harness owns the validated graph, dependency unlocking, and publication
terminal.  A normal executable node is a ``solve_problem`` Tool call.  The
older stage graph is retained as a compatibility contract for the original
durable workflow.  Conditional solve revisions never change the publication
terminal node.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DAGTaskKind(StrEnum):
    INGEST = "ingest"
    SOLVE_PROBLEM = "solve_problem"
    MODEL = "model"
    CODE = "code"
    REVIEW = "review"
    PUBLISH_LATEX = "publish_latex"


class DAGTaskNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=1, max_length=200)
    kind: DAGTaskKind
    depends_on: tuple[str, ...] = ()
    dependency_outputs: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    required_capabilities: tuple[str, ...] = ()
    output_contract: str = Field(min_length=1, max_length=200)
    terminal: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class DAGTaskTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["m2harness/dag/v1"] = "m2harness/dag/v1"
    tasks: tuple[DAGTaskNode, ...] = Field(min_length=1, max_length=100)
    terminal_task_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph_and_publication_terminal(self) -> "DAGTaskTable":
        nodes = {task.id: task for task in self.tasks}
        if len(nodes) != len(self.tasks):
            raise ValueError("DAG task IDs must be unique")
        for task in self.tasks:
            unknown = sorted(set(task.depends_on) - nodes.keys())
            if unknown:
                raise ValueError(f"task {task.id} depends on unknown task(s): {', '.join(unknown)}")
            if task.id in task.depends_on:
                raise ValueError(f"task {task.id} cannot depend on itself")
            undeclared = sorted(set(task.dependency_outputs) - set(task.depends_on))
            if undeclared:
                raise ValueError(f"task {task.id} requests outputs from non-dependency task(s): {', '.join(undeclared)}")
        if self.terminal_task_id not in nodes:
            raise ValueError("terminal_task_id must reference a task")
        terminal = nodes[self.terminal_task_id]
        if terminal.kind != DAGTaskKind.PUBLISH_LATEX or not terminal.terminal:
            raise ValueError("the terminal DAG task must be terminal publish_latex")
        outgoing = [task.id for task in self.tasks if self.terminal_task_id in task.depends_on]
        if outgoing:
            raise ValueError("terminal publish_latex task cannot have downstream tasks")
        marked_terminal = [task.id for task in self.tasks if task.terminal]
        if marked_terminal != [self.terminal_task_id] and set(marked_terminal) != {self.terminal_task_id}:
            raise ValueError("exactly the terminal task must be marked terminal")
        order = self.topological_order()
        if order[-1] != self.terminal_task_id:
            raise ValueError("the publication terminal must be the last task in topological order")
        reachable = self._reachable_from_roots(nodes)
        if reachable != set(nodes):
            missing = sorted(set(nodes) - reachable)
            raise ValueError("DAG contains unreachable task(s): " + ", ".join(missing))
        return self

    def topological_order(self) -> tuple[str, ...]:
        """Return a deterministic order or raise on a cycle."""

        nodes = {task.id: task for task in self.tasks}
        indegree = {task.id: len(task.depends_on) for task in self.tasks}
        children: dict[str, list[str]] = {task.id: [] for task in self.tasks}
        for task in self.tasks:
            for dependency in task.depends_on:
                children[dependency].append(task.id)
        ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for child in sorted(children[current]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(order) != len(nodes):
            raise ValueError("DAG task table contains a dependency cycle")
        return tuple(order)

    @staticmethod
    def _reachable_from_roots(nodes: dict[str, DAGTaskNode]) -> set[str]:
        children: dict[str, list[str]] = {task_id: [] for task_id in nodes}
        for task in nodes.values():
            for dependency in task.depends_on:
                children[dependency].append(task.id)
        roots = [task_id for task_id, task in nodes.items() if not task.depends_on]
        seen: set[str] = set()
        stack = list(roots)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(children[current])
        return seen


def canonical_single_question_dag() -> DAGTaskTable:
    """Return the immutable mainline graph used for every new question."""

    return DAGTaskTable(
        tasks=(
            DAGTaskNode(
                id="ingest", title="Multimodal intake", kind=DAGTaskKind.INGEST,
                output_contract="problem_brief", required_capabilities=("document.read", "image.inspect", "artifact.read"),
            ),
            DAGTaskNode(
                id="model", title="Model formulation", kind=DAGTaskKind.MODEL,
                depends_on=("ingest",), output_contract="modeling_report", required_capabilities=("modeling.formulate",),
            ),
            DAGTaskNode(
                id="code", title="Local implementation and execution", kind=DAGTaskKind.CODE,
                depends_on=("model",), output_contract="coding_evidence", required_capabilities=("python.execute", "artifact.write"),
                metadata={"revision_policy": "review_may_return_to_code"},
            ),
            DAGTaskNode(
                id="review", title="Independent evidence review", kind=DAGTaskKind.REVIEW,
                depends_on=("code",), output_contract="review_verdict", required_capabilities=("report.review",),
            ),
            DAGTaskNode(
                id="publish-paper", title="Final report and LaTeX paper", kind=DAGTaskKind.PUBLISH_LATEX,
                depends_on=("review",), output_contract="final_question_report+final_latex_paper",
                required_capabilities=("report.finalize", "report.publish"), terminal=True,
            ),
        ),
        terminal_task_id="publish-paper",
    )


def canonical_main_harness_dag(
    task_id: str = "q1",
    *,
    scope: str = "question-1",
    task_problem: str | None = None,
    task_problems: Mapping[str, str] | None = None,
    question_count: int = 1,
) -> DAGTaskTable:
    """Return a TODO graph whose solve nodes are independently scoped.

    Unlike the legacy stage graph above, this is the graph the top-level Agent
    should normally construct: one or more serial ``solve_problem`` nodes
    followed by the Harness-owned publication terminal. ``question_count=1``
    preserves the compatibility graph used by the unit tests; the production
    Qwen CLI uses four nodes so the four numbered questions are solved one at a
    time, with each node receiving only the previous node's compressed output.
    """

    if not task_id:
        raise ValueError("task_id cannot be empty")
    if not scope.strip():
        raise ValueError("scope cannot be empty")
    if question_count < 1 or question_count > 4:
        raise ValueError("question_count must be between 1 and 4")
    if question_count > 1 and task_id != "q1":
        raise ValueError("multi-question canonical graph requires task_id='q1'")
    metadata: dict[str, Any] = {
        "scope": scope.strip(),
        "todo": f"Solve only {scope.strip()} and return its reviewed solve_problem_report.",
        "question_number": 1 if scope.strip().lower() in {"question-1", "question 1", "first-question", "first question"} else None,
        "exploration_mode": "auto",
        "difficulty": 1,
    }
    if task_problem is not None:
        if not task_problem.strip():
            raise ValueError("task_problem cannot be empty when provided")
        metadata["problem"] = task_problem
    solve_nodes: list[DAGTaskNode] = []
    for index in range(1, question_count + 1):
        node_id = task_id if question_count == 1 else f"q{index}"
        node_scope = scope.strip() if question_count == 1 else f"question-{index}"
        node_metadata = dict(metadata)
        node_metadata["scope"] = node_scope
        node_metadata["question_number"] = index
        node_metadata["todo"] = f"Solve only {node_scope} and return its reviewed solve_problem_report."
        scoped_problem = (task_problems or {}).get(node_id, task_problem)
        if scoped_problem is not None:
            node_metadata["problem"] = (
                scoped_problem if question_count == 1 else
                scoped_problem + f"\n\nSCOPE LOCK: analyze only {node_scope}; do not solve or summarize any other numbered question."
            )
        solve_nodes.append(DAGTaskNode(
            id=node_id,
            title="解决问题" if question_count == 1 else f"解决第 {index} 问",
            kind=DAGTaskKind.SOLVE_PROBLEM,
            depends_on=(() if index == 1 else (f"q{index - 1}",)),
            output_contract="solve_problem_report",
            required_capabilities=("problem.solve",),
            metadata=node_metadata,
        ))
    last_id = solve_nodes[-1].id
    return DAGTaskTable(
        tasks=(*solve_nodes, DAGTaskNode(
            id="publish-paper", title="汇总最终报告与 LaTeX 论文",
            kind=DAGTaskKind.PUBLISH_LATEX, depends_on=(last_id,), terminal=True,
            output_contract="final_question_report+final_latex_paper",
            required_capabilities=("report.finalize", "report.publish"),
        )),
        terminal_task_id="publish-paper",
    )
