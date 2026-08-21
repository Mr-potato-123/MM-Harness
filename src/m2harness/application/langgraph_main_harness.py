"""Durable LangGraph runtime for the M2Harness top-level workflow.

``MainHarness`` remains the domain authority: it validates DAG transitions,
leases the ``solve_problem`` tool, persists reports, and enforces the two-round
revision policy.  This module supplies the framework runtime around that
authority, giving operators durable graph checkpoints and a stream of explicit
dispatch/compose events without duplicating the domain state machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypedDict

from m2harness.application.main_harness import MainHarness, MainHarnessState, PaperComposerPort
from m2harness.domain.dag import DAGTaskKind, DAGTaskTable
from m2harness.domain.solve_problem import SolveProblemContext


class MainHarnessGraphState(TypedDict, total=False):
    run_state: dict[str, Any]
    halt_reason: str


class LangGraphMainHarness:
    """Run one Main Harness DAG on a checkpointed LangGraph thread."""

    def __init__(
        self,
        main_harness: MainHarness,
        *,
        checkpoint_path: Path,
        context_for_task: Callable[[str, MainHarnessState], SolveProblemContext | None] | None = None,
    ) -> None:
        self.main_harness = main_harness
        self.checkpoint_path = checkpoint_path.resolve()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.context_for_task = context_for_task or (lambda _task_id, _state: None)
        self._checkpoint_context: Any | None = None
        self._checkpointer: Any | None = None
        self._graph: Any | None = None

    def close(self) -> None:
        if self._checkpoint_context is not None:
            self._checkpoint_context.__exit__(None, None, None)
            self._checkpoint_context = None
            self._checkpointer = None
            self._graph = None

    def _ensure_graph(self, composer: PaperComposerPort, max_iterations: int) -> Any:
        if self._graph is not None:
            return self._graph
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError("LangGraph Main Harness requires langgraph-checkpoint-sqlite") from exc

        self._checkpoint_context = SqliteSaver.from_conn_string(str(self.checkpoint_path))
        self._checkpointer = self._checkpoint_context.__enter__()
        self._checkpointer.setup()
        builder = StateGraph(MainHarnessGraphState)

        def dispatch_node(graph_state: MainHarnessGraphState) -> dict[str, Any]:
            state = MainHarnessState.model_validate(graph_state["run_state"], strict=False)
            ready = self.main_harness.ready_tasks(state)
            task_id = next(
                (
                    item.id for item in state.dag.tasks
                    if item.id in ready and item.kind == DAGTaskKind.SOLVE_PROBLEM
                ),
                None,
            )
            if task_id is None:
                return {"run_state": state.model_dump(mode="json"), "halt_reason": self._halt_reason(state)}
            context = self.context_for_task(task_id, state)
            updated = self.main_harness.dispatch(state, task_id, context=context, max_iterations=max_iterations)
            return {"run_state": updated.model_dump(mode="json")}

        def compose_node(graph_state: MainHarnessGraphState) -> dict[str, Any]:
            state = MainHarnessState.model_validate(graph_state["run_state"], strict=False)
            updated = self.main_harness.generate_paper(state, composer)
            return {"run_state": updated.model_dump(mode="json")}

        def route(graph_state: MainHarnessGraphState) -> str:
            state = MainHarnessState.model_validate(graph_state["run_state"], strict=False)
            if state.terminal:
                return END
            ready = self.main_harness.ready_tasks(state)
            solve_ready = [
                item.id for item in state.dag.tasks
                if item.id in ready and item.kind == DAGTaskKind.SOLVE_PROBLEM
            ]
            if solve_ready:
                return "dispatch"
            solve_nodes = [item.id for item in state.dag.tasks if item.kind == DAGTaskKind.SOLVE_PROBLEM]
            records = {item.task_id: item for item in state.tasks}
            if all(records[item].status.value == "completed" for item in solve_nodes):
                return "compose"
            return END

        builder.add_node("dispatch", dispatch_node)
        builder.add_node("compose", compose_node)
        builder.add_edge(START, "dispatch")
        builder.add_conditional_edges("dispatch", route, {"dispatch": "dispatch", "compose": "compose", END: END})
        builder.add_edge("compose", END)
        self._graph = builder.compile(checkpointer=self._checkpointer)
        return self._graph

    def run(
        self,
        problem: str,
        dag: DAGTaskTable,
        *,
        composer: PaperComposerPort,
        max_iterations: int = 3,
        run_id: Any | None = None,
    ) -> MainHarnessState:
        state = self.main_harness.start(problem, dag, run_id=run_id)
        graph = self._ensure_graph(composer, max_iterations)
        config = {
            "configurable": {"thread_id": str(state.run_id)},
            "metadata": {"m2h_run_id": str(state.run_id), "m2h_runtime": "langgraph-main-harness"},
            "recursion_limit": max(32, len(dag.tasks) * (max_iterations + 3) + 8),
        }
        try:
            result = graph.invoke({"run_state": state.model_dump(mode="json")}, config=config)
        except Exception:
            # A failed graph invocation is no longer an active run; release
            # the SQLite handle so a supervisor can inspect/retry it on
            # Windows as well as POSIX.  We never cancel a still-running
            # invocation from an observation path.
            self.close()
            raise
        return MainHarnessState.model_validate(result["run_state"], strict=False)

    @staticmethod
    def _halt_reason(state: MainHarnessState) -> str:
        failed = [item for item in state.tasks if item.status.value in {"failed", "blocked"}]
        if failed:
            return "; ".join(f"{item.task_id}: {item.last_error or item.status.value}" for item in failed)[:4_000]
        return "no ready solve_problem task remains"
