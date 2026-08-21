from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from m2harness.application.langgraph_main_harness import LangGraphMainHarness
from m2harness.application.main_harness import MainHarness, MainHarnessState, MainTaskStatus
from m2harness.domain.dag import canonical_main_harness_dag
from m2harness.domain.solve_problem import SolveProblemReport, SolveProblemStatus
from m2harness.models import ArtifactKind, ProducedArtifact, ReportPayload


class _FakeMainHarness(MainHarness):
    def __init__(self) -> None:
        super().__init__(tool_runtime=None, capabilities=None)

    def dispatch(self, state, task_id, *, context=None, max_iterations=3):
        report = SolveProblemReport(
            task_id=task_id, status=SolveProblemStatus.COMPLETED, iteration_count=1,
            final_report=ReportPayload(title="solve", markdown="# solve", summary="ok"),
        )
        tasks = tuple(
            item.model_copy(update={"status": MainTaskStatus.COMPLETED}) if item.task_id == task_id else item
            for item in state.tasks
        )
        return state.model_copy(update={"tasks": tasks, "reports": (*state.reports, report), "version": state.version + 1})

    def generate_paper(self, state, composer):
        report = ReportPayload(title="t", markdown="# t", summary="s")
        latex = ProducedArtifact(
            logical_name="paper.tex", kind=ArtifactKind.FINAL_LATEX_PAPER,
            media_type="text/x-tex", text="\\documentclass{article}\\begin{document}x\\end{document}",
        )
        terminal = next(item for item in state.tasks if item.task_id == state.dag.terminal_task_id)
        tasks = tuple(
            item.model_copy(update={"status": MainTaskStatus.COMPLETED}) if item.task_id == terminal.task_id else item
            for item in state.tasks
        )
        return state.model_copy(update={"tasks": tasks, "final_report": report, "final_latex_paper": latex, "terminal": True})


class LangGraphRuntimeTests(unittest.TestCase):
    def test_main_harness_graph_dispatches_and_composes(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = LangGraphMainHarness(_FakeMainHarness(), checkpoint_path=Path(directory) / "checkpoints.sqlite")
            try:
                state = runner.run(
                    "problem", canonical_main_harness_dag(scope="x", task_problem="problem", question_count=1),
                    composer=object(),
                )
                self.assertTrue(state.terminal)
                self.assertEqual(state.final_latex_paper.kind, ArtifactKind.FINAL_LATEX_PAPER)
            finally:
                runner.close()
