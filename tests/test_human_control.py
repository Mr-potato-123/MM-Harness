from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from m2harness.application.solve_problem import SolveProblemService
from m2harness.domain.solve_problem import SolveProblemContext, SolveProblemTask
from m2harness.human_control import HumanControlStore, HumanInterruptRequested
from m2harness.infrastructure.local_sandbox import LocalSandboxClient
from m2harness.toy_frontend import _Handler


class HumanControlTest(unittest.TestCase):
    def test_suggestion_is_consumed_at_a_solve_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = HumanControlStore(root)
            service = SolveProblemService(object(), object(), human_control=store)
            task = SolveProblemTask(task_id="q1", title="Q1", problem="P")
            context = SolveProblemContext(metadata={"run_id": "run-1"})

            store.submit_suggestion("run-1", "请先做一个小规模可行性验证", task_id="q1")
            updated = service._control_checkpoint(context, task, 1, "iteration_start")
            self.assertIn("小规模可行性验证", updated.instructions[-1])
            self.assertEqual(updated.metadata["human_control_cursor"], 1)

            store.request_interrupt("run-1", "不要继续正式求解", task_id="q1")
            with self.assertRaises(HumanInterruptRequested):
                service._control_checkpoint(updated, task, 1, "code_execute_start")

    def test_sandbox_stops_when_operator_interrupts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = HumanControlStore(root)
            sandbox = LocalSandboxClient(root, allow_host_processes=True)
            holder: list = []

            def run() -> None:
                holder.append(sandbox.run(
                    (sys.executable, "-I", "-c", "import time; time.sleep(30)"),
                    timeout_seconds=60,
                    cancel_check=lambda: store.is_interrupted("run-2"),
                ))

            worker = threading.Thread(target=run)
            started = time.monotonic()
            worker.start()
            time.sleep(0.25)
            store.request_interrupt("run-2", "operator test stop")
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(holder), 1)
            self.assertTrue(holder[0].cancelled)
            self.assertLess(time.monotonic() - started, 5)

    def test_solve_returns_blocked_report_after_interrupt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = HumanControlStore(Path(temp))
            store.request_interrupt("run-4", "operator stopped the solve")
            service = SolveProblemService(object(), object(), human_control=store)
            report = service.solve(
                SolveProblemTask(task_id="q1", title="Q1", problem="P"),
                SolveProblemContext(metadata={"run_id": "run-4"}),
            )
            self.assertEqual(report.status.value, "blocked")
            self.assertIn("operator stopped the solve", report.error or "")
            self.assertEqual(store.read_status("run-4")["status"], "blocked")

    def test_toy_ui_reads_status_and_accepts_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = HumanControlStore(root)
            store.publish_status("run-3", task_id="q1", event="solve_start", actor="Main Harness", status="started")
            handler = type("TestToyHandler", (_Handler,), {"store": store})
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/runs", timeout=3) as response:
                    runs = json.loads(response.read().decode("utf-8"))
                self.assertEqual(runs[0]["run_id"], "run-3")

                request = Request(
                    base + "/api/runs/run-3/suggest",
                    data=json.dumps({"message": "请显示当前迭代的验证指标"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urlopen(request, timeout=3) as response:
                    self.assertEqual(response.status, 202)
                self.assertEqual(store.pending("run-3")[0].message, "请显示当前迭代的验证指标")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
