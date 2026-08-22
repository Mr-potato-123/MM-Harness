"""Small, composable prompts for the Code Agent boundary.

The static role prompt contains policy only.  The per-call task contract
contains the accepted model and paths.  Keeping those layers separate makes a
prompt auditable and prevents a domain-specific problem rule from becoming a
global Code Agent instruction.
"""

from __future__ import annotations

import json
from typing import Any


CODE_AGENT_SYSTEM_PROMPT = """你是 M2Harness 的 Code Agent。

你的唯一职责是把当前轮已经锁定的建模契约实现成一个可重复执行的 Python 源文件，并把真实执行证据交回 Harness。你不负责重新建模、选择另一个算法、审查结论、处理其他题目或发布最终论文。

事实优先级：最新的 Model→Code 返修要求 > 首轮 Model→Code 契约 > 当前任务契约 > 其他上下文。若实现需要偏离已锁定方案，必须停止并在交接中明确说明，不能悄悄换成启发式、简化模型或搜索替代品。

执行协议：读取本轮契约，写入指定源文件；先做语法/最小烟雾检查，再做一次完整执行；每次 python_execute 都必须给出 1 到 86400 秒的有限 timeout_seconds。执行失败时先根据新错误修改源文件，禁止对同一个工具和同一个参数重复调用；同一类失败无法通过可观察修改解决时，停止并返回失败证据。

输出协议：源文件只能写在指定路径，运行产生的文件和图片只能写在该轮 outputs 目录。脚本 stdout 只输出简短、可读的中文 Markdown（结论、关键数值、限制、文件/图片位置）；详细机器字段可写入 outputs/result.json。最终只返回一个内部交接对象，不能把源代码塞进最终消息。

不要把“工具调用成功”当成“模型验证通过”。只有脚本真实运行并留下可核验结果，才能声称对应验证完成。"""


def build_code_agent_task_prompt(
    task: Any,
    context: Any,
    modeling: Any,
    *,
    iteration: int,
    target_relative: str,
    output_relative: str | None = None,
    session_managed: bool = False,
) -> str:
    """Build one compact task envelope; no policy is repeated here."""

    metadata = context.metadata if isinstance(getattr(context, "metadata", None), dict) else {}
    exchange_paths = [
        item.relative_path
        for item in getattr(context, "readonly_files", ())
        if "/exchanges/" in item.relative_path.replace("\\", "/")
    ]
    repair_paths = [path for path in exchange_paths if path.endswith("model-to-code-revision.md")]
    initial_paths = [path for path in exchange_paths if path.endswith("model-to-code.md")]
    latest_code_reports = [path for path in exchange_paths if path.endswith("code-to-model.md")]
    if session_managed and iteration > 1:
        locked_model: dict[str, Any] = {
            "continuation": "The session already contains the initial locked modeling contract; do not restate or redesign it.",
            "initial_contract": initial_paths[-1] if initial_paths else None,
            "required_validations": list(modeling.required_validations),
            "expected_outputs": list(modeling.expected_outputs),
            "expected_figures": list(getattr(modeling, "expected_figures", ())),
        }
    else:
        locked_model = {
            "selected_branch_ids": list(modeling.selected_branch_ids),
            "main_scheme": modeling.main_scheme,
            "required_validations": list(modeling.required_validations),
            "expected_outputs": list(modeling.expected_outputs),
            "expected_figures": list(getattr(modeling, "expected_figures", ())),
            "coding_instructions": list(modeling.coding_instructions),
        }
    contract: dict[str, Any] = {
        "run": {
            "run_id": str(metadata.get("run_id", "unscoped")),
            "run_name": str(metadata.get("run_name", "unscoped")),
            "task_attempt": int(metadata.get("task_attempt", 1)),
        },
        "task": {
            "task_id": task.task_id,
            "title": task.title,
            "requested_outputs": list(task.requested_outputs),
        },
        "iteration": iteration,
        "phase": "initial implementation" if iteration == 1 else "repair existing implementation",
        "source_of_truth_paths": {
            "latest_repair": repair_paths[-1] if repair_paths else None,
            "initial_contract": initial_paths[-1] if initial_paths else None,
            "latest_code_report": latest_code_reports[-1] if latest_code_reports else None,
        },
        "locked_model": locked_model,
        "context_owner": "persistent agent session; this envelope is a delta, not a transcript replay" if session_managed else "caller envelope",
        "revision_request": list(getattr(context, "instructions", ())[-12:]),
        "paths": {
            "target_source": target_relative,
            "output_directory": output_relative or target_relative.rsplit("/", 1)[0] + "/outputs",
        },
        "handoff": {
            "logical_name": target_relative.rsplit("/", 1)[-1],
            "timeout_seconds": "required integer in [1, 86400]",
            "expected_validations": list(modeling.required_validations),
        },
    }
    return (
        "以下是本次调用唯一的任务契约。只执行当前 task 和当前 iteration；不要把它扩展成整个项目的任务。\n\n"
        + json.dumps(contract, ensure_ascii=False, indent=2)
    )
