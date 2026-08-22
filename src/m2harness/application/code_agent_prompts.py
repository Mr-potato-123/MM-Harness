"""Small, composable prompts for the Code Agent boundary.

The static role prompt contains policy only.  The per-call task contract
contains the accepted model and paths.  Keeping those layers separate makes a
prompt auditable and prevents a domain-specific problem rule from becoming a
global Code Agent instruction.
"""

from __future__ import annotations

from typing import Any


CODE_AGENT_SYSTEM_PROMPT = """你是 M2Harness 的 Code Agent。

你的唯一职责是把当前轮已经锁定的建模契约实现成一个可重复执行的 Python 源文件，并把真实执行证据交回 Harness。你不负责重新建模、选择另一个算法、审查结论、处理其他题目或发布最终论文。

事实优先级：最新的 Model→Code 返修要求 > 首轮 Model→Code 契约 > 当前任务契约 > 其他上下文。若实现需要偏离已锁定方案，必须停止并在交接中明确说明，不能悄悄换成启发式、简化模型或搜索替代品。

执行协议：读取本轮契约，写入指定源文件；先做语法/最小烟雾检查，再做一次完整执行；每次 python_execute 都必须给出 1 到 86400 秒的 timeout_seconds。timeout_seconds 现在是审计字段，不是主机执行的硬终止时限；主机执行最多每 30 分钟返回一次软检查点，进程仍会继续运行。若返回 error_code=execution_checkpoint，必须判断是否继续：继续时调用 python_continue(job_id)，放弃时调用 python_cancel(job_id)。若返回 error_code=execution_timeout 或 output.timed_out=true，才视为硬超时失败。禁止使用相同源文件和相同 timeout_seconds 重复执行；执行失败时先根据新错误修改源文件，同一类失败无法通过可观察修改解决时，停止并返回失败证据。

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
    """Build the readable Code Agent task envelope.

    The Code Agent receives a progressive-disclosure index, not a second copy
    of the model.  The accepted details live in ``modeling_report.md`` and the
    original problem remains available as a read-only PDF.  Keeping this
    function Markdown also makes the exact provider prompt easy to audit.
    """

    metadata = context.metadata if isinstance(getattr(context, "metadata", None), dict) else {}
    available_paths = [
        item.relative_path
        for item in getattr(context, "readonly_files", ())
        if isinstance(getattr(item, "relative_path", None), str)
    ]
    exchange_paths = [path for path in available_paths if "/exchanges/" in path.replace("\\", "/")]
    repair_paths = [path for path in exchange_paths if path.endswith("model-to-code-revision.md")]
    initial_paths = [path for path in exchange_paths if path.endswith("model-to-code.md")]
    latest_code_reports = [path for path in exchange_paths if path.endswith("code-to-model.md")]
    modeling_paths = [path for path in available_paths if path.endswith("/modeling/modeling_report.md")]
    problem_paths: list[str] = []
    for item in getattr(context, "readonly_files", ()):
        path = getattr(item, "relative_path", "")
        role = getattr(getattr(item, "role", None), "value", getattr(item, "role", ""))
        if role == "problem" or path.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            problem_paths.append(path)
    output_dir = output_relative or target_relative.rsplit("/", 1)[0] + "/outputs"
    initial_reference = initial_paths[-1] if initial_paths else "首轮 model-to-code.md（按白名单路径读取）"
    modeling_reference = modeling_paths[-1] if modeling_paths else "同一题目的 modeling_report.md（按白名单路径读取）"
    phase = "首轮实现" if iteration == 1 else "Code 返修"
    lines = [
        "# Code Agent 任务信封", "",
        f"- 运行：`{metadata.get('run_name', metadata.get('run_id', 'unscoped'))}`",
        f"- 题目：`{task.task_id}` — {task.title}",
        f"- 迭代：`{iteration}`（{phase}）",
        "- 范围：只处理当前题目和当前迭代，不处理其他题目，不重新设计已接受模型。",
        "",
        "## 唯一来源（先读）", "",
        f"- Model→Code 交接：`{initial_reference if iteration == 1 else (repair_paths[-1] if repair_paths else initial_reference)}`",
        f"- 统一建模报告：`{modeling_reference}`",
    ]
    if problem_paths:
        lines.extend(["- 原始题面（只读 PDF/图片）：", *(f"  - `{path}`" for path in dict.fromkeys(problem_paths))])
    else:
        lines.append("- 原始题面：当前白名单没有 PDF/图片；不得自行假设题面参数。")
    if latest_code_reports:
        lines.append(f"- 最近一次 Code→Model 报告：`{latest_code_reports[-1]}`")
    lines.extend([
        "",
        "## 本轮动作", "",
        "1. 用 workspace 工具读取上面的来源文件；它们是内容唯一来源，不能用 prompt 中的猜测替代。",
        f"2. 首轮必须先将完整、确定性的 Python 源文件写入：`{target_relative}`；源文件写入前不得执行 python_execute 或 validation_run。",
        f"3. 输出目录：将运行产生的文件和图片只写入 `{output_dir}`",
        "4. 源文件写入后，按建模报告中的验证 ID 执行并留下可复核证据；脚本 stdout 只写简短中文结果，详细机器字段放在 outputs/result.json。",
        "5. TODO 由 Harness 预置，表示实现生命周期；只能更新既有条目的 status，不得改写、删除或增加任务。",
        "",
        "## 返回要求", "",
        "只返回一个 CodeProposal 交接对象：`logical_name`、`source_path`、`expected_validations`、`timeout_seconds`。",
        "`timeout_seconds` 是必填整数，范围为 1–86400，用于记录本次执行意图；主机执行采用 30 分钟软检查点，不会因该字段直接硬终止。没有该字段不得返回成功交接。",
        "不要在最终回复中粘贴源代码；不要把工具调用成功当作模型验证通过。",
    ])
    if session_managed:
        lines.extend(["", "## 会话边界", "", "这是持久 Code Agent 会话中的当前增量；沿用首轮建模报告，不重复建模。"])
    return "\n".join(lines).strip() + "\n"
