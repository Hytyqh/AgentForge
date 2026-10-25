"""agent/loop.py — ReAct 执行循环：AgentForge 的心脏。

Think → Act → Observe 闭环。ReAct 模式的最小实现，
保留生产级系统不可或缺的四个护栏：轮次预算、结果截断、错误当观察回灌、预算耗尽强制收敛。
"""

from typing import Callable

from ..llm import LLM
from ..tools.registry import ToolRegistry

SYSTEM_PROMPT = """你是 AgentForge 驱动的助手。

工作原则：
1. 涉及事实（当前时间、文件内容、计算结果）必须先调用工具获取，不许凭空编造。
2. 拿到工具结果后，先用一句话概括你看到了什么，再给出结论。
3. 同一轮可以调用多个互不依赖的工具。
4. 工具返回错误时，读错误原因，修正参数最多重试一次；仍失败就如实说明。
"""


def run_task(
    llm: LLM,
    registry: ToolRegistry,
    user_input: str,
    history: list[dict] | None = None,
    max_rounds: int = 8,
    on_token: Callable[[str], None] | None = None,
    on_event: Callable[[str, str], None] | None = None,
) -> list[dict]:
    """执行一个任务。返回本轮新增消息（不含 system 与 history）。"""
    turn: list[dict] = [{"role": "user", "content": user_input}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(history or []), *turn]

    for _ in range(max_rounds):
        result = llm.stream_chat(messages, registry.schemas(), on_token=on_token)
        assistant_msg: dict = {"role": "assistant", "content": result.content or ""}
        if result.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in result.tool_calls
            ]
        messages.append(assistant_msg)
        turn.append(assistant_msg)

        if not result.tool_calls:
            return turn

        for tc in result.tool_calls:
            if on_event:
                on_event("tool_call", f"{tc.name} {tc.arguments}")
            observation = registry.execute(tc.name, tc.arguments)
            if on_event:
                on_event("tool_result", observation)
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": observation}
            messages.append(tool_msg)
            turn.append(tool_msg)

    if on_event:
        on_event("budget", "工具轮次预算耗尽，强制收敛作答")
    messages.append({"role": "user", "content": "工具轮次已用完，请基于以上信息直接给出最终回答。"})
    result = llm.stream_chat(messages, None, on_token=on_token)
    final = {"role": "assistant", "content": result.content}
    turn.append(final)
    return turn
