"""cli.py — 命令行入口：`agentforge` 或 `python -m agentforge`。"""

import argparse
import os
import sys

from .agent.loop import run_task
from .config import Settings
from .llm import LLM
from .rag.ingest import ingest
from .rag.tools import close_rag, register_rag_tools
from .tools import registry


def _print_event(kind: str, payload: str) -> None:
    head = {"tool_call": "\n  [调用]", "tool_result": "  [结果]", "budget": "  [提示]"}.get(kind, f"[{kind}]")
    text = payload if len(payload) <= 300 else payload[:300] + f" ...(+{len(payload) - 300}字)"
    print(f"{head} {text}", flush=True)


def main(argv=None) -> int:
    try:
        return _run(argv)
    finally:
        close_rag()


def _run(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentforge", description="AgentForge — 自研 Agent 平台 CLI")
    parser.add_argument("--ask", help="一次性提问后退出")
    parser.add_argument("--ingest", action="store_true", help="构建/刷新个人知识库")
    parser.add_argument("--rounds", type=int, default=8, help="单次任务的工具轮次预算")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    kb_path = os.environ.get("AGENTFORGE_KB_PATH", "data/kb")

    if args.ingest:
        stats = ingest(settings, kb_path)
        print(f"知识库构建完成: {stats['files']} 个文件 → {stats['chunks']} 个片段, 向量化 {stats['seconds']}s")
        return 0

    register_rag_tools(registry, settings, kb_path)
    llm = LLM(settings)
    history: list[dict] = []

    if args.ask:
        run_task(llm, registry, args.ask, max_rounds=args.rounds, on_token=_emit, on_event=_print_event)
        print()
        return 0

    print("AgentForge REPL — /exit 退出  /clear 清空上下文  /tools 查看工具")
    while True:
        try:
            user = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not user:
            continue
        if user == "/exit":
            return 0
        if user == "/clear":
            history.clear()
            print("(上下文已清空)")
            continue
        if user == "/tools":
            for s in registry.schemas():
                fn = s["function"]
                print(f"  - {fn['name']}: {fn['description']}")
            continue

        first_token = {"done": False}

        def on_token(tok: str, state=first_token):
            if not state["done"]:
                print("\n小铸> ", end="", flush=True)
                state["done"] = True
            sys.stdout.write(tok)
            sys.stdout.flush()

        new_msgs = run_task(
            llm, registry, user, history=history,
            max_rounds=args.rounds, on_token=on_token, on_event=_print_event,
        )
        history.append({"role": "user", "content": user})
        finals = [m for m in new_msgs if m["role"] == "assistant" and m.get("content")]
        history.append({"role": "assistant", "content": finals[-1]["content"] if finals else ""})
        print()
    return 0


def _emit(tok: str) -> None:
    sys.stdout.write(tok)
    sys.stdout.flush()
