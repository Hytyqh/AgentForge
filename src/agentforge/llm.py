"""llm.py — 对 OpenAI 兼容接口的一层薄封装。

两个职责：
1. 非流式 chat()：一次请求拿到完整回复（脚本测试用）。
2. 流式 stream_chat()：逐 token 回调打印，并把流里碎片化的
   tool_calls 按 index 分槽拼回完整调用——手写 Agent 的第一道坎。
"""

from dataclasses import dataclass, field

from openai import OpenAI

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResult:
        resp = self.client.chat.completions.create(
            model=self.settings.chat_model,
            messages=messages,
            tools=tools or None,
        )
        msg = resp.choices[0].message
        return ChatResult(content=msg.content or "", tool_calls=_parse_tool_calls(msg.tool_calls))

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> ChatResult:
        stream = self.client.chat.completions.create(
            model=self.settings.chat_model,
            messages=messages,
            tools=tools or None,
            stream=True,
        )
        buf: list[str] = []
        acc: dict[int, dict] = {}
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                buf.append(delta.content)
                if on_token:
                    on_token(delta.content)
            for tc in delta.tool_calls or []:
                slot = acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        calls = [ToolCall(id=v["id"], name=v["name"], arguments=v["arguments"]) for _, v in sorted(acc.items())]
        return ChatResult(content="".join(buf), tool_calls=calls)


def _parse_tool_calls(raw) -> list[ToolCall]:
    out = []
    for tc in raw or []:
        out.append(ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments))
    return out
