"""探针 1：模型到底是怎么被调用的——打印真实请求体和响应体。

跑法: uv run python probes/01_raw_chat.py
你看完会明白：所谓"调用模型"= 一个 HTTP POST，模型的"记忆"全在 messages 数组里，
服务端是无状态的。
"""
import json

from agentforge.config import Settings
from agentforge.llm import LLM

settings = Settings.from_env()
llm = LLM(settings)

messages = [{"role": "user", "content": "北京今天适合跑步吗？需要信息就查工具"}]
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    },
}]

print("=" * 70)
print("【发出去的请求体】POST", settings.base_url, "/chat/completions")
print("=" * 70)
print(json.dumps(
    {"model": settings.chat_model, "messages": messages, "tools": tools, "stream": False},
    ensure_ascii=False, indent=2,
))

resp = llm.client.chat.completions.create(
    model=settings.chat_model, messages=messages, tools=tools,
)
print()
print("=" * 70)
print("【收回来的响应体】(choices[0].message 的原始 JSON)")
print("=" * 70)
print(resp.choices[0].message.model_dump_json(indent=2, exclude_none=True))
print()
print("结论：模型没有执行任何东西。它只是'同意在回复里放一个调用意图 JSON'，")
print("真正跑函数的是我们的代码——这就是 function calling 的全部真相。")
