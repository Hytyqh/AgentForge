"""探针 2：流式输出到底在流什么——打印原始 chunk 序列。

跑法: uv run python probes/02_stream_deltas.py
重点观察：每个 chunk 只有几十字节的 delta；tool_calls 是被切碎分批送来的，
arguments 一次只给一小段 JSON 文本，拼完才是一个合法 JSON 对象。
"""
import json

from agentforge.config import Settings
from agentforge.llm import LLM

settings = Settings.from_env()
llm = LLM(settings)

messages = [{"role": "user", "content": "上海现在天气怎么样？查工具"}]
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询城市天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]

stream = llm.client.chat.completions.create(
    model=settings.chat_model, messages=messages, tools=tools, stream=True,
)

interesting = 0
for i, chunk in enumerate(stream):
    d = chunk.model_dump(exclude_none=True)
    choices = d.get("choices") or []
    if not choices:  # 最后的 usage 块，没有 delta
        print(f"chunk#{i:03d} (收尾块) = {json.dumps(d, ensure_ascii=False)[:120]}")
        continue
    delta = choices[0].get("delta", {})
    shown = json.dumps(delta, ensure_ascii=False)
    if delta and (len(shown) > 2 or "content" in delta):
        interesting += 1
        print(f"chunk#{i:03d} delta = {shown[:160]}")
    if interesting > 40:
        break

print("\n结论：流式 = 服务器把一次回复切成几百个小 JSON（delta）连着发。")
print("普通文本的 delta 拼起来是回答；tool_calls 的 delta 按 index 分槽拼起来是调用参数。")
print("llm.py 的 stream_chat() 干的就是这个拼接活。")
