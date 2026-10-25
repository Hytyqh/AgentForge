"""tools/registry.py — 装饰器注册工具，自动生成 function-calling Schema。

"任务即工具"思想的最小实现：生产级系统用 AST 解析任务函数与结构化注释
生成 Schema；这里用 inspect 解析函数签名 + 类型注解 + Google 风格 Args 文档，
达成同样效果。

一份注册表，三处消费：模型工具声明、MCP list_tools、本地执行——以后都复用这个模式。
"""

import inspect
import json
import re
from dataclasses import dataclass
from typing import Callable, get_type_hints

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_ARGS_RE = re.compile(r"^(args|arguments|参数)\s*[:：]?\s*$", re.IGNORECASE)
_STOP_RE = re.compile(
    r"^(returns?|return|yields?|examples?|raises?|usage|用法|示例|注意)\s*[:：]?\s*$",
    re.IGNORECASE,
)
_PARAM_RE = re.compile(r"^([A-Za-z_]\w*)\s*(\([^)]*\))?\s*[:：]\s*(.*)$")


@dataclass
class ToolDef:
    name: str
    description: str
    func: Callable
    parameters: dict

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def tool(self, func: Callable | None = None, *, name: str | None = None, description: str | None = None):
        def register(f: Callable) -> Callable:
            tool_name = name or f.__name__
            summary, param_docs = _parse_docstring(inspect.getdoc(f) or "")
            self._tools[tool_name] = ToolDef(
                name=tool_name,
                description=description or summary or tool_name,
                func=f,
                parameters=_build_parameters(f, param_docs),
            )
            return f

        if func is not None:
            return register(func)
        return register

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, name: str, arguments: str) -> str:
        """执行工具并把结果规整成字符串。

        关键设计：所有错误都以文本形式返回给模型（而不是抛异常），
        让模型把错误当观察结果来自救——生产级 ReAct 实现的通行容错做法。
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"[工具不存在] {name}，可用工具: {self.names()}"
        try:
            kwargs = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            return f"[参数解析失败] 不是合法 JSON: {arguments!r} ({e})"
        if not isinstance(kwargs, dict):
            return f"[参数格式错误] 期望 JSON 对象，收到: {type(kwargs).__name__}"
        try:
            result = tool.func(**kwargs)
        except TypeError as e:
            return f"[参数不匹配] {e}"
        except Exception as e:
            return f"[工具执行失败] {type(e).__name__}: {e}"
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        return _truncate(text)


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    summary: list[str] = []
    param_docs: dict[str, str] = {}
    mode = "summary"
    for raw in doc.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _ARGS_RE.match(line):
            mode = "args"
            continue
        if _STOP_RE.match(line):
            mode = "stop"
            continue
        if mode == "args":
            m = _PARAM_RE.match(line)
            if m:
                param_docs[m.group(1)] = m.group(3).strip()
            elif param_docs:
                last = next(reversed(param_docs))
                param_docs[last] += " " + line
        elif mode == "summary":
            summary.append(line)
    return " ".join(summary).strip(), param_docs


def _build_parameters(f: Callable, param_docs: dict[str, str]) -> dict:
    hints = get_type_hints(f)
    props: dict[str, dict] = {}
    required: list[str] = []
    for pname, p in inspect.signature(f).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        prop = {"type": _TYPE_MAP.get(hints.get(pname, str), "string")}
        if pname in param_docs:
            prop["description"] = param_docs[pname]
        props[pname] = prop
        if p.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": props, "required": required}


def _truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[已截断，共 {len(text)} 字符]"
