"""tools/builtin.py — 第一批本地工具。

工程点：路径守卫（限制在工作区内）、结果截断、纯算术表达式白名单。
工作区在第一次使用时才从环境读取，保证测试可离线导入本模块。
"""

import ast
import fnmatch
import os
from datetime import datetime

from .registry import ToolRegistry

registry = ToolRegistry()

_root_cache: str | None = None


def _workspace_root() -> str:
    global _root_cache
    if _root_cache is None:
        from ..config import Settings

        _root_cache = os.path.realpath(Settings.from_env().workspace_root)
    return _root_cache


def _safe_path(user_path: str) -> str:
    root = _workspace_root()
    if os.path.isabs(user_path):
        target = os.path.realpath(user_path)
    else:
        target = os.path.realpath(os.path.join(root, user_path))
    if not target.startswith(root):
        raise PermissionError(f"越界访问: {user_path} 不在工作区 {root} 内")
    return target


@registry.tool
def get_now() -> str:
    """获取当前日期、时间和星期。需要知道现在是什么时候时调用。"""
    now = datetime.now()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {weekdays[now.weekday()]}"


@registry.tool
def calculate(expression: str) -> str:
    """计算一个算术表达式并返回结果。

    Args:
        expression: 只含数字与 + - * / // % ** 括号的表达式，例如 "(3+5)*2**3"
    """
    node = ast.parse(expression, mode="eval")
    _check_arith(node)
    return str(eval(compile(node, "<expr>", "eval")))


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.FloorDiv,
)


def _check_arith(node: ast.AST):
    if not isinstance(node, _ALLOWED_NODES):
        raise ValueError(f"表达式包含不允许的节点: {type(node).__name__}")
    for child in ast.iter_child_nodes(node):
        _check_arith(child)


@registry.tool
def list_files(directory: str, pattern: str = "*") -> str:
    """列出工作区内某目录下的文件与子目录。

    Args:
        directory: 相对工作区根目录的路径，根目录用 "."
        pattern: 文件名通配模式，例如 "*.md"
    """
    target = _safe_path(directory)
    if not os.path.isdir(target):
        raise NotADirectoryError(f"目录不存在: {directory}")
    lines = []
    for entry in sorted(os.scandir(target), key=lambda e: e.name):
        if entry.is_dir():
            lines.append(f"[DIR]  {entry.name}")
        elif fnmatch.fnmatch(entry.name, pattern):
            lines.append(f"[FILE] {entry.name} ({entry.stat().st_size:,} bytes)")
    return "\n".join(lines) if lines else "（没有匹配的文件）"


@registry.tool
def read_text_file(file_path: str, max_chars: int = 4000) -> str:
    """读取工作区内一个文本文件的内容。

    Args:
        file_path: 相对工作区根目录的文件路径
        max_chars: 最多返回的字符数
    """
    target = _safe_path(file_path)
    if not os.path.isfile(target):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(target, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read(max_chars)
