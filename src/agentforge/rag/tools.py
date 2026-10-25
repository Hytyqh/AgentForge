"""tools — 把知识库检索注册成 Agent 工具。

懒初始化：Embedder/Store 在第一次真正调用时才创建，
缺 key / 没建库都以文本返回给模型，不炸主流程。
进程退出前调 close_rag() 显式释放 Qdrant 本地句柄。
"""

import gc

from ..config import Settings
from ..tools.registry import ToolRegistry
from .embeddings import Embedder
from .store import KnowledgeStore

_state: dict = {}


def register_rag_tools(registry: ToolRegistry, settings: Settings, kb_path: str) -> None:

    def _lazy():
        if "store" not in _state:
            _state["store"] = KnowledgeStore(kb_path)
            _state["embedder"] = Embedder(settings)
        return _state["store"], _state["embedder"]

    def rag_search(query: str, top_k: int = 5) -> str:
        """在个人知识库（用户提供的文档语料）中语义检索，返回原文片段及出处。

        凡问题涉及用户提及的历史项目、工作文档、技术方案时，
        必须先调用本工具拿依据，再基于检索结果作答并标注出处。

        Args:
            query: 自然语言问题，或关键概念组合
            top_k: 返回片段数量，一般 3 到 8
        """
        try:
            store, embedder = _lazy()
            if store.count() == 0:
                return "[知识库未建] 请先运行: python -m agentforge --ingest"
            vector = embedder.embed_one(query)
            hits = store.search(vector, max(1, min(int(top_k), 12)))
        except Exception as e:
            return f"[知识库检索失败] {type(e).__name__}: {e}"
        if not hits:
            return "（没有检索到相关内容）"
        blocks = []
        for i, (score, payload) in enumerate(hits, 1):
            blocks.append(
                f"[{i}] 出处: {payload.get('source', '?')}｜{payload.get('heading', '')} (相似度 {score:.3f})\n{payload.get('text', '')}"
            )
        return "\n\n".join(blocks)

    registry.tool(rag_search)


def close_rag() -> None:
    store = _state.pop("store", None)
    if store is not None:
        try:
            store.close()
        except Exception:
            pass
        gc.collect()
