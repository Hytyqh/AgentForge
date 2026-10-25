"""rag — 个人知识库检索链路。

流水线：文件收集 → Markdown 分块 → text-embedding-v4 向量化 → Qdrant 存储 → 语义检索。
每个环节独立成模块，方便逐个替换加强（查询改写/混合检索/重排是后续开关）。
"""

from .chunking import chunk_markdown
from .embeddings import Embedder
from .store import KnowledgeStore

__all__ = ["chunk_markdown", "Embedder", "KnowledgeStore"]
