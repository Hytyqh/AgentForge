"""ingest — 建库流程：收集文件 → 分块 → 向量化 → 入库。

语料范围由环境变量 AGENTFORGE_CORPUS_PATTERNS 指定（逗号分隔的 glob 列表），
默认只收工作区根目录的 *.md。只处理 Markdown（docx/pdf 解析为后续加强项）。
"""

import os
import time
from pathlib import Path

from ..config import Settings
from .chunking import chunk_markdown
from .embeddings import Embedder
from .store import KnowledgeStore

EXCLUDE_DIRS = {".git", ".venv", "__pycache__", "node_modules", "logs", "tmp"}
MAX_FILE_BYTES = 2_000_000


def default_patterns() -> list[str]:
    raw = os.environ.get("AGENTFORGE_CORPUS_PATTERNS", "*.md")
    return [p.strip() for p in raw.split(",") if p.strip()]


def gather_files(root: Path, patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for p in sorted(root.glob(pattern)):
            if not p.is_file() or p.name.startswith("."):
                continue
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            if p not in seen:
                seen.add(p)
                found.append(p)
    return found


def ingest(settings: Settings, kb_path: str, patterns: list[str] | None = None, verbose=True) -> dict:
    patterns = patterns or default_patterns()
    root = Path(settings.workspace_root)
    files = gather_files(root, patterns)
    if not files:
        return {"files": 0, "chunks": 0, "seconds": 0.0}

    store = KnowledgeStore(kb_path)
    embedder = Embedder(settings)
    try:
        store.rebuild(dim=embedder.DIM)

        all_chunks = []
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            source = str(f.relative_to(root))
            all_chunks.extend(chunk_markdown(content, source))

        started = time.time()
        batch: list = []
        for i, chunk in enumerate(all_chunks):
            batch.append(chunk)
            if len(batch) >= 20 or i == len(all_chunks) - 1:
                vectors = embedder.embed([c.text for c in batch])
                store.add(list(zip(vectors, [
                    {"source": c.source, "heading": c.heading, "text": c.text, "index": c.index}
                    for c in batch
                ], strict=True)))
                if verbose:
                    print(f"  向量化 {min(i + 1, len(all_chunks))}/{len(all_chunks)} 片段", flush=True)
                batch = []

        stats = {"files": len(files), "chunks": store.count(), "seconds": round(time.time() - started, 1)}
        return stats
    finally:
        store.close()
