import tempfile

from agentforge.rag.chunking import chunk_markdown
from agentforge.rag.store import KnowledgeStore

SAMPLE = """# 调度模块笔记

简介段落，描述整体链路。

## 状态流转

状态一说明。这里是一段比较长的正文，用来测试段落合并行为是否合理，超过设定长度时应该拆分成新的片段继续携带同样的标题元数据信息用于向量检索定位。

状态二说明。

## 表单持久化

表单写入 Redis，前端轮询消费。
"""


def test_chunk_groups_by_heading():
    chunks = chunk_markdown(SAMPLE, "doc.md", max_chars=120)
    assert chunks
    headings = {c.heading for c in chunks}
    assert "状态流转" in headings and "表单持久化" in headings
    for c in chunks:
        assert c.text.startswith(f"【doc.md｜{c.heading}】")
        assert len(c.text) <= 120 + 60


def test_chunk_indices_unique():
    chunks = chunk_markdown(SAMPLE, "doc.md")
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = KnowledgeStore(tmp)
        try:
            store.rebuild(dim=4)
            assert store.count() == 0
            store.add([
                ([1.0, 0.0, 0.0, 0.0], {"source": "a.md", "heading": "H1", "text": "甲", "index": 0}),
                ([0.0, 1.0, 0.0, 0.0], {"source": "b.md", "heading": "H2", "text": "乙", "index": 0}),
            ])
            assert store.count() == 2
            hits = store.search([0.9, 0.1, 0.0, 0.0], top_k=1)
            assert len(hits) == 1
            assert hits[0][1]["source"] == "a.md"
            assert hits[0][0] > 0.5
        finally:
            store.close()


def test_store_rebuild_wipes():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = KnowledgeStore(tmp)
        try:
            store.rebuild(dim=4)
            store.add([([1.0, 0, 0, 0], {"source": "a", "heading": "h", "text": "t", "index": 0})])
            store.rebuild(dim=4)
            assert store.count() == 0
        finally:
            store.close()
