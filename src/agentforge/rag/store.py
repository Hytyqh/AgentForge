"""store — Qdrant 本地嵌入式向量库。

零部署：数据落在本地目录（data/kb），无需 Docker/服务端，
但接口形态和 Qdrant Server 完全一致——阶段 8 服务化时可直接换连接串。
"""

import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

COLLECTION = "kb"


class KnowledgeStore:
    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.dir))

    def _point_id(self, source: str, index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{index}"))

    def rebuild(self, dim: int = 1024) -> None:
        if self.client.collection_exists(COLLECTION):
            self.client.delete_collection(COLLECTION)
        self.client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self._wipe_residual_points()

    def _wipe_residual_points(self) -> None:
        """qdrant 本地模式的坑：删掉再建同名集合，旧数据会从 WAL 恢复。
        用"取一页删一页"的方式显式清空。"""
        while True:
            points, _ = self.client.scroll(
                COLLECTION, limit=256, with_payload=False, with_vectors=False
            )
            if not points:
                return
            self.client.delete(
                COLLECTION, points_selector=PointIdsList(points=[p.id for p in points])
            )

    def add(self, items: list[tuple[list[float], dict]]) -> None:
        points = [
            PointStruct(id=self._point_id(p["source"], p["index"]), vector=v, payload=p)
            for v, p in items
        ]
        for i in range(0, len(points), 100):
            self.client.upsert(COLLECTION, points=points[i : i + 100])

    def search(self, vector: list[float], top_k: int = 5) -> list[tuple[float, dict]]:
        if not self.client.collection_exists(COLLECTION):
            return []
        result = self.client.query_points(COLLECTION, query=vector, limit=top_k, with_payload=True)
        return [(p.score, dict(p.payload or {})) for p in result.points]

    def count(self) -> int:
        if not self.client.collection_exists(COLLECTION):
            return 0
        return self.client.count(COLLECTION).count

    def close(self) -> None:
        self.client.close()
