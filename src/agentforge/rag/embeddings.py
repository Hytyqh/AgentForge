"""embeddings — DashScope text-embedding-v4 封装（OpenAI 兼容接口）。

工程点：批大小限制、限流退避重试、维度固定 1024。
"""

import time

from openai import OpenAI

from ..config import Settings


class Embedder:
    BATCH_SIZE = 10
    DIM = 1024

    def __init__(self, settings: Settings):
        if not settings.embed_api_key:
            raise RuntimeError("缺少 AGENTFORGE_EMBED_API_KEY（百炼通用 sk- key）")
        self.client = OpenAI(api_key=settings.embed_api_key, base_url=settings.embed_base_url)
        self.model = settings.embed_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.DIM,
                )
                return [item.embedding for item in resp.data]
            except Exception as e:  # 限流/瞬时错误退避重试
                last_error = e
                time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"embedding 连续 3 次失败: {last_error}")
