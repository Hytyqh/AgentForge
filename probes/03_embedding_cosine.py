"""探针 3：embedding 到底是什么东西——向量、余弦、检索的数学本质。

跑法: uv run python probes/03_embedding_cosine.py
你会亲眼看到：一段中文变成一个 1024 个数字的列表；语义近的向量夹角小。
"""
import numpy as np

from agentforge.config import Settings
from agentforge.rag.embeddings import Embedder

settings = Settings.from_env()
embedder = Embedder(settings)

texts = [
    "周期性任务按调度自动执行并评估结果",   # 和查询主题相关
    "用户确认表单后才写入数据库",           # 无关（表单）
    "长期任务每轮执行完成后由大模型评估是否达标",  # 高度相关
    "今天食堂的红烧肉很好吃",               # 完全无关
]
vectors = np.array(embedder.embed(texts))

print(f"模型: {settings.embed_model}, 输出维度: {vectors.shape[1]}")
print(f"第一条向量的前 8 维: {np.round(vectors[0][:8], 4)}")
print(f"每条向量的模长(是否归一化): {np.round(np.linalg.norm(vectors, axis=1), 4)}")

query = embedder.embed_one("长期任务是怎么调度和评估的")  # 1024 维
q = np.array(query)

print(f"\n{'文本':<24} 与问题的余弦相似度")
sims = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
for t, s in zip(texts, sims):
    print(f"{t[:20]:<24} {s:.4f}")

order = np.argsort(-sims)
print(f"\nTop-1 命中: {texts[order[0]][:20]}  ← 检索就是按这个分数排序取前 K 条")
print("\n结论：向量检索没有任何魔法 —— 算余弦 + 排序 + 截断。")
print("Qdrant 存在的意义只是让你不用把 519 条(或 5 亿条)向量读进内存来算。")
