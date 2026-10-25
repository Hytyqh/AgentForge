"""探针 4：向量到底存在哪——解剖 'data/kb' 目录。

跑法: uv run python probes/04_where_vectors_live.py
澄清一个误解：项目里【有】向量数据库——Qdrant，只是用的"嵌入式模式"：
以 Python 库的形式在你进程里跑，数据就是磁盘上的 SQLite 文件，没有独立服务/容器。
"""
import sqlite3
from pathlib import Path

root = Path("data/kb")
print("【目录树】data/kb 就是整个'向量数据库'：")
for p in sorted(root.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(root.parent)}  ({p.stat().st_size:,} bytes)")
    else:
        print(f"  {p.relative_to(root.parent)}/")

db = root / "collection" / "kb" / "storage.sqlite"
if db.exists():
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    print("\n【storage.sqlite 的表】", tables)
    for t in tables:
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {n} 行")
        except sqlite3.Error:
            pass
    con.close()

print("\n结论：'本地模式' = 库在进程内 + 数据在 SQLite 文件，零部署。")
print("换成 'server 模式' 只是把 QdrantClient(path=...) 改成 QdrantClient(host=...)，")
print("向量还是那些向量，存在独立服务的 RocksDB 里——代码一行不用改。")
