"""chunking — Markdown 感知分块。

策略（v1，够用的工程版）：
1. 按标题(#~######)切"章节"，标题作为元数据保存；
2. 章节内按空行切段落，贪心合并到 max_chars，保证语义单元不被腰斩；
3. 向量文本前置【来源｜标题】——让 embedding 带上定位信息，
   命中后引用展示也直接用这两项。
"""

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    source: str
    heading: str
    text: str
    index: int


def chunk_markdown(content: str, source: str, max_chars: int = 600) -> list[Chunk]:
    sections: list[tuple[str, str]] = []
    heading = ""
    lines: list[str] = []
    for line in content.splitlines():
        m = HEADING_RE.match(line)
        if m:
            if "".join(lines).strip():
                sections.append((heading, "\n".join(lines)))
            heading = m.group(2).strip()
            lines = []
        else:
            lines.append(line)
    if "".join(lines).strip():
        sections.append((heading, "\n".join(lines)))

    chunks: list[Chunk] = []
    for sec_heading, body in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        current = ""
        for para in paragraphs:
            if current and len(current) + len(para) + 2 > max_chars:
                _emit(chunks, source, sec_heading, current)
                current = para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            _emit(chunks, source, sec_heading, current)
    return chunks


def _emit(chunks: list[Chunk], source: str, heading: str, body: str) -> None:
    h = heading or "（开头）"
    chunks.append(Chunk(
        source=source,
        heading=h,
        text=f"【{source}｜{h}】\n{body}",
        index=len(chunks),
    ))
