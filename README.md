# AgentForge

> 自研 AI Agent 平台：手写 ReAct 执行循环、工具自动注册（函数签名→JSON Schema）、
> 混合检索 RAG、分层记忆、任务状态机与断点恢复、HITL 人工介入、MCP 协议开放。

一个"五脏俱全"的 Agent 平台：核心机制全部从零手写，不依赖 LangChain/LangGraph；
存储与向量组件用成熟方案（Qdrant / DashScope），把精力花在机制而不是轮子上。

## 快速开始

```powershell
# 1. 配置密钥：复制 .env.example 为 .env，填入 AGENTFORGE_API_KEY 与 AGENTFORGE_EMBED_API_KEY
# 2. 安装依赖（uv 或 pip 均可，Python >= 3.12）
uv sync

# 3. 构建个人知识库（分块→embedding→Qdrant）
#    语料范围用 AGENTFORGE_CORPUS_PATTERNS 指定（逗号分隔 glob），默认工作区根目录 *.md
uv run python -m agentforge --ingest

# 4. 一次性提问
uv run python -m agentforge --ask "现在几点？我的知识库里 HITL 是怎么设计的？"

# 5. 交互 REPL
uv run python -m agentforge
```

## 架构总览（当前 + 规划）

```
                 ┌─────────────────────────────┐
   CLI / API ──▶ │  Agent 循环 (agent/loop.py)  │ Think → Act → Observe
                 │  · 轮次/预算护栏 · 强制收敛   │
                 └──────┬──────────────┬───────┘
                        │              │
              ┌─────────▼───┐   ┌──────▼────────┐
              │ ToolRegistry │   │  LLM 客户端    │ 流式 + tool_calls 增量拼接
              │ 签名→Schema  │   │ (OpenAI 兼容)  │
              └─────────┬───┘   └───────────────┘
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   builtin 工具      rag 检索        memory 读写
  (时间/计算/文件)   (改写→混合召回    (短期历史/长期事实)
                     →重排→引用)          [规划]
        │
        ▼
  [规划] 任务状态机 + 队列调度 + HITL 中断恢复 + MCP Server
```

## 路线图

| 台阶 | 内容 | 状态 |
|---|---|---|
| 1 | ReAct 循环 + 工具注册 + 流式客户端 + 测试 | ✅ 完成 |
| 2 | RAG：解析→分块→向量化→检索→引用回传（agentic 检索已涌现） | ✅ 完成 |
| 2.5 | RAG 加强：BM25 混合 + RRF + qwen3-rerank + 指标评测 | ⏳ 下一步 |
| 3 | 记忆：对话历史持久化 + 事实抽取/去重 | |
| 4 | 规划：意图识别→任务拆解→队列执行 | |
| 5 | 调度：周期任务闭环（执行→评估→再调度） | |
| 6 | HITL：中断→持久化→人工确认→恢复 | |
| 7 | MCP：把平台能力暴露成标准 MCP 工具 | |
| 8 | 服务化：FastAPI + SSE 流式接口，替换 CLI | |

## 核心机制（阶段 2 · RAG）

- **两段式流水线**：建库（收集→Markdown 分块→embedding→Qdrant）与查询（embedding→ANN 检索→带出处引用），语料就是本地文件，零上传。
- **Markdown 感知分块**：按标题切章节、段落贪心合并到 600 字符，向量文本前置【来源｜标题】——检索命中即自带引用定位。
- **Agentic RAG**：检索作为工具（`rag_search`）暴露给 ReAct 循环，模型会自主多次检索、改写查询词、逐步逼近答案——多查询改写不是硬编码管线，是循环涌现的行为（实测 6 次递进检索）。
- **组件选择**：Qdrant 本地嵌入式（零部署、接口与服务端一致，服务化时无缝切换）+ DashScope `text-embedding-v4` 1024 维。

## 核心机制（阶段 1）

- **工具即函数**：`@registry.tool` 装饰器用 `inspect` 解析函数签名、类型注解、
  Google 风格 docstring，自动生成 function-calling JSON Schema。一份注册表多处消费。
- **流式 tool_calls 拼接**：流式响应里工具调用是碎片（按 index 分槽），
  客户端逐 chunk 把 `id/name/arguments` 拼回完整调用——手写 Agent 的第一道坎。
- **四个护栏**（踩坑经验，非玩具特性）：
  1. 轮次预算 `max_rounds`，耗尽后撤工具强制模型收敛作答；
  2. 工具结果 4000 字符截断，防止上下文爆炸；
  3. 工具错误以文本回灌为观察，模型可自救（实测已验证）；
  4. 文件工具路径守卫，限制在工作区内；计算器走 AST 白名单，杜绝代码注入。

## 目录结构

```
src/agentforge/
├── config.py            # 环境配置（.env）
├── llm.py               # OpenAI 兼容客户端：非流式 + 流式 tool_calls 拼接
├── agent/loop.py        # ReAct 循环 + 护栏（全项目心脏，65 行）
├── tools/
│   ├── registry.py      # 装饰器→Schema 自动生成，错误文本化
│   └── builtin.py       # 时间/计算/列目录/读文件
├── rag/
│   ├── chunking.py      # Markdown 感知分块
│   ├── embeddings.py    # text-embedding-v4 封装（批量/重试/归一化）
│   ├── store.py         # Qdrant 嵌入式向量库
│   ├── ingest.py        # 建库流水线
│   └── tools.py         # rag_search 工具注册
├── cli.py               # REPL + 一次性 --ask + --ingest
tests/                   # pytest：Schema 生成、执行容错、注入防护、分块/存储
probes/                  # 教学探针：HTTP 抓包 / 流式碎片 / 余弦实算 / 向量存储解剖
```

## 运行环境

- Python ≥ 3.12（开发用 3.15）
- 模型：阿里云百炼 Token Plan（OpenAI 兼容接口，qwen3.8-flash，支持 function calling）
- Embedding：百炼 `text-embedding-v4`（阶段 2）
