# 标书智能助手

基于 **LangGraph + RAG** 的标书生成与完善系统。

## 功能

- **文档管理**：上传标书文档（PDF/Word/TXT）到 ChromaDB 向量知识库
- **标书生成**：输入草稿或需求，AI 自动分析、补全、完善标书
- **标书分析**：对已有标书进行质量评估和改进建议
- **知识库管理**：查看、搜索、删除已入库文档

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env` 并填写 API Key，或直接在应用的系统设置页面配置：

```bash
cp .env.example .env
```

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`

## 架构

```
标书智能助手/
├── app.py                  # Streamlit 主页
├── pages/
│   ├── 1_文档管理.py       # 文档上传与知识库管理
│   ├── 2_标书生成.py       # 标书生成与完善
│   └── 3_系统设置.py       # API 和 RAG 参数配置
├── config/
│   └── settings.py         # 配置管理（支持持久化）
├── rag/
│   ├── loader.py           # 文档加载与分块
│   ├── embeddings.py       # Embedding 模型
│   ├── vector_store.py     # ChromaDB 操作
│   └── retriever.py        # 检索逻辑
├── graphs/
│   ├── state.py            # LangGraph 状态定义
│   ├── nodes.py            # 各处理节点
│   └── bidding_graph.py    # 图编排
└── utils/
    └── helpers.py          # 工具函数
```

## LangGraph 工作流

```
START → 文档分析 → RAG检索 → [完善模式] 章节完善 → 生成缺失章节 → 整合输出 → END
                           → [分析模式] 质量分析报告 → END
```

## 支持的 AI 提供商

| 提供商 | 模型示例 |
|--------|---------|
| OpenAI | gpt-4o, gpt-4o-mini |
| Anthropic | claude-opus-4-8, claude-sonnet-4-6 |
| 兼容 OpenAI 格式的其他服务 | 配置 Base URL 即可 |

## 支持的文档格式

- PDF (`.pdf`)
- Word (`.docx`, `.doc`)
- 纯文本 (`.txt`, `.md`)
- 直接粘贴文本
