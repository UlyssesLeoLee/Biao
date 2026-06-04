import streamlit as st

st.set_page_config(page_title="系统设置 - 标书智能助手", page_icon="⚙️", layout="wide")
st.title("⚙️ 系统设置")
st.caption("配置 AI API 和向量数据库参数")

from config.settings import Settings, reload_settings

settings = reload_settings()

tab_ai, tab_rag, tab_about = st.tabs(["AI 配置", "RAG 配置", "关于"])

_PROVIDERS = ["openai", "anthropic", "nvidia"]
_PROVIDER_LABELS = {
    "openai": "OpenAI (GPT 系列)",
    "anthropic": "Anthropic (Claude 系列)",
    "nvidia": "NVIDIA NIM (免费额度)",
}

with tab_ai:
    st.markdown("### AI API 配置")

    current_idx = _PROVIDERS.index(settings.api_provider) if settings.api_provider in _PROVIDERS else 0
    api_provider = st.selectbox(
        "API 提供商",
        _PROVIDERS,
        index=current_idx,
        format_func=lambda x: _PROVIDER_LABELS.get(x, x),
    )

    st.markdown("---")

    # Initialize with existing values so unused fields stay unchanged
    openai_key = settings.openai_api_key
    openai_base_url = settings.openai_base_url
    openai_model = settings.openai_model
    anthropic_key = settings.anthropic_api_key
    anthropic_model = settings.anthropic_model
    nvidia_key = settings.nvidia_api_key
    nvidia_base_url_val = settings.nvidia_base_url
    nvidia_model_val = settings.nvidia_model

    if api_provider == "openai":
        st.markdown("#### OpenAI 配置")
        openai_key = st.text_input("OpenAI API Key", value=settings.openai_api_key,
                                   type="password", placeholder="sk-...")
        openai_base_url = st.text_input(
            "API Base URL（支持第三方兼容接口，留空用官方地址）",
            value=settings.openai_base_url if settings.openai_base_url != "https://api.openai.com/v1" else "",
            placeholder="https://api.openai.com/v1",
        )
        openai_model = st.text_input("模型名称", value=settings.openai_model, placeholder="gpt-4o")
        st.info("常用模型：gpt-4o · gpt-4o-mini · gpt-4-turbo · gpt-3.5-turbo")

    elif api_provider == "anthropic":
        st.markdown("#### Anthropic 配置")
        anthropic_key = st.text_input("Anthropic API Key", value=settings.anthropic_api_key,
                                      type="password", placeholder="sk-ant-...")
        anthropic_model = st.text_input("模型名称", value=settings.anthropic_model,
                                        placeholder="claude-opus-4-8")
        st.info("常用模型：claude-opus-4-8 · claude-sonnet-4-6 · claude-haiku-4-5-20251001")

    else:  # nvidia
        st.markdown("#### NVIDIA NIM 配置")
        st.markdown(
            "前往 [build.nvidia.com](https://build.nvidia.com) 免费注册并获取 `nvapi-` 开头的 API Key。"
            " 免费额度可使用多个开源大模型。"
        )
        nvidia_key = st.text_input("NVIDIA API Key", value=settings.nvidia_api_key,
                                   type="password", placeholder="nvapi-...")
        nvidia_model_val = st.text_input("模型名称", value=settings.nvidia_model,
                                         placeholder="meta/llama-3.1-70b-instruct")
        nvidia_base_url_val = st.text_input(
            "NIM Base URL（一般不需要修改）",
            value=settings.nvidia_base_url,
            placeholder="https://integrate.api.nvidia.com/v1",
        )
        st.info(
            "常用免费模型：\n"
            "- `meta/llama-3.1-70b-instruct`（推荐）\n"
            "- `meta/llama-3.1-8b-instruct`（速度更快）\n"
            "- `nvidia/llama-3.1-nemotron-70b-instruct`\n"
            "- `mistralai/mixtral-8x22b-instruct-v0.1`"
        )

    st.markdown("---")
    st.markdown("#### Embedding 配置")
    st.caption("Embedding 将文档向量化后存入知识库，与 LLM 提供商可独立配置")

    _EMB_PROVIDERS = ["openai", "huggingface", "nvidia"]
    _EMB_LABELS = {
        "openai": "OpenAI Embeddings（需要 API Key）",
        "huggingface": "HuggingFace（本地运行，无需 API）",
        "nvidia": "NVIDIA NIM Embeddings（需要 nvapi- Key）",
    }
    emb_idx = _EMB_PROVIDERS.index(settings.embedding_provider) if settings.embedding_provider in _EMB_PROVIDERS else 0
    embedding_provider = st.selectbox(
        "Embedding 提供商",
        _EMB_PROVIDERS,
        index=emb_idx,
        format_func=lambda x: _EMB_LABELS.get(x, x),
    )

    if embedding_provider == "openai":
        embedding_model = st.text_input("Embedding 模型", value=settings.embedding_model,
                                        placeholder="text-embedding-3-small")
        st.caption("推荐：text-embedding-3-small（性价比高）· text-embedding-3-large（精度更高）")
    elif embedding_provider == "nvidia":
        emb_default = settings.embedding_model if settings.embedding_provider == "nvidia" else "nvidia/nv-embedqa-e5-v5"
        embedding_model = st.text_input("Embedding 模型", value=emb_default,
                                        placeholder="nvidia/nv-embedqa-e5-v5")
        st.caption("推荐：nvidia/nv-embedqa-e5-v5 · nvidia/nv-embed-v1")
    else:
        hf_default = (settings.embedding_model if settings.embedding_provider == "huggingface"
                      else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        embedding_model = st.text_input("HuggingFace 模型名称", value=hf_default,
                                        placeholder="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        st.caption("推荐中文模型：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    if st.button("保存 AI 配置", type="primary", use_container_width=True):
        new_settings = Settings(
            api_provider=api_provider,
            openai_api_key=openai_key,
            openai_base_url=openai_base_url or "https://api.openai.com/v1",
            openai_model=openai_model or "gpt-4o",
            anthropic_api_key=anthropic_key,
            anthropic_model=anthropic_model or "claude-opus-4-8",
            nvidia_api_key=nvidia_key,
            nvidia_base_url=nvidia_base_url_val or "https://integrate.api.nvidia.com/v1",
            nvidia_model=nvidia_model_val or "meta/llama-3.1-70b-instruct",
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            chroma_persist_dir=settings.chroma_persist_dir,
            chroma_collection_name=settings.chroma_collection_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            retrieval_top_k=settings.retrieval_top_k,
        )
        new_settings.save()
        reload_settings()
        st.success("AI 配置已保存！")

    st.markdown("---")
    if st.button("测试 API 连接", use_container_width=True):
        with st.spinner("测试连接中..."):
            try:
                test_settings = reload_settings()
                from graphs.nodes import get_llm
                llm = get_llm(test_settings)
                from langchain_core.messages import HumanMessage
                resp = llm.invoke([HumanMessage(content="你好，请回复：连接成功")])
                st.success(f"API 连接成功！模型回复：{resp.content[:100]}")
            except Exception as e:
                st.error(f"连接失败：{str(e)}")


with tab_rag:
    st.markdown("### RAG 与文档处理配置")

    st.markdown("#### 向量数据库")
    chroma_dir = st.text_input("ChromaDB 持久化目录", value=settings.chroma_persist_dir,
                               placeholder="./chroma_db")
    chroma_collection = st.text_input("集合名称", value=settings.chroma_collection_name,
                                      placeholder="bidding_documents")

    st.markdown("#### 文档分块参数")
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("分块大小（字符数）", min_value=200, max_value=4000,
                                     value=settings.chunk_size, step=100,
                                     help="每个文档块的最大字符数，建议 600-1200")
    with col2:
        chunk_overlap = st.number_input("分块重叠（字符数）", min_value=0, max_value=500,
                                        value=settings.chunk_overlap, step=50,
                                        help="相邻块之间的重叠字符数，建议 100-200")

    retrieval_top_k = st.slider("检索返回数量 (Top K)", min_value=1, max_value=20,
                                value=settings.retrieval_top_k,
                                help="每次检索返回的最相关文档块数量")

    if st.button("保存 RAG 配置", type="primary", use_container_width=True):
        cur = reload_settings()
        new_settings = Settings(
            api_provider=cur.api_provider,
            openai_api_key=cur.openai_api_key,
            openai_base_url=cur.openai_base_url,
            openai_model=cur.openai_model,
            anthropic_api_key=cur.anthropic_api_key,
            anthropic_model=cur.anthropic_model,
            nvidia_api_key=cur.nvidia_api_key,
            nvidia_base_url=cur.nvidia_base_url,
            nvidia_model=cur.nvidia_model,
            embedding_provider=cur.embedding_provider,
            embedding_model=cur.embedding_model,
            chroma_persist_dir=chroma_dir,
            chroma_collection_name=chroma_collection,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            retrieval_top_k=retrieval_top_k,
        )
        new_settings.save()
        reload_settings()
        st.success("RAG 配置已保存！")


with tab_about:
    st.markdown("### 关于标书智能助手")
    st.markdown("""
**标书智能助手** 是一个基于 LangGraph + RAG 的标书生成与完善系统，采用插件化架构。

#### 技术栈
| 组件 | 技术 |
|------|------|
| AI 工作流 | LangGraph |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| 前端界面 | Streamlit |
| AI 模型 | OpenAI / Anthropic / NVIDIA NIM |
| 部署 | Docker + Kubernetes (Sealos) |

#### 插件架构
所有功能组件均为独立插件，位于 `plugins/` 目录：
- `plugins/providers/` — LLM 提供商（openai / anthropic / nvidia，可自行新增）
- `plugins/embeddings/` — Embedding 模型（openai / huggingface / nvidia）
- `plugins/loaders/` — 文档格式加载器（pdf / docx / text）
- `plugins/nodes/` — LangGraph 节点（分析 / 检索 / 完善 / 生成 / 整合）

新增插件只需：在对应目录创建 `.py` 文件，实现接口类，调用 `registry.register_*(...)` 即可自动发现。

#### 工作流
```
START → 文档分析 → RAG检索 ─┬─[完善模式]→ 章节完善 → 生成缺失 → 整合输出 → END
                             └─[分析模式]→ 质量分析报告 → END
```

#### 数据安全
- 所有文档和 API Key 均存储在本地（或 K8s PVC）
- API Key 仅用于调用 AI 接口，不传输到任何其他服务器
    """)
