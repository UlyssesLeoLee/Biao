import streamlit as st

st.set_page_config(page_title="系统设置 - 标书智能助手", page_icon="⚙️", layout="wide")
st.title("⚙️ 系统设置")
st.caption("配置 AI API 和向量数据库参数")

from config.settings import Settings, reload_settings

settings = reload_settings()

tab_ai, tab_rag, tab_about = st.tabs(["AI 配置", "RAG 配置", "关于"])

with tab_ai:
    st.markdown("### AI API 配置")

    api_provider = st.selectbox(
        "API 提供商",
        ["openai", "anthropic"],
        index=0 if settings.api_provider == "openai" else 1,
        format_func=lambda x: {"openai": "OpenAI (GPT 系列)", "anthropic": "Anthropic (Claude 系列)"}[x],
    )

    st.markdown("---")

    if api_provider == "openai":
        st.markdown("#### OpenAI 配置")
        openai_key = st.text_input(
            "OpenAI API Key",
            value=settings.openai_api_key,
            type="password",
            placeholder="sk-...",
        )
        openai_base_url = st.text_input(
            "API Base URL（可选，留空使用官方地址）",
            value=settings.openai_base_url if settings.openai_base_url != "https://api.openai.com/v1" else "",
            placeholder="https://api.openai.com/v1",
        )
        openai_model = st.text_input(
            "模型名称",
            value=settings.openai_model,
            placeholder="gpt-4o",
        )
        st.info("常用模型：gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo")
        anthropic_key = settings.anthropic_api_key
        anthropic_model = settings.anthropic_model
    else:
        openai_key = settings.openai_api_key
        openai_base_url = settings.openai_base_url
        openai_model = settings.openai_model

        st.markdown("#### Anthropic 配置")
        anthropic_key = st.text_input(
            "Anthropic API Key",
            value=settings.anthropic_api_key,
            type="password",
            placeholder="sk-ant-...",
        )
        anthropic_model = st.text_input(
            "模型名称",
            value=settings.anthropic_model,
            placeholder="claude-opus-4-8",
        )
        st.info("常用模型：claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5-20251001")

    st.markdown("---")
    st.markdown("#### Embedding 配置")
    st.caption("Embedding 用于将文档转换为向量，存入知识库")

    embedding_provider = st.selectbox(
        "Embedding 提供商",
        ["openai", "huggingface"],
        index=0 if settings.embedding_provider == "openai" else 1,
        format_func=lambda x: {"openai": "OpenAI Embeddings（需要 API Key）", "huggingface": "HuggingFace（本地，无需 API）"}[x],
    )
    if embedding_provider == "openai":
        embedding_model = st.text_input(
            "Embedding 模型",
            value=settings.embedding_model,
            placeholder="text-embedding-3-small",
        )
        st.caption("推荐：text-embedding-3-small（性价比高）或 text-embedding-3-large（精度更高）")
    else:
        embedding_model = st.text_input(
            "HuggingFace 模型名称",
            value=settings.embedding_model if settings.embedding_provider == "huggingface" else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            placeholder="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        st.caption("推荐中文模型：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    if st.button("保存 AI 配置", type="primary", use_container_width=True):
        new_settings = Settings(
            api_provider=api_provider,
            openai_api_key=openai_key,
            openai_base_url=openai_base_url or "https://api.openai.com/v1",
            openai_model=openai_model or "gpt-4o",
            anthropic_api_key=anthropic_key,
            anthropic_model=anthropic_model or "claude-opus-4-8",
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
                from config.settings import reload_settings as rs
                test_settings = rs()
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
    chroma_dir = st.text_input(
        "ChromaDB 持久化目录",
        value=settings.chroma_persist_dir,
        placeholder="./chroma_db",
    )
    chroma_collection = st.text_input(
        "集合名称",
        value=settings.chroma_collection_name,
        placeholder="bidding_documents",
    )

    st.markdown("#### 文档分块参数")
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input(
            "分块大小（字符数）",
            min_value=200,
            max_value=4000,
            value=settings.chunk_size,
            step=100,
            help="每个文档块的最大字符数，建议 600-1200",
        )
    with col2:
        chunk_overlap = st.number_input(
            "分块重叠（字符数）",
            min_value=0,
            max_value=500,
            value=settings.chunk_overlap,
            step=50,
            help="相邻块之间的重叠字符数，建议 100-200",
        )

    retrieval_top_k = st.slider(
        "检索返回数量 (Top K)",
        min_value=1,
        max_value=20,
        value=settings.retrieval_top_k,
        help="每次检索返回的最相关文档块数量",
    )

    if st.button("保存 RAG 配置", type="primary", use_container_width=True):
        new_settings = Settings(
            api_provider=settings.api_provider,
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
            openai_model=settings.openai_model,
            anthropic_api_key=settings.anthropic_api_key,
            anthropic_model=settings.anthropic_model,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
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
**标书智能助手** 是一个基于 LangGraph + RAG 的标书生成与完善系统。

#### 技术栈
| 组件 | 技术 |
|------|------|
| AI 工作流 | LangGraph |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| 前端界面 | Streamlit |
| AI 模型 | OpenAI / Anthropic |

#### 工作流程
```
输入标书
   ↓
文档分析节点（LangGraph）
   ↓
RAG 检索节点（从 ChromaDB 检索）
   ↓
章节完善节点 / 分析节点
   ↓
缺失章节生成节点
   ↓
文档整合节点
   ↓
输出 + 可选入库
```

#### 数据安全
- 所有文档和 API Key 均存储在本地
- API Key 仅用于调用 AI 接口，不会上传到任何服务器
    """)
