import streamlit as st

st.set_page_config(
    page_title="标书智能助手",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📋 标书智能助手")
st.markdown("基于 LangGraph + RAG 的标书生成与完善系统")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📁 文档管理")
    st.markdown("上传标书文档到知识库，支持 PDF、Word、TXT 格式。文档将被分块向量化存储，供 RAG 检索使用。")
    st.page_link("pages/1_文档管理.py", label="进入文档管理", icon="📁")

with col2:
    st.markdown("### ✨ 标书生成")
    st.markdown("输入标书内容和需求，系统将自动分析结构、检索参考资料，并生成完善的标书文档。")
    st.page_link("pages/2_标书生成.py", label="开始生成标书", icon="✨")

with col3:
    st.markdown("### ⚙️ 系统设置")
    st.markdown("配置 AI API 密钥（支持 OpenAI / Anthropic）、向量数据库参数及文档处理选项。")
    st.page_link("pages/3_系统设置.py", label="系统设置", icon="⚙️")

st.markdown("---")
st.markdown("### 工作流程")
st.markdown("""
```
输入标书 → 文档分析 → RAG检索参考资料 → 章节完善/生成 → 整合输出 → 入库保存
```
""")

with st.expander("系统状态"):
    try:
        from config.settings import get_settings
        settings = get_settings()
        if settings.is_configured():
            st.success(f"AI API 已配置：{settings.api_provider.upper()} - {settings.openai_model if settings.api_provider == 'openai' else settings.anthropic_model}")
        else:
            st.warning("AI API 未配置，请前往系统设置页面填写 API Key。")

        try:
            from rag.vector_store import VectorStoreManager
            vsm = VectorStoreManager(settings)
            stats = vsm.get_collection_stats()
            st.info(f"知识库：{stats['collection_name']} | 文档块数：{stats['total_documents']}")
        except Exception as e:
            st.info("知识库尚未初始化（首次上传文档后自动创建）")
    except Exception as e:
        st.error(f"配置加载失败：{e}")
