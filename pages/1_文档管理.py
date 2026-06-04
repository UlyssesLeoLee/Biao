import streamlit as st

st.set_page_config(page_title="文档管理 - 标书智能助手", page_icon="📁", layout="wide")
st.title("📁 文档管理")
st.caption("上传标书文档到知识库，支持 PDF、Word、TXT 格式")

from config.settings import get_settings, reload_settings
from rag.loader import DocumentLoader
from rag.vector_store import VectorStoreManager

settings = get_settings()

if not settings.is_configured():
    st.warning("请先在系统设置页面配置 AI API Key，Embedding 功能需要 API Key。")
    st.page_link("pages/3_系统设置.py", label="前往系统设置", icon="⚙️")
    st.stop()

tab_upload, tab_manage, tab_search = st.tabs(["上传文档", "知识库管理", "内容检索"])

with tab_upload:
    st.markdown("### 上传文档到知识库")

    uploaded_files = st.file_uploader(
        "选择标书文档",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True,
        help="支持 PDF、Word(.docx)、纯文本(.txt/.md) 格式",
    )

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        doc_category = st.selectbox(
            "文档分类",
            ["投标书", "招标文件", "合同范本", "技术方案", "报价文件", "其他"],
        )
    with col_meta2:
        doc_tags = st.text_input("标签（逗号分隔）", placeholder="例如：IT项目,2024年,政府采购")

    direct_text = st.text_area(
        "或直接粘贴文本内容",
        height=200,
        placeholder="将标书文本直接粘贴到这里...",
    )
    direct_text_name = st.text_input("文本标题", placeholder="例如：XX项目投标书2024", key="direct_name")

    if st.button("上传入库", type="primary", use_container_width=True):
        settings = reload_settings()
        loader = DocumentLoader(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        vsm = VectorStoreManager(settings)

        metadata = {
            "category": doc_category,
            "tags": doc_tags,
        }

        total_added = 0
        errors = []

        if uploaded_files:
            progress = st.progress(0, text="处理中...")
            for i, file in enumerate(uploaded_files):
                progress.progress((i + 1) / len(uploaded_files), text=f"处理 {file.name}...")
                try:
                    content = file.read()
                    docs = loader.load_from_bytes(content, file.name, metadata.copy())
                    vsm.add_documents(docs)
                    total_added += len(docs)
                    st.success(f"{file.name}：已入库 {len(docs)} 个文档块")
                except Exception as e:
                    errors.append(f"{file.name}: {str(e)}")
            progress.empty()

        if direct_text.strip():
            name = direct_text_name.strip() or "直接输入文本"
            try:
                docs = loader.load_from_text(direct_text, source=name, metadata=metadata.copy())
                vsm.add_documents(docs)
                total_added += len(docs)
                st.success(f"文本内容已入库 {len(docs)} 个文档块")
            except Exception as e:
                errors.append(f"文本内容: {str(e)}")

        if errors:
            for err in errors:
                st.error(err)

        if total_added > 0:
            st.balloons()
            st.success(f"入库完成，共添加 {total_added} 个文档块到知识库")
        elif not errors:
            st.info("请上传文件或输入文本内容")


with tab_manage:
    st.markdown("### 知识库管理")
    settings = reload_settings()

    try:
        vsm = VectorStoreManager(settings)
        stats = vsm.get_collection_stats()
        sources = vsm.list_sources()

        col1, col2 = st.columns(2)
        col1.metric("知识库文档块总数", stats["total_documents"])
        col2.metric("文档来源数", len(sources))

        if sources:
            st.markdown("#### 已入库文档")
            for source in sources:
                col_s, col_d = st.columns([4, 1])
                col_s.markdown(f"- `{source}`")
                if col_d.button("删除", key=f"del_{source}", help=f"删除来自 {source} 的所有文档块"):
                    deleted = vsm.delete_by_source(source)
                    st.success(f"已删除 {deleted} 个文档块（来源：{source}）")
                    st.rerun()
        else:
            st.info("知识库为空，请先上传文档")

        st.markdown("---")
        with st.expander("危险操作", expanded=False):
            st.warning("以下操作不可撤销！")
            if st.button("清空整个知识库", type="secondary"):
                vsm.reset_collection()
                st.success("知识库已清空")
                st.rerun()

    except Exception as e:
        st.error(f"加载知识库失败：{e}")


with tab_search:
    st.markdown("### 知识库内容检索")
    query = st.text_input("输入搜索关键词", placeholder="例如：项目管理方案、技术实施路线...")
    k = st.slider("返回数量", 1, 10, 5)

    if st.button("搜索", type="primary") and query:
        settings = reload_settings()
        try:
            from rag.retriever import BiddingRetriever
            retriever = BiddingRetriever(settings)
            results = retriever.store.similarity_search_with_score(query, k=k)

            if results:
                st.markdown(f"找到 **{len(results)}** 条相关内容：")
                for i, (doc, score) in enumerate(results, 1):
                    with st.expander(f"结果 {i}｜相关度: {score:.3f}｜来源: {doc.metadata.get('source', '未知')}"):
                        st.markdown(doc.page_content)
                        st.caption(f"元数据：{doc.metadata}")
            else:
                st.info("未找到相关内容，请尝试其他关键词或先上传相关文档")
        except Exception as e:
            st.error(f"检索失败：{e}")
