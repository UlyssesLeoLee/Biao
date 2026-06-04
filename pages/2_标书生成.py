import streamlit as st
import time
from datetime import datetime

st.set_page_config(page_title="标书生成 - 标书智能助手", page_icon="✨", layout="wide")
st.title("✨ 标书生成与完善")
st.caption("输入标书内容和需求，AI 将自动分析、检索参考资料并生成完善的标书")

from config.settings import get_settings, reload_settings

settings = get_settings()
if not settings.is_configured():
    st.warning("请先在系统设置页面配置 AI API Key。")
    st.page_link("pages/3_系统设置.py", label="前往系统设置", icon="⚙️")
    st.stop()

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 输入区域")

    doc_type = st.selectbox(
        "标书类型",
        ["投标书", "招标文件", "技术方案", "商务方案", "项目建议书", "自定义"],
        help="选择要处理的标书类型",
    )
    if doc_type == "自定义":
        doc_type = st.text_input("自定义类型名称", placeholder="例如：可行性研究报告")

    operation_mode = st.radio(
        "处理模式",
        ["enhance", "analyze"],
        format_func=lambda x: {"enhance": "完善/生成（修改文档）", "analyze": "分析（只分析不修改）"}[x],
        horizontal=True,
    )

    input_tab1, input_tab2 = st.tabs(["粘贴文本", "上传文件"])

    input_document = ""
    with input_tab1:
        input_document = st.text_area(
            "标书原文",
            height=300,
            placeholder="将标书内容粘贴到这里，可以是草稿或完整文档...\n\n如果是全新标书，可以只写标题和基本要求，AI 会自动补全。",
            key="doc_text",
        )

    with input_tab2:
        uploaded = st.file_uploader(
            "上传标书文件",
            type=["pdf", "docx", "doc", "txt"],
            key="doc_file",
        )
        if uploaded:
            from rag.loader import DocumentLoader
            loader = DocumentLoader()
            chunks = loader.load_from_bytes(uploaded.read(), uploaded.name)
            input_document = "\n\n".join(c.page_content for c in chunks)
            st.success(f"已加载：{uploaded.name}（{len(input_document)} 字）")
            with st.expander("预览内容"):
                st.text(input_document[:1000] + ("..." if len(input_document) > 1000 else ""))

    user_requirements = st.text_area(
        "用户需求",
        height=120,
        placeholder="描述你的具体需求，例如：\n- 需要突出公司的技术实力和项目经验\n- 增加风险控制方案章节\n- 完善报价明细表\n- 语言要更加专业正式",
    )

    col_adv1, col_adv2 = st.columns(2)
    with col_adv1:
        save_to_kb = st.checkbox("处理完成后入库", value=True, help="将生成的标书存入知识库供后续参考")
    with col_adv2:
        use_rag = st.checkbox("启用 RAG 检索", value=True, help="从知识库检索相关参考资料")

    run_btn = st.button(
        "开始处理" if operation_mode == "enhance" else "开始分析",
        type="primary",
        use_container_width=True,
        disabled=not (input_document.strip() or user_requirements.strip()),
    )

with col_output:
    st.markdown("### 输出区域")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "processing_logs" not in st.session_state:
        st.session_state.processing_logs = []

    if run_btn:
        if not input_document.strip() and not user_requirements.strip():
            st.error("请输入标书内容或需求描述")
        else:
            settings = reload_settings()
            from graphs.bidding_graph import BiddingDocumentGraph

            log_container = st.empty()
            logs = []

            def update_log(msg):
                logs.append(f"{datetime.now().strftime('%H:%M:%S')} {msg}")
                log_container.info("\n".join(logs[-6:]))

            update_log("正在初始化 AI 处理流程...")
            start_time = time.time()

            try:
                graph = BiddingDocumentGraph(settings)
                initial_state = {
                    "input_document": input_document,
                    "user_requirements": user_requirements,
                    "document_type": doc_type,
                    "operation_mode": operation_mode,
                }

                with st.spinner("AI 正在处理中，请稍候..."):
                    result = graph.run(initial_state)

                elapsed = time.time() - start_time
                logs_from_result = result.get("processing_log", [])

                st.session_state.last_result = result
                st.session_state.processing_logs = logs_from_result

                log_container.success(f"处理完成，耗时 {elapsed:.1f} 秒")

                if result.get("error_message"):
                    st.warning(f"处理中有警告：{result['error_message']}")

                if save_to_kb and result.get("final_document"):
                    from rag.loader import DocumentLoader
                    from rag.vector_store import VectorStoreManager
                    loader = DocumentLoader(settings.chunk_size, settings.chunk_overlap)
                    vsm = VectorStoreManager(settings)
                    source_name = f"AI生成_{doc_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    docs = loader.load_from_text(
                        result["final_document"],
                        source=source_name,
                        metadata={"category": doc_type, "generated": "true"},
                    )
                    vsm.add_documents(docs)
                    st.success(f"已将结果保存到知识库（{len(docs)} 个块）")

            except Exception as e:
                log_container.error(f"处理失败：{str(e)}")
                st.exception(e)

    if st.session_state.last_result:
        result = st.session_state.last_result

        res_tab1, res_tab2, res_tab3 = st.tabs(["最终结果", "处理分析", "处理日志"])

        with res_tab1:
            final_doc = result.get("final_document", "")
            if final_doc:
                st.markdown(final_doc)
                st.markdown("---")
                st.download_button(
                    "下载结果",
                    data=final_doc,
                    file_name=f"{doc_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            else:
                st.info("暂无输出结果")

        with res_tab2:
            summary = result.get("summary", "")
            if summary:
                st.markdown("#### 处理摘要")
                st.markdown(summary)

            sections = result.get("section_results", [])
            if sections:
                st.markdown(f"#### 章节处理情况（共 {len(sections)} 个）")
                for s in sections:
                    status_icon = {"enhanced": "✅", "generated": "🆕", "skipped": "⏭️"}.get(s.get("status", ""), "•")
                    with st.expander(f"{status_icon} {s.get('name', '未命名')} [{s.get('status', '')}]"):
                        st.markdown(s.get("content", ""))

            identified = result.get("identified_sections", [])
            missing = result.get("missing_sections", [])
            improvements = result.get("improvement_points", [])
            if identified:
                st.markdown("#### 原文已有章节")
                for s in identified:
                    st.markdown(f"- {s}")
            if missing:
                st.markdown("#### 补充生成章节")
                for s in missing:
                    st.markdown(f"- {s}")
            if improvements:
                st.markdown("#### 改进点")
                for p in improvements:
                    st.markdown(f"- {p}")

        with res_tab3:
            logs = st.session_state.processing_logs
            if logs:
                for log in logs:
                    st.text(log)
            else:
                st.info("暂无日志")
    else:
        st.info("输入内容并点击「开始处理」后，结果将显示在这里")
        with st.expander("使用说明"):
            st.markdown("""
**完善/生成模式：**
1. 粘贴或上传现有标书（草稿也可以）
2. 描述你的具体需求（如：突出技术优势、增加风险方案...）
3. 点击「开始处理」
4. AI 会分析文档结构，从知识库检索参考，然后完善各章节

**分析模式：**
- 对标书进行全面质量评估
- 输出评分报告和改进建议
- 不修改原文档

**建议：**
- 先在文档管理页面上传一些参考标书，RAG 效果更好
- 需求描述越具体，AI 处理结果越精准
            """)
