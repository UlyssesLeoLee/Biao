import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import Settings
from graphs.state import BiddingState, SectionResult
from rag.retriever import BiddingRetriever


def get_llm(settings: Settings):
    if settings.api_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.anthropic_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.3,
        )
    else:
        from langchain_openai import ChatOpenAI
        kwargs: Dict[str, Any] = {
            "model": settings.openai_model,
            "openai_api_key": settings.openai_api_key,
            "temperature": 0.3,
        }
        if settings.openai_base_url and settings.openai_base_url != "https://api.openai.com/v1":
            kwargs["openai_api_base"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)


def make_nodes(settings: Settings):
    llm = get_llm(settings)
    retriever = BiddingRetriever(settings)

    def analyze_document(state: BiddingState) -> BiddingState:
        input_doc = state.get("input_document", "")
        user_reqs = state.get("user_requirements", "")
        doc_type = state.get("document_type", "标书")

        prompt = f"""你是一位专业的标书分析专家。请对以下{doc_type}进行深入分析。

用户需求：{user_reqs}

标书内容：
{input_doc[:4000]}

请以JSON格式返回分析结果，包含以下字段：
{{
  "document_structure": "文档整体结构描述（2-3句话）",
  "identified_sections": ["已有章节1", "已有章节2", ...],
  "missing_sections": ["缺失章节1", "缺失章节2", ...],
  "improvement_points": ["改进点1", "改进点2", ...]
}}

注意：只返回JSON，不要有其他内容。"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            result = json.loads(content)
            return {
                **state,
                "document_structure": result.get("document_structure", ""),
                "identified_sections": result.get("identified_sections", []),
                "missing_sections": result.get("missing_sections", []),
                "improvement_points": result.get("improvement_points", []),
                "processing_log": [f"[分析] 识别到 {len(result.get('identified_sections', []))} 个章节，"
                                   f"{len(result.get('missing_sections', []))} 个缺失章节"],
                "current_section_index": 0,
            }
        except Exception as e:
            return {
                **state,
                "document_structure": "分析失败",
                "identified_sections": [],
                "missing_sections": [],
                "improvement_points": [],
                "error_message": f"文档分析失败: {str(e)}",
                "processing_log": [f"[错误] 文档分析失败: {str(e)}"],
                "current_section_index": 0,
            }

    def retrieve_context(state: BiddingState) -> BiddingState:
        user_reqs = state.get("user_requirements", "")
        doc_type = state.get("document_type", "标书")
        improvement_points = state.get("improvement_points", [])
        query = f"{doc_type} {user_reqs} {' '.join(improvement_points[:3])}"

        try:
            docs = retriever.retrieve_similar_bids(query, k=settings.retrieval_top_k)
            context = retriever.format_context(docs)
            return {
                **state,
                "retrieved_context": context,
                "processing_log": [f"[检索] 找到 {len(docs)} 条相关参考资料"],
            }
        except Exception as e:
            return {
                **state,
                "retrieved_context": "暂无相关参考资料。",
                "processing_log": [f"[检索] 向量检索异常（可能知识库为空）: {str(e)}"],
            }

    def enhance_sections(state: BiddingState) -> BiddingState:
        input_doc = state.get("input_document", "")
        user_reqs = state.get("user_requirements", "")
        context = state.get("retrieved_context", "")
        identified = state.get("identified_sections", [])
        improvement_points = state.get("improvement_points", [])
        doc_type = state.get("document_type", "标书")

        if not identified:
            return {**state, "section_results": [], "processing_log": ["[完善] 无章节需要完善"]}

        sections_to_process = identified[:6]  # limit to avoid token overflow
        improvement_str = "\n".join(f"- {p}" for p in improvement_points)
        sections_str = "\n".join(f"- {s}" for s in sections_to_process)

        prompt = f"""你是一位专业的标书撰写专家。请根据用户需求和参考资料，完善以下{doc_type}的各章节内容。

用户需求：{user_reqs}

需要完善的章节：
{sections_str}

改进要点：
{improvement_str}

原始文档（部分）：
{input_doc[:3000]}

参考资料：
{context[:2500]}

请逐章节提供完善后的内容，格式如下（严格按此格式，章节之间用 === 分隔）：
[章节名称]
完善后的内容...
===
[下一章节名称]
完善后的内容...
==="""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            section_results = _parse_section_results(raw, sections_to_process, "enhanced")
            return {
                **state,
                "section_results": section_results,
                "processing_log": [f"[完善] 完成 {len(section_results)} 个章节的完善"],
            }
        except Exception as e:
            return {
                **state,
                "section_results": [],
                "error_message": f"章节完善失败: {str(e)}",
                "processing_log": [f"[错误] 章节完善失败: {str(e)}"],
            }

    def generate_missing_sections(state: BiddingState) -> BiddingState:
        missing = state.get("missing_sections", [])
        user_reqs = state.get("user_requirements", "")
        context = state.get("retrieved_context", "")
        doc_type = state.get("document_type", "标书")
        input_doc = state.get("input_document", "")

        if not missing:
            return {**state, "processing_log": ["[生成] 无缺失章节需要生成"]}

        missing_to_generate = missing[:4]
        missing_str = "\n".join(f"- {s}" for s in missing_to_generate)

        prompt = f"""你是一位专业的标书撰写专家。请为以下{doc_type}生成缺失的章节内容。

用户需求：{user_reqs}

需要生成的章节：
{missing_str}

现有文档摘要：
{input_doc[:2000]}

参考资料：
{context[:2500]}

请生成各缺失章节的完整内容，格式如下（严格按此格式，章节之间用 === 分隔）：
[章节名称]
生成的内容...
===
[下一章节名称]
生成的内容...
==="""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            new_sections = _parse_section_results(raw, missing_to_generate, "generated")
            return {
                **state,
                "section_results": new_sections,
                "processing_log": [f"[生成] 生成 {len(new_sections)} 个缺失章节"],
            }
        except Exception as e:
            return {
                **state,
                "error_message": f"章节生成失败: {str(e)}",
                "processing_log": [f"[错误] 章节生成失败: {str(e)}"],
            }

    def compile_document(state: BiddingState) -> BiddingState:
        input_doc = state.get("input_document", "")
        user_reqs = state.get("user_requirements", "")
        section_results = state.get("section_results", [])
        doc_type = state.get("document_type", "标书")
        doc_structure = state.get("document_structure", "")

        enhanced_content = "\n\n".join(
            f"## {s['name']}\n\n{s['content']}"
            for s in section_results
            if s.get("content")
        )

        prompt = f"""你是一位专业的标书撰写专家。请将以下内容整合成一份完整、规范的{doc_type}。

用户需求：{user_reqs}

原始文档：
{input_doc[:2000]}

已完善/生成的章节内容：
{enhanced_content[:4000]}

文档结构分析：{doc_structure}

请输出一份格式规范、内容完整的{doc_type}，要求：
1. 保持专业的标书格式和语言
2. 章节结构清晰，层次分明
3. 内容具体，有说服力
4. 符合用户需求：{user_reqs}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            final_doc = response.content.strip()
            processed_count = len(section_results)
            summary = (
                f"处理完成：共完善/生成 {processed_count} 个章节。\n"
                f"文档结构：{doc_structure[:200]}"
            )
            return {
                **state,
                "final_document": final_doc,
                "summary": summary,
                "processing_log": ["[整合] 标书整合完成"],
            }
        except Exception as e:
            fallback = f"# {doc_type}\n\n{enhanced_content}"
            return {
                **state,
                "final_document": fallback,
                "summary": f"整合时出现错误，已输出原始内容: {str(e)}",
                "processing_log": [f"[警告] 文档整合异常: {str(e)}"],
            }

    def analyze_only(state: BiddingState) -> BiddingState:
        input_doc = state.get("input_document", "")
        user_reqs = state.get("user_requirements", "")
        doc_type = state.get("document_type", "标书")
        context = state.get("retrieved_context", "")
        doc_structure = state.get("document_structure", "")
        missing = state.get("missing_sections", [])
        improvements = state.get("improvement_points", [])

        prompt = f"""你是一位专业的标书审核专家。请对以下{doc_type}进行全面的质量分析和评估。

用户需求：{user_reqs}

标书内容：
{input_doc[:3000]}

已识别的问题：
- 缺失章节：{', '.join(missing) if missing else '无'}
- 改进点：{', '.join(improvements[:5]) if improvements else '无'}

参考资料：
{context[:1500]}

请提供详细的分析报告，包括：
1. 文档总体评估
2. 各章节质量评分（1-10分）
3. 主要优点
4. 主要问题和改进建议
5. 与用户需求的匹配度分析"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return {
                **state,
                "final_document": response.content.strip(),
                "summary": f"文档分析完成。结构：{doc_structure[:100]}",
                "processing_log": ["[分析] 标书分析报告生成完成"],
            }
        except Exception as e:
            return {
                **state,
                "final_document": f"分析失败: {str(e)}",
                "summary": "分析失败",
                "processing_log": [f"[错误] 分析失败: {str(e)}"],
            }

    return {
        "analyze_document": analyze_document,
        "retrieve_context": retrieve_context,
        "enhance_sections": enhance_sections,
        "generate_missing_sections": generate_missing_sections,
        "compile_document": compile_document,
        "analyze_only": analyze_only,
    }


def _parse_section_results(raw: str, section_names: list, status: str) -> list:
    results = []
    parts = raw.split("===")
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n")
        name = ""
        content_lines = []
        for j, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]") and not name:
                name = stripped[1:-1]
            else:
                content_lines.append(line)
        if not name and i < len(section_names):
            name = section_names[i]
        content = "\n".join(content_lines).strip()
        if name and content:
            results.append(SectionResult(name=name, content=content, status=status))
    return results
