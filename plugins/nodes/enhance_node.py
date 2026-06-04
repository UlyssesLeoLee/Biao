from langchain_core.messages import HumanMessage

from plugins.base import NodePlugin, NodeContext
from plugins.nodes import parse_section_results
from plugins.registry import registry


class EnhanceSectionsNode(NodePlugin):
    plugin_name = "enhance_sections"

    def execute(self, state: dict, ctx: NodeContext) -> dict:
        input_doc = state.get("input_document", "")
        user_reqs = state.get("user_requirements", "")
        context = state.get("retrieved_context", "")
        identified = state.get("identified_sections", [])
        improvement_points = state.get("improvement_points", [])
        doc_type = state.get("document_type", "标书")

        if not identified:
            return {**state, "section_results": [], "processing_log": ["[完善] 无章节需要完善"]}

        sections_to_process = identified[:6]
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
            response = ctx.llm.invoke([HumanMessage(content=prompt)])
            results = parse_section_results(response.content.strip(), sections_to_process, "enhanced")
            return {
                **state,
                "section_results": results,
                "processing_log": [f"[完善] 完成 {len(results)} 个章节的完善"],
            }
        except Exception as e:
            return {
                **state,
                "section_results": [],
                "error_message": f"章节完善失败: {str(e)}",
                "processing_log": [f"[错误] 章节完善失败: {str(e)}"],
            }


registry.register_node(EnhanceSectionsNode())
