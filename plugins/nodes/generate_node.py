from langchain_core.messages import HumanMessage

from plugins.base import NodePlugin, NodeContext
from plugins.nodes import parse_section_results
from plugins.registry import registry


class GenerateMissingSectionsNode(NodePlugin):
    plugin_name = "generate_missing_sections"

    def execute(self, state: dict, ctx: NodeContext) -> dict:
        missing = state.get("missing_sections", [])
        user_reqs = state.get("user_requirements", "")
        context = state.get("retrieved_context", "")
        doc_type = state.get("document_type", "标书")
        input_doc = state.get("input_document", "")

        if not missing:
            return {**state, "processing_log": ["[生成] 无缺失章节需要生成"]}

        to_generate = missing[:4]
        missing_str = "\n".join(f"- {s}" for s in to_generate)

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
            response = ctx.llm.invoke([HumanMessage(content=prompt)])
            new_sections = parse_section_results(response.content.strip(), to_generate, "generated")
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


registry.register_node(GenerateMissingSectionsNode())
