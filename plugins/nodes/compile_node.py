from langchain_core.messages import HumanMessage

from plugins.base import NodePlugin, NodeContext
from plugins.registry import registry


class CompileDocumentNode(NodePlugin):
    plugin_name = "compile_document"

    def execute(self, state: dict, ctx: NodeContext) -> dict:
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
            response = ctx.llm.invoke([HumanMessage(content=prompt)])
            summary = (
                f"处理完成：共完善/生成 {len(section_results)} 个章节。\n"
                f"文档结构：{doc_structure[:200]}"
            )
            return {
                **state,
                "final_document": response.content.strip(),
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


registry.register_node(CompileDocumentNode())
