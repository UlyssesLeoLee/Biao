from langchain_core.messages import HumanMessage

from plugins.base import NodePlugin, NodeContext
from plugins.registry import registry


class AnalyzeOnlyNode(NodePlugin):
    plugin_name = "analyze_only"

    def execute(self, state: dict, ctx: NodeContext) -> dict:
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
            response = ctx.llm.invoke([HumanMessage(content=prompt)])
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


registry.register_node(AnalyzeOnlyNode())
