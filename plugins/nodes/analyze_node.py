import json

from langchain_core.messages import HumanMessage

from plugins.base import NodePlugin, NodeContext
from plugins.registry import registry


class AnalyzeDocumentNode(NodePlugin):
    plugin_name = "analyze_document"

    def execute(self, state: dict, ctx: NodeContext) -> dict:
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
            response = ctx.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            result = json.loads(content)
            identified = result.get("identified_sections", [])
            missing = result.get("missing_sections", [])
            return {
                **state,
                "document_structure": result.get("document_structure", ""),
                "identified_sections": identified,
                "missing_sections": missing,
                "improvement_points": result.get("improvement_points", []),
                "processing_log": [f"[分析] 识别到 {len(identified)} 个章节，{len(missing)} 个缺失章节"],
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


registry.register_node(AnalyzeDocumentNode())
