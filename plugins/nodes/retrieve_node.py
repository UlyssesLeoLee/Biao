from plugins.base import NodePlugin, NodeContext
from plugins.registry import registry


class RetrieveContextNode(NodePlugin):
    plugin_name = "retrieve_context"

    def execute(self, state: dict, ctx: NodeContext) -> dict:
        user_reqs = state.get("user_requirements", "")
        doc_type = state.get("document_type", "标书")
        improvements = state.get("improvement_points", [])
        query = f"{doc_type} {user_reqs} {' '.join(improvements[:3])}".strip()

        try:
            docs = ctx.retriever.retrieve_similar_bids(query, k=ctx.settings.retrieval_top_k)
            context = ctx.retriever.format_context(docs)
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


registry.register_node(RetrieveContextNode())
