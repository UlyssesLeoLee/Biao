from typing import Dict, Any

from langgraph.graph import END, START, StateGraph

from config.settings import Settings
from graphs.nodes import make_nodes
from graphs.state import BiddingState


class BiddingDocumentGraph:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._graph = None

    def _build(self):
        nodes = make_nodes(self.settings)
        builder = StateGraph(BiddingState)

        builder.add_node("analyze_document", nodes["analyze_document"])
        builder.add_node("retrieve_context", nodes["retrieve_context"])
        builder.add_node("enhance_sections", nodes["enhance_sections"])
        builder.add_node("generate_missing_sections", nodes["generate_missing_sections"])
        builder.add_node("compile_document", nodes["compile_document"])
        builder.add_node("analyze_only", nodes["analyze_only"])

        builder.add_edge(START, "analyze_document")
        builder.add_edge("analyze_document", "retrieve_context")

        builder.add_conditional_edges(
            "retrieve_context",
            _route_after_retrieval,
            {
                "enhance": "enhance_sections",
                "analyze": "analyze_only",
            },
        )

        builder.add_edge("enhance_sections", "generate_missing_sections")
        builder.add_edge("generate_missing_sections", "compile_document")
        builder.add_edge("compile_document", END)
        builder.add_edge("analyze_only", END)

        self._graph = builder.compile()

    def run(self, initial_state: Dict[str, Any]) -> BiddingState:
        if self._graph is None:
            self._build()
        defaults = {
            "section_results": [],
            "processing_log": [],
            "current_section_index": 0,
            "error_message": None,
            "retrieved_context": "",
            "document_structure": "",
            "identified_sections": [],
            "missing_sections": [],
            "improvement_points": [],
            "final_document": "",
            "summary": "",
        }
        state = {**defaults, **initial_state}
        return self._graph.invoke(state)


def _route_after_retrieval(state: BiddingState) -> str:
    mode = state.get("operation_mode", "enhance")
    if mode == "analyze":
        return "analyze"
    return "enhance"
