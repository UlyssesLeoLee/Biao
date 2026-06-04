from config.settings import Settings
from graphs.state import BiddingState
from rag.retriever import BiddingRetriever


def get_llm(settings: Settings):
    from plugins.registry import registry
    return registry.get_llm(settings.api_provider).build(settings)


def make_nodes(settings: Settings):
    from plugins.base import NodeContext
    from plugins.registry import registry

    llm = get_llm(settings)
    retriever = BiddingRetriever(settings)
    ctx = NodeContext(llm=llm, retriever=retriever, settings=settings)

    node_names = [
        "analyze_document",
        "retrieve_context",
        "enhance_sections",
        "generate_missing_sections",
        "compile_document",
        "analyze_only",
    ]

    def _wrap(name: str):
        plugin = registry.get_node(name)
        def fn(state: BiddingState) -> dict:
            return plugin.execute(state, ctx)
        fn.__name__ = name
        return fn

    return {name: _wrap(name) for name in node_names}
