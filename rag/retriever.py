from typing import List, Optional

from langchain_core.documents import Document

from config.settings import Settings
from rag.vector_store import VectorStoreManager


class BiddingRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = VectorStoreManager(settings)

    def retrieve_for_section(self, section_name: str, context: str = "", k: int = None) -> List[Document]:
        query = f"{section_name} {context}".strip()
        return self.store.similarity_search(query, k=k)

    def retrieve_similar_bids(self, requirement: str, k: int = None) -> List[Document]:
        return self.store.similarity_search(requirement, k=k)

    def retrieve_with_scores(self, query: str, k: int = None, threshold: float = 0.3) -> List[Document]:
        results = self.store.similarity_search_with_score(query, k=k)
        return [doc for doc, score in results if score >= threshold]

    def format_context(self, documents: List[Document], max_chars: int = 6000) -> str:
        if not documents:
            return "暂无相关参考资料。"
        parts = []
        total = 0
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "未知来源")
            content = doc.page_content.strip()
            entry = f"[参考{i} - 来源: {source}]\n{content}"
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n---\n\n".join(parts)
