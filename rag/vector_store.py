from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import Settings
from rag.embeddings import get_embeddings


class VectorStoreManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._store: Optional[Chroma] = None
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    def _get_store(self) -> Chroma:
        if self._store is None:
            embeddings = get_embeddings(self.settings)
            self._store = Chroma(
                collection_name=self.settings.chroma_collection_name,
                embedding_function=embeddings,
                persist_directory=self.settings.chroma_persist_dir,
            )
        return self._store

    def add_documents(self, documents: List[Document]) -> List[str]:
        store = self._get_store()
        ids = store.add_documents(documents)
        return ids

    def similarity_search(self, query: str, k: int = None, filter: Optional[Dict] = None) -> List[Document]:
        k = k or self.settings.retrieval_top_k
        store = self._get_store()
        return store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        k = k or self.settings.retrieval_top_k
        store = self._get_store()
        return store.similarity_search_with_relevance_scores(query, k=k)

    def get_collection_stats(self) -> Dict[str, Any]:
        store = self._get_store()
        collection = store._collection
        count = collection.count()
        return {"total_documents": count, "collection_name": self.settings.chroma_collection_name}

    def list_sources(self) -> List[str]:
        store = self._get_store()
        collection = store._collection
        results = collection.get(include=["metadatas"])
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(list(sources))

    def delete_by_source(self, source: str) -> int:
        store = self._get_store()
        collection = store._collection
        results = collection.get(where={"source": source}, include=["metadatas"])
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def reset_collection(self) -> None:
        store = self._get_store()
        store.delete_collection()
        self._store = None

    def as_retriever(self, k: int = None, filter: Optional[Dict] = None):
        k = k or self.settings.retrieval_top_k
        store = self._get_store()
        search_kwargs = {"k": k}
        if filter:
            search_kwargs["filter"] = filter
        return store.as_retriever(search_kwargs=search_kwargs)
