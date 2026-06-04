from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )

    def load_from_bytes(self, content: bytes, filename: str, metadata: dict = None) -> List[Document]:
        from plugins.registry import registry
        suffix = Path(filename).suffix.lower()
        loader_plugin = registry.get_loader_for(filename)
        text = loader_plugin.extract(content, filename)
        base_meta = {"source": filename, "file_type": suffix}
        if metadata:
            base_meta.update(metadata)
        doc = Document(page_content=text, metadata=base_meta)
        return self.splitter.split_documents([doc])

    def load_from_text(self, text: str, source: str = "direct_input", metadata: dict = None) -> List[Document]:
        base_meta = {"source": source, "file_type": "text"}
        if metadata:
            base_meta.update(metadata)
        doc = Document(page_content=text, metadata=base_meta)
        return self.splitter.split_documents([doc])
