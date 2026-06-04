from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List


@dataclass
class NodeContext:
    llm: Any
    retriever: Any
    settings: Any


class LLMPlugin(ABC):
    plugin_name: str = ""
    display_name: str = ""

    @abstractmethod
    def build(self, settings: Any) -> Any:
        """Return a LangChain BaseChatModel instance."""

    def is_configured(self, settings: Any) -> bool:
        return bool(self.get_api_key(settings))

    def get_api_key(self, settings: Any) -> str:
        return ""


class EmbeddingPlugin(ABC):
    plugin_name: str = ""
    display_name: str = ""

    @abstractmethod
    def build(self, settings: Any) -> Any:
        """Return a LangChain Embeddings instance."""


class LoaderPlugin(ABC):
    plugin_name: str = ""
    supported_extensions: List[str] = []

    def can_load(self, ext: str) -> bool:
        return ext.lower() in self.supported_extensions

    @abstractmethod
    def extract(self, content: bytes, filename: str) -> str:
        """Extract plain text from file bytes."""


class NodePlugin(ABC):
    plugin_name: str = ""

    @abstractmethod
    def execute(self, state: dict, ctx: NodeContext) -> dict:
        """Execute this LangGraph node. Returns a partial state update dict."""
