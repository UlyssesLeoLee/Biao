from pathlib import Path
from typing import Any, Dict, List


class PluginRegistry:
    _llm: Dict[str, Any] = {}
    _embeddings: Dict[str, Any] = {}
    _loaders: List[Any] = []
    _nodes: Dict[str, Any] = {}

    @classmethod
    def register_llm(cls, plugin: Any) -> None:
        cls._llm[plugin.plugin_name] = plugin

    @classmethod
    def register_embedding(cls, plugin: Any) -> None:
        cls._embeddings[plugin.plugin_name] = plugin

    @classmethod
    def register_loader(cls, plugin: Any) -> None:
        cls._loaders.append(plugin)

    @classmethod
    def register_node(cls, plugin: Any) -> None:
        cls._nodes[plugin.plugin_name] = plugin

    @classmethod
    def get_llm(cls, provider: str) -> Any:
        if provider not in cls._llm:
            raise ValueError(f"未找到 LLM 插件 '{provider}'，已注册: {list(cls._llm)}")
        return cls._llm[provider]

    @classmethod
    def get_embedding(cls, provider: str) -> Any:
        if provider not in cls._embeddings:
            raise ValueError(f"未找到 Embedding 插件 '{provider}'，已注册: {list(cls._embeddings)}")
        return cls._embeddings[provider]

    @classmethod
    def get_loader_for(cls, filename: str) -> Any:
        ext = Path(filename).suffix.lower()
        for loader in cls._loaders:
            if loader.can_load(ext):
                return loader
        for loader in cls._loaders:
            if loader.plugin_name == "text":
                return loader
        raise ValueError(f"无法找到支持 '{ext}' 格式的加载插件")

    @classmethod
    def get_node(cls, name: str) -> Any:
        if name not in cls._nodes:
            raise ValueError(f"未找到节点插件 '{name}'，已注册: {list(cls._nodes)}")
        return cls._nodes[name]

    @classmethod
    def list_llm_providers(cls) -> List[str]:
        return list(cls._llm)

    @classmethod
    def list_embedding_providers(cls) -> List[str]:
        return list(cls._embeddings)


registry = PluginRegistry
