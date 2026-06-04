from plugins.base import EmbeddingPlugin
from plugins.registry import registry


class OpenAIEmbeddingPlugin(EmbeddingPlugin):
    plugin_name = "openai"
    display_name = "OpenAI Embeddings"

    def build(self, settings):
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": settings.embedding_model or "text-embedding-3-small",
            "openai_api_key": settings.openai_api_key,
        }
        if settings.openai_base_url and settings.openai_base_url != "https://api.openai.com/v1":
            kwargs["openai_api_base"] = settings.openai_base_url
        return OpenAIEmbeddings(**kwargs)


registry.register_embedding(OpenAIEmbeddingPlugin())
