from plugins.base import EmbeddingPlugin
from plugins.registry import registry

_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class HuggingFaceEmbeddingPlugin(EmbeddingPlugin):
    plugin_name = "huggingface"
    display_name = "HuggingFace（本地，无需 API）"

    def build(self, settings):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model or _DEFAULT_MODEL
        )


registry.register_embedding(HuggingFaceEmbeddingPlugin())
