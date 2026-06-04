"""
NVIDIA NIM Embedding plugin.

Uses the OpenAI-compatible embeddings endpoint at NVIDIA NIM.
Default model: nvidia/nv-embedqa-e5-v5 (free tier available).
Other options: nvidia/nv-embed-v1, nvidia/nv-embedqa-mistral-7b-v2
"""
from plugins.base import EmbeddingPlugin
from plugins.registry import registry

_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
_DEFAULT_MODEL = "nvidia/nv-embedqa-e5-v5"


class NvidiaEmbeddingPlugin(EmbeddingPlugin):
    plugin_name = "nvidia"
    display_name = "NVIDIA NIM Embeddings"

    def build(self, settings):
        from langchain_openai import OpenAIEmbeddings
        nvidia_key = getattr(settings, "nvidia_api_key", "")
        base_url = getattr(settings, "nvidia_base_url", _NVIDIA_BASE_URL) or _NVIDIA_BASE_URL
        return OpenAIEmbeddings(
            model=settings.embedding_model or _DEFAULT_MODEL,
            openai_api_key=nvidia_key,
            openai_api_base=base_url,
        )


registry.register_embedding(NvidiaEmbeddingPlugin())
