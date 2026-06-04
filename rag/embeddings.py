from typing import Any

from config.settings import Settings


def get_embeddings(settings: Settings) -> Any:
    from plugins.registry import registry
    return registry.get_embedding(settings.embedding_provider).build(settings)
