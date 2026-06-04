from typing import Any

from config.settings import Settings


def get_embeddings(settings: Settings) -> Any:
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": settings.embedding_model,
            "openai_api_key": settings.openai_api_key,
        }
        if settings.openai_base_url and settings.openai_base_url != "https://api.openai.com/v1":
            kwargs["openai_api_base"] = settings.openai_base_url
        return OpenAIEmbeddings(**kwargs)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
