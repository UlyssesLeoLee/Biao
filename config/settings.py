import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

SETTINGS_FILE = Path("./user_settings.json")


class Settings(BaseModel):
    api_provider: str = Field(default="openai", description="API提供商: openai | anthropic | nvidia")
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="OpenAI API Base URL")
    openai_model: str = Field(default="gpt-4o", description="OpenAI 模型名称")
    anthropic_api_key: str = Field(default="", description="Anthropic API Key")
    anthropic_model: str = Field(default="claude-opus-4-8", description="Anthropic 模型名称")
    nvidia_api_key: str = Field(default="", description="NVIDIA NIM API Key (nvapi-...)")
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", description="NVIDIA NIM Base URL")
    nvidia_model: str = Field(default="meta/llama-3.1-70b-instruct", description="NVIDIA 模型名称")
    embedding_provider: str = Field(default="openai", description="Embedding提供商: openai | huggingface | nvidia")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding模型")
    chroma_persist_dir: str = Field(default="./chroma_db", description="ChromaDB持久化目录")
    chroma_collection_name: str = Field(default="bidding_documents", description="ChromaDB集合名称")
    chunk_size: int = Field(default=800, description="文档分块大小")
    chunk_overlap: int = Field(default=150, description="文档分块重叠")
    retrieval_top_k: int = Field(default=5, description="检索返回数量")

    @classmethod
    def load(cls) -> "Settings":
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(**data)
            except Exception:
                pass
        return cls(
            api_provider=os.getenv("API_PROVIDER", "openai"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8"),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
            nvidia_base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            nvidia_model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "bidding_documents"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
        )

    def save(self) -> None:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=2)

    def get_llm_api_key(self) -> str:
        if self.api_provider == "anthropic":
            return self.anthropic_api_key
        if self.api_provider == "nvidia":
            return self.nvidia_api_key
        return self.openai_api_key

    def is_configured(self) -> bool:
        if self.api_provider == "anthropic":
            return bool(self.anthropic_api_key)
        if self.api_provider == "nvidia":
            return bool(self.nvidia_api_key)
        return bool(self.openai_api_key)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings.load()
    return _settings
