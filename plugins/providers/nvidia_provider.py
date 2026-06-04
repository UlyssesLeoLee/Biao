"""
NVIDIA NIM LLM plugin.

NVIDIA NIM exposes an OpenAI-compatible REST API, so we reuse ChatOpenAI
with a custom base_url. No additional SDK dependencies required.

Free models at https://build.nvidia.com (register for nvapi- key):
  meta/llama-3.1-70b-instruct
  meta/llama-3.1-8b-instruct
  nvidia/llama-3.1-nemotron-70b-instruct
  mistralai/mixtral-8x22b-instruct-v0.1
"""
from plugins.base import LLMPlugin
from plugins.registry import registry

_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaProvider(LLMPlugin):
    plugin_name = "nvidia"
    display_name = "NVIDIA NIM (免费额度)"

    def build(self, settings):
        from langchain_openai import ChatOpenAI
        base_url = getattr(settings, "nvidia_base_url", _NVIDIA_BASE_URL) or _NVIDIA_BASE_URL
        return ChatOpenAI(
            model=settings.nvidia_model,
            openai_api_key=settings.nvidia_api_key,
            openai_api_base=base_url,
            temperature=0.3,
        )

    def get_api_key(self, settings) -> str:
        return getattr(settings, "nvidia_api_key", "")


registry.register_llm(NvidiaProvider())
