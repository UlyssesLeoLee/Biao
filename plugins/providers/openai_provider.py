from plugins.base import LLMPlugin
from plugins.registry import registry


class OpenAIProvider(LLMPlugin):
    plugin_name = "openai"
    display_name = "OpenAI (GPT 系列)"

    def build(self, settings):
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": settings.openai_model,
            "openai_api_key": settings.openai_api_key,
            "temperature": 0.3,
        }
        if settings.openai_base_url and settings.openai_base_url != "https://api.openai.com/v1":
            kwargs["openai_api_base"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)

    def get_api_key(self, settings) -> str:
        return settings.openai_api_key


registry.register_llm(OpenAIProvider())
