from plugins.base import LLMPlugin
from plugins.registry import registry


class AnthropicProvider(LLMPlugin):
    plugin_name = "anthropic"
    display_name = "Anthropic (Claude 系列)"

    def build(self, settings):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.anthropic_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.3,
        )

    def get_api_key(self, settings) -> str:
        return settings.anthropic_api_key


registry.register_llm(AnthropicProvider())
