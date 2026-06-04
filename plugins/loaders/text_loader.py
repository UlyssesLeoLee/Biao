from plugins.base import LoaderPlugin
from plugins.registry import registry


class TextLoaderPlugin(LoaderPlugin):
    plugin_name = "text"
    supported_extensions = [".txt", ".md", ".text", ".csv"]

    def extract(self, content: bytes, filename: str) -> str:
        return content.decode("utf-8", errors="ignore")


registry.register_loader(TextLoaderPlugin())
