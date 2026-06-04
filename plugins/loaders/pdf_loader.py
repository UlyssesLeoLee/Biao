import io

from plugins.base import LoaderPlugin
from plugins.registry import registry


class PdfLoaderPlugin(LoaderPlugin):
    plugin_name = "pdf"
    supported_extensions = [".pdf"]

    def extract(self, content: bytes, filename: str) -> str:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            return "\n\n".join(pages)
        except Exception as e:
            return f"[PDF解析错误: {e}]"


registry.register_loader(PdfLoaderPlugin())
