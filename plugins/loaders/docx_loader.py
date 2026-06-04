import io

from plugins.base import LoaderPlugin
from plugins.registry import registry


class DocxLoaderPlugin(LoaderPlugin):
    plugin_name = "docx"
    supported_extensions = [".docx", ".doc"]

    def extract(self, content: bytes, filename: str) -> str:
        try:
            from docx import Document
            doc = Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except Exception as e:
            return f"[DOCX解析错误: {e}]"


registry.register_loader(DocxLoaderPlugin())
