import re
import unicodedata


def format_document_preview(text: str, max_chars: int = 500) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def sanitize_filename(filename: str) -> str:
    filename = unicodedata.normalize("NFKD", filename)
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    filename = filename.strip(". ")
    return filename or "untitled"


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[已截断，共{len(text)}字]"


def word_count(text: str) -> int:
    cn_chars = len(re.findall(r'[一-鿿]', text))
    en_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    return cn_chars + en_words
