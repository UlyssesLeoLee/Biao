from graphs.state import SectionResult


def parse_section_results(raw: str, section_names: list, status: str) -> list:
    """Parse LLM output (sections separated by ===) into SectionResult list."""
    results = []
    parts = raw.split("===")
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n")
        name = ""
        content_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]") and not name:
                name = stripped[1:-1]
            else:
                content_lines.append(line)
        if not name and i < len(section_names):
            name = section_names[i]
        content = "\n".join(content_lines).strip()
        if name and content:
            results.append(SectionResult(name=name, content=content, status=status))
    return results
