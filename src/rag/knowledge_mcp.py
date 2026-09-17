from pathlib import Path

from fastmcp import FastMCP

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

mcp = FastMCP(name="Knowledge Search")


@mcp.tool
def scan_knowledge(
    query: str,
    case_sensitive: bool = False,
    max_results: int = 50,
    context_chars: int = 160,
) -> list[dict[str, int | str]]:
    """Search every markdown file in the knowledge folder for plain text."""
    if not query or max_results <= 0:
        return []

    context_chars = max(context_chars, 0)
    needle = query if case_sensitive else query.casefold()
    results = []

    for file in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        text = file.read_text(encoding="utf-8")
        haystack = text if case_sensitive else text.casefold()
        start = 0

        while (index := haystack.find(needle, start)) != -1:
            line = text.count("\n", 0, index) + 1
            left = max(0, index - context_chars)
            right = min(len(text), index + len(query) + context_chars)
            results.append(
                {
                    "source": file.relative_to(KNOWLEDGE_DIR).as_posix(),
                    "line": line,
                    "offset": index,
                    "snippet": text[left:right].replace("\n", " ").strip(),
                }
            )

            if len(results) >= max_results:
                return results

            start = index + max(len(query), 1)

    print(results)
    return results


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8000)
