import threading

_language_tool = None
_language_tool_lock = threading.Lock()


def get_language_tool():
    global _language_tool
    if _language_tool is None:
        with _language_tool_lock:
            if _language_tool is None:
                try:
                    import language_tool_python
                except ModuleNotFoundError:
                    raise ImportError(
                        "language_tool_python is required for grammar_check but is not installed. "
                        "Install it with: pip install language_tool_python"
                    )
                _language_tool = language_tool_python.LanguageTool(
                    "en-US",
                    config={"maxCheckThreads": 1, "maxSpellingSuggestions": 1},
                )
    return _language_tool


def grammar_check(text):
    text = text.strip()

    if text == "":
        return False

    tool = get_language_tool()
    matches = tool.check(text)
    return len(matches) == 0
