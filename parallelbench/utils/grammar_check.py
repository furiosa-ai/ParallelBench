

language_tool = None


def get_language_tool():
    global language_tool
    if language_tool is None:
        import language_tool_python
        language_tool = language_tool_python.LanguageTool('en-US', config={"maxCheckThreads": 1, "maxSpellingSuggestions": 1})
    return language_tool


def grammar_check(text):
    text = text.strip()

    if text == "":
        return False
    
    tool = get_language_tool()
    matches = tool.check(text)
    return len(matches) == 0