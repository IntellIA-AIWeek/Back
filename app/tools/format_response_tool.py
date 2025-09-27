from langchain_core.tools import InjectedToolCallId, tool

@tool
def format_response_tool():
    return "Implement format response tool"