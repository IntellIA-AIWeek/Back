from langchain_core.tools import InjectedToolCallId, tool


@tool
def rag_tool():
    return "Implement rag tool"