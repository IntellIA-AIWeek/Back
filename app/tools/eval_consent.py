from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from app.utils.state import DeepAgentState
from typing import Annotated


@tool(description="Evalúa el consentimiento del usuario")
def eval_consent(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    user_response: str,
):
    if user_response.lower() in ["acepto"]:
        return "Consentimiento otorgado"
    else:
        return "Consentimiento no otorgado"


