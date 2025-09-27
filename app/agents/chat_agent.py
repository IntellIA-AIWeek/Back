import os

from dotenv import load_dotenv


load_dotenv(os.path.join("..", ".env"), override=True)

os.environ["GOOGLE_API_KEY"] = "AIzaSyAr-18BxO1xmvvvk2hrao_KQogkrUaNNpM"
os.environ["TAVILY_API_KEY"] = "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"


"""Research Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""
import os
from datetime import datetime
import uuid, base64

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import Annotated, Literal
from app.utils.utils import format_messages
from app.utils.prompts import SUMMARIZE_WEB_SEARCH, CHAT_PROMPT
from app.utils.state import DeepAgentState
from datetime import datetime

from IPython.display import Image, display
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from app.tools.task_tools import _create_task_tool
from app.utils.utils import show_prompt, stream_agent
from app.tools.task_tools import _create_task_tool

from app.utils.prompts import (
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)
from app.utils.todo_tools import write_todos, read_todos

tavily_client = TavilyClient()


class Summary(BaseModel):
    """Schema for webpage content summarization."""

    filename: str = Field(description="Name of the file to store.")
    summary: str = Field(description="Key learnings from the webpage.")


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now()

@tool(description="Evalúa el consentimiento del usuario si es apto para continuar con la conversación")
def eval_consent(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    user_response: str,
):
    if user_response.lower() in ["acepto"]:
        return "Consentimiento otorgado"
    else:
        return "Consentimiento no otorgado"

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?
    - How complex is the question: Have I reached the number of search limits?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"

@tool(description="Evalúa el consentimiento del usuario")
def consent_tool(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    user_response: str,
):
    if user_response.lower() in ["Acepto"]:
        return "Consentimiento otorgado"
    else:
        return "Consentimiento no otorgado"

# Deep Agent Sub-agent delegation tool



# Create agent using create_react_agent directly
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)


# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3

# Tools
sub_agent_tools = [eval_consent, think_tool]
built_in_tools = [write_todos, read_todos, think_tool]

# Create research sub-agent
# consent_agent = {
#     "name": "consent_agent",
#     "description": "Delegate consent to the sub-agent consent. Only give this consent one topic at a time.",
#     "prompt": CONSENT_PROMPT,
#     "tools": ["eval_consent", "think_tool"],
# }

# Create task tool to delegate tasks to sub-agents
# task_tool = _create_task_tool(
#     sub_agent_tools, [], model, DeepAgentState
# )

# delegation_tools = [task_tool]
all_tools = (
    sub_agent_tools + built_in_tools
)  # search available to main agent for trivial cases

# Build prompt
SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTIONS.format(
    max_concurrent_research_units=max_concurrent_research_units,
    max_researcher_iterations=max_researcher_iterations,
    date=datetime.now(),
)

INSTRUCTIONS = (
    "# Chat Agent\n"
    + CHAT_PROMPT
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# TODO MANAGEMENT\n"
    + TODO_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# SUB-AGENT DELEGATION\n"
    + SUBAGENT_INSTRUCTIONS
)


# Create agent
chat_agent = create_react_agent(
    model, all_tools, prompt=INSTRUCTIONS, state_schema=DeepAgentState
)


result = chat_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hola, tengo una pregunta medica ",
            },
        ],
    }
)

format_messages(result["messages"])
