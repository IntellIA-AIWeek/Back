from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from app.utils.state import DeepAgentState
from app.tools.todo_tool import write_todos, read_todos
from app.tools.task_tools import _create_task_tool
from app.tools.eval_consent import eval_consent
from app.utils.prompts import CHAT_PROMPT, SUBAGENT_USAGE_INSTRUCTIONS, TODO_USAGE_INSTRUCTIONS, CONSENT_PROMPT
from datetime import datetime

import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyDp12UiwEopucQa4FzmbtQ59u2luGFg4HI"


model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0,
)

tools = [write_todos, read_todos, eval_consent]

consent_agent = create_react_agent(
    model, tools, prompt=CONSENT_PROMPT, state_schema=DeepAgentState
)

