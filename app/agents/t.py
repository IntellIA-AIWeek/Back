from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from app.utils.state import DeepAgentState
from app.tools.todo_tool import write_todos, read_todos
from app.tools.task_tools import _create_task_tool
# from app.tools.rag_tool import rag_tool
# from app.tools.format_response_tool import format_response_tool
from app.utils.prompts import CHAT_PROMPT, SUBAGENT_USAGE_INSTRUCTIONS, TODO_USAGE_INSTRUCTIONS, CONSENT_PROMPT
from datetime import datetime
from app.utils.utils import format_messages
from app.agents.consent_agent import consent_agent
import os
from app.tools.eval_consent import eval_consent

os.environ["GOOGLE_API_KEY"] = "AIzaSyDp12UiwEopucQa4FzmbtQ59u2luGFg4HI"


model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0,
)

consent_agent = {
    "name": "consent_agent",
    "description": "Delegate consent to the sub-agent consent. Only give this consent one topic at a time.",
    "prompt": CONSENT_PROMPT,
    "tools": ["eval_consent"],
}

sub_agent_tools = [eval_consent]
task_tool = _create_task_tool(
    sub_agent_tools, [consent_agent], model, DeepAgentState
)
delegation_tools = [task_tool]

all_tools = (
    delegation_tools
) 

# Prompt específico de este sub-agente
chat_prompt = CHAT_PROMPT

max_concurrent_research_units = 3
max_researcher_iterations = 3   

#Instrucciones
SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTIONS.format(
    max_concurrent_research_units=max_concurrent_research_units,
    max_researcher_iterations=max_researcher_iterations,
    date=datetime.now(),
)

INSTRUCTIONS = (
    "#Chat Agent Instructions\n"
    + chat_prompt
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



# Crear el sub-agente de investigación
chat_agent = create_react_agent(
    model, all_tools, prompt=INSTRUCTIONS, state_schema=DeepAgentState
)




result = chat_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hola",
            }
        ],

    }
)
format_messages(result["messages"])