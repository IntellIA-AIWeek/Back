import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple, Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, BaseMessage, HumanMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field

load_dotenv(os.path.join("..", ".env"), override=True)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAr-18BxO1xmvvvk2hrao_KQogkrUaNNpM"

QA_SYSTEM_PROMPT = """\
# QA Sub-Agent (Anamnesis)
Eres un sub-agente delegado. Tu tarea es GUÍAR la anamnesis sin diagnosticar.
- Usa únicamente las tools `qa_next_question` y `qa_submit_answer`.
- Haz una pregunta a la vez; espera respuesta antes de continuar.
- Si ya hay una respuesta registrada en el estado (`qa["anamnesis"]`), no repitas la misma pregunta.
- Usa `qa_submit_answer` solo si `qa["pending_qid"]` tiene valor.
- Cuando `qa["pending_qid"]` es None, espera la siguiente instrucción.
"""

ANAMNESIS_QUESTIONS = [
    ("sx_ppal", "¿Cuál es tu síntoma principal?"),
    ("duracion", "¿Desde hace cuánto tiempo tienes este síntoma?"),
    ("intensidad", "¿Qué tan intenso es el dolor en una escala del 1 al 10?"),
    ("localizacion", "¿En qué parte del cuerpo se localiza el síntoma?"),
    ("antecedentes", "¿Tienes antecedentes médicos relevantes?"),
]

class QAState(AgentState):
    qa: dict


def get_next_question(current_qid: str) -> tuple[str, str] | None:
    """Dado el id actual, devuelve la siguiente pregunta."""
    ids = [qid for qid, _ in ANAMNESIS_QUESTIONS]
    if current_qid is None:
        return ANAMNESIS_QUESTIONS[0]
    try:
        idx = ids.index(current_qid)
        return ANAMNESIS_QUESTIONS[idx + 1] if idx + 1 < len(ids) else None
    except ValueError:
        return None

@tool()
def qa_next_question(
    state: Annotated[QAState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Iniciar o continuar la anamnesis con la siguiente pregunta."""
    qa = state.get("qa", {"anamnesis": {}, "pending_qid": None})
    if qa.get("pending_qid") is None:
        next_q = get_next_question(None)
        if next_q:
            qid, question = next_q
            qa["pending_qid"] = qid
            msg = question
        else:
            qa["pending_qid"] = None
            msg = "La anamnesis ha finalizado. Gracias por tus respuestas."
    else:
        msg = f"Ya hay una pregunta pendiente: {qa['pending_qid']}"
    return Command(update={"qa": qa, "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

@tool()
def qa_submit_answer(
    answer: str,
    state: Annotated[QAState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Registrar respuesta y preparar la siguiente pregunta."""
    qa = state.get("qa", {"anamnesis": {}, "pending_qid": None})
    if "anamnesis" not in qa:
        qa["anamnesis"] = {} 
    qid = qa.get("pending_qid")
    if qid:
        qa["anamnesis"][qid] = answer
    # preparar siguiente pregunta
    next_q = get_next_question(qid)
    if next_q is None:
        qa["pending_qid"] = None
        msg = f"Registrado: {answer}. La anamnesis ha terminado."
    else:
        next_qid, question = next_q
        qa["pending_qid"] = next_qid
        msg = f"Registrado: {answer}. {question}"
    return Command(update={
        "qa": qa,
        "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]
    })

model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.0,
)

interview_agent = create_react_agent(
    model,
    tools=[qa_next_question, qa_submit_answer],
    prompt=QA_SYSTEM_PROMPT,
    state_schema = QAState,
).with_config({"recursion_limit": 20})

out1 = interview_agent.invoke(
    {
        "messages": [HumanMessage(content="Iniciar anamnesis")],
        "qa": {},
        "remaining_steps": 10,
    }
)

print("Agente: ", out1["messages"][-1].content)
print("Estado QA: ", out1["qa"])

out2 = interview_agent.invoke(
    {
        "messages": out1["messages"] + [HumanMessage(content="Tengo dolor en el pecho")],
        "qa": out1["qa"],
        "remaining_steps": out1.get("remaining_steps", 5) - 1,
    }
)

print("Agente: ", out2["messages"][-1].content)
print("Estado QA: ", out2["qa"])

out3 = interview_agent.invoke(
    {
        "messages": out2["messages"] + [HumanMessage(content="Hace 2 días")],
        "qa": out2["qa"],
        "remaining_steps": out2.get("remaining_steps", 5) - 1,
    }
)

print("Agente: ", out3["messages"][-1].content)
print("Estado QA: ", out3["qa"])