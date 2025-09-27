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

ANAMNESIS_QUESTIONS: List[Tuple[str, str]] = [
    ("sx_ppal", "¿Cuál es tu síntoma principal?"),
    ("inicio", "¿Cómo comenzó el síntoma (repentino o gradual) y en qué momento?"),
    ("duracion", "¿Desde hace cuánto tiempo presentas este síntoma?"),
    ("curso", "¿El síntoma ha mejorado, empeorado o se mantiene igual con el tiempo?"),
    ("intensidad", "En una escala de 1 a 10, ¿qué tan intenso es en promedio?"),
    ("localizacion", "¿Dónde se ubica exactamente?"),
    ("irradiacion", "¿El dolor/sensación se irradia a otra zona? ¿A cuál?"),
    ("factores_agravantes", "¿Qué cosas lo empeoran? (ejercicio, posturas, alimentos, estrés, etc.)"),
    ("factores_aliviantes", "¿Qué cosas lo mejoran? (reposo, fármacos, compresas, etc.)"),
    ("sintomas_asociados", "¿Tienes otros síntomas acompañantes (fiebre, náuseas, tos, disnea, etc.)?"),
    ("antecedentes", "¿Antecedentes médicos relevantes (hipertensión, diabetes, cirugías, etc.)?"),
    ("medicamentos", "¿Estás tomando medicamentos actualmente? Indica dosis/frecuencia si recuerdas."),
    ("alergias", "¿Tienes alergias a medicamentos o alimentos?"),
    ("habitos", "¿Fumas, consumes alcohol u otras sustancias? ¿Con qué frecuencia?"),
    ("red_flags", "¿Has tenido desmayos, dolor torácico intenso, dificultad para respirar o sangrado reciente?"),
]

class QAState(AgentState):
    qa: dict
    remaining_steps: int

def _ids() -> List[str]:
    return [qid for qid, _ in ANAMNESIS_QUESTIONS]

def get_next_question(current_qid: Optional[str]) -> Optional[Tuple[str, str]]:
    ids = _ids()
    if current_qid is None:
        return ANAMNESIS_QUESTIONS[0] if ANAMNESIS_QUESTIONS else None
    try:
        i = ids.index(current_qid)
        return ANAMNESIS_QUESTIONS[i + 1] if i + 1 < len(ids) else None
    except ValueError:
        # Si el current_qid no existe en la lista (corrupción de estado), volver al inicio
        return ANAMNESIS_QUESTIONS[0] if ANAMNESIS_QUESTIONS else None
    
def _ensure_qa(state: Dict[str, Any]) -> Dict[str, Any]:
    qa = state.get("qa") or {}
    qa.setdefault("anamnesis", {})
    qa.setdefault("pending_qid", None)
    qa.setdefault("done", False)
    return qa

def _validate_answer(qid: str, answer: str) -> Tuple[bool, Optional[str]]:
    """Validaciones simples por qid (puedes extenderlas). Devuelve (ok, msg_error)."""
    if qid == "intensidad":
        # intentar extraer número 1-10
        import re
        nums = [int(x) for x in re.findall(r"\d+", answer)]
        if nums:
            n = nums[0]
            if 1 <= n <= 10:
                return True, None
            return False, "Por favor, responde un número entre 1 y 10 (p. ej., 7)."
        return False, "Por favor, responde un número entre 1 y 10 (p. ej., 7)."
    # Agrega otras validaciones si quieres (duración con formato, etc.)
    return True, None

@tool()
def qa_next_question(
    state: Annotated[QAState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Inicia la anamnesis si no hay pregunta pendiente."""
    qa = _ensure_qa(state)
    if qa.get("done"):
        msg = "La anamnesis ya está finalizada."
    elif qa.get("pending_qid") is None:
        nxt = get_next_question(None)
        if nxt:
            qid, question = nxt
            qa["pending_qid"] = qid
            qa["done"] = False
            msg = question
        else:
            qa["pending_qid"] = None
            qa["done"] = True
            msg = "No hay preguntas configuradas. Anamnesis finalizada."
    else:
        # Ya hay una pregunta pendiente; no avanzar automáticamente
        qid = qa["pending_qid"]
        # Recuperar el texto de esa pregunta
        try:
            question = dict(ANAMNESIS_QUESTIONS)[qid]
        except KeyError:
            # fallback si el qid no está
            question = f"Pregunta pendiente: {qid}"
        msg = question
    return Command(update={"qa": qa, "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

@tool()
def qa_submit_answer(
    answer: str,
    state: Annotated[QAState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Registra respuesta de la pregunta pendiente; avanza o marca done."""
    qa = _ensure_qa(state)
    if qa.get("done"):
        return Command(update={
            "qa": qa,
            "messages": [ToolMessage("La anamnesis ya terminó.", tool_call_id=tool_call_id)]
        })

    qid = qa.get("pending_qid")
    if not qid:
        # No hay pregunta pendiente aún: invita a iniciar
        return Command(update={
            "qa": qa,
            "messages": [ToolMessage("Primero iniciemos la anamnesis.", tool_call_id=tool_call_id)]
        })

    # Validación simple por qid
    ok, err = _validate_answer(qid, answer)
    if not ok:
        return Command(update={
            "qa": qa,
            "messages": [ToolMessage(err, tool_call_id=tool_call_id)]
        })

    # Registrar respuesta
    qa["anamnesis"][qid] = answer

    # Calcular siguiente
    nxt = get_next_question(qid)
    if nxt is None:
        qa["pending_qid"] = None
        qa["done"] = True
        msg = f"Registrado: {answer}. La anamnesis ha terminado."
    else:
        next_qid, question = nxt
        qa["pending_qid"] = next_qid
        qa["done"] = False
        msg = f"Registrado: {answer}. {question}"

    return Command(update={"qa": qa, "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

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

# out1 = interview_agent.invoke(
#     {
#         "messages": [HumanMessage(content="Iniciar anamnesis")],
#         "qa": {},
#         "remaining_steps": 10,
#     }
# )

# print("Agente: ", out1["messages"][-1].content)
# print("Estado QA: ", out1["qa"])

# out2 = interview_agent.invoke(
#     {
#         "messages": out1["messages"] + [HumanMessage(content="Tengo dolor en el pecho")],
#         "qa": out1["qa"],
#         "remaining_steps": out1.get("remaining_steps", 5) - 1,
#     }
# )

# print("Agente: ", out2["messages"][-1].content)
# print("Estado QA: ", out2["qa"])

# out3 = interview_agent.invoke(
#     {
#         "messages": out2["messages"] + [HumanMessage(content="Hace 2 días")],
#         "qa": out2["qa"],
#         "remaining_steps": out2.get("remaining_steps", 5) - 1,
#     }
# )

# print("Agente: ", out3["messages"][-1].content)
# print("Estado QA: ", out3["qa"])


SIMULATED_ANSWERS: Dict[str, str] = {
    "sx_ppal": "Tengo dolor en el pecho",
    "inicio": "Comenzó de forma gradual, hace unos días en la noche",
    "duracion": "3 días",
    "curso": "Ha empeorado ligeramente desde que empezó",
    "intensidad": "7",
    "localizacion": "Centro del pecho",
    "irradiacion": "A veces hacia el brazo izquierdo",
    "factores_agravantes": "Esfuerzo físico y subir escaleras",
    "factores_aliviantes": "Reposo y respiración profunda",
    "sintomas_asociados": "Un poco de falta de aire y sudoración",
    "antecedentes": "Hipertensión controlada",
    "medicamentos": "Losartán 50 mg diarios",
    "alergias": "Ninguna conocida",
    "habitos": "No fumo, alcohol social ocasional",
    "red_flags": "No he tenido desmayos ni sangrados",
}

def last_tool_message_text(messages: List[BaseMessage]) -> Optional[str]:
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    return tool_msgs[-1].content if tool_msgs else None

# Estado inicial
state = {
    "messages": [HumanMessage(content="Iniciar anamnesis")],
    "qa": {},
    "remaining_steps": 100,
}

# 1) Arranque: hacer primera pregunta
state = interview_agent.invoke(state)
print("Agente:", last_tool_message_text(state["messages"]))
qid_order = _ids()

# 2) Recorremos todas las preguntas en orden usando las respuestas simuladas
for qid in qid_order:
    # Si el agente todavía no ha puesto este qid como pendiente, deja que repita la última pregunta
    pending = state["qa"].get("pending_qid")
    if pending != qid:
        # Si el flujo se descuadra, mostramos qué espera el agente
        pass

    # Enviar respuesta del usuario para el qid actual
    user_answer = SIMULATED_ANSWERS[qid]
    state = interview_agent.invoke({
        "messages": state["messages"] + [HumanMessage(content=user_answer)],
        "qa": state["qa"],
        "remaining_steps": state["remaining_steps"] - 1,
    })
    print("Usuario:", user_answer)
    print("Agente:", last_tool_message_text(state["messages"]))
    print("Estado QA:", state["qa"])

    # Si terminó, rompe
    if state["qa"].get("done"):
        break

# 3) Resultado final
print("\n=== ANAMNESIS COMPLETA ===")
anam = state["qa"].get("anamnesis", {})
# Mostrar en el orden de ANAMNESIS_QUESTIONS
for qid, label in ANAMNESIS_QUESTIONS:
    if qid in anam:
        print(f"- {label}: {anam[qid]}")
print("\nDone:", state["qa"].get("done"), "| Pending:", state["qa"].get("pending_qid"))