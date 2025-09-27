import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId # CAMBIO: Importar InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import TypedDict

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv(os.path.join("..", ".env"), override=True)
# Asegúrate de que tu clave esté en un archivo .env o reemplázala aquí
os.environ["GOOGLE_API_KEY"] = "AIzaSyAPDNhsimSdZEb5A8hs9HVC7MhUXfBzkig"

# --- 2. DEFINICIÓN DE ESTADO ---
class QAState(TypedDict):
    anamnesis: Dict[str, str]
    pending_qid: Optional[str]
    done: bool

class CombinedAgentState(AgentState):
    qa: QAState

# --- 3. LÓGICA DE LA ENTREVISTA ---
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
    ("sintomas_asociados", "¿Tienes otros síntomas acompañantes (fiebre, náuseas, tos, etc.)?"),
    ("antecedentes", "¿Antecedentes médicos relevantes (hipertensión, diabetes, cirugías, etc.)?"),
    ("medicamentos", "¿Estás tomando medicamentos actualmente?"),
    ("alergias", "¿Tienes alergias a medicamentos o alimentos?"),
    ("habitos", "¿Fumas, consumes alcohol u otras sustancias?"),
    ("red_flags", "¿Has tenido desmayos, dolor torácico intenso o dificultad para respirar?"),
]

# --- 4. FUNCIONES AUXILIARES ---
def _ids() -> List[str]:
    return [qid for qid, _ in ANAMNESIS_QUESTIONS]

def get_next_question(current_qid: Optional[str]) -> Optional[Tuple[str, str]]:
    ids = _ids()
    if current_qid is None: return ANAMNESIS_QUESTIONS[0]
    try:
        i = ids.index(current_qid)
        return ANAMNESIS_QUESTIONS[i + 1] if i + 1 < len(ids) else None
    except ValueError: return ANAMNESIS_QUESTIONS[0]

def _ensure_qa(state: Dict[str, Any]) -> Dict[str, Any]:
    qa = state.get("qa") or {}
    qa.setdefault("anamnesis", {})
    qa.setdefault("pending_qid", None)
    qa.setdefault("done", False)
    return qa

# --- 5. DEFINICIÓN DE HERRAMIENTAS (CORREGIDAS) ---

@tool
def eval_consent(user_response: str) -> str:
    """Evalúa la respuesta del usuario para otorgar o negar el consentimiento para la entrevista médica."""
    if user_response.lower().strip() in ["acepto", "si", "de acuerdo", "ok", "sí"]:
        return "Consentimiento otorgado. El agente ahora debe iniciar la entrevista llamando a la herramienta `qa_next_question`."
    else:
        return "Consentimiento no otorgado. El agente debe informar al usuario que no puede continuar."

@tool
def qa_next_question(
    state: Annotated[CombinedAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] # CAMBIO: Se inyecta el ID de la llamada
) -> Command:
    """Inicia la anamnesis si no hay una pregunta pendiente."""
    qa = _ensure_qa(state)
    if qa.get("done"):
        msg = "La anamnesis ya está finalizada."
    elif qa.get("pending_qid") is None:
        nxt = get_next_question(None)
        qid, question = nxt
        qa["pending_qid"] = qid
        msg = question
    else:
        qid = qa["pending_qid"]
        question = dict(ANAMNESIS_QUESTIONS)[qid]
        msg = question
    # CAMBIO: Se usa el ID dinámico en lugar de 'static_tool_call'
    return Command(update={"qa": qa, "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

@tool
def qa_submit_answer(
    answer: str,
    state: Annotated[CombinedAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId] # CAMBIO: Se inyecta el ID de la llamada
) -> Command:
    """Registra la respuesta a la pregunta pendiente y avanza a la siguiente."""
    qa = _ensure_qa(state)
    qid = qa.get("pending_qid")
    if not qid:
        # CAMBIO: Se usa el ID dinámico
        return Command(update={"qa": qa, "messages": [ToolMessage("No hay ninguna pregunta pendiente. Inicia la entrevista primero.", tool_call_id=tool_call_id)]})

    qa["anamnesis"][qid] = answer

    nxt = get_next_question(qid)
    if nxt is None:
        qa["pending_qid"] = None
        qa["done"] = True
        msg = f"Registrado: {answer}. La anamnesis ha terminado. Ahora resume la información recopilada."
    else:
        next_qid, question = nxt
        qa["pending_qid"] = next_qid
        msg = f"Registrado: {answer}. Siguiente pregunta: {question}"
    
    # CAMBIO: Se usa el ID dinámico
    return Command(update={"qa": qa, "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

@tool
def summarize_interview(state: Annotated[CombinedAgentState, InjectedState]) -> str:
    """Genera un resumen final de la anamnesis una vez que ha concluido."""
    qa = _ensure_qa(state)
    if not qa.get("done"):
        return "No puedo generar un resumen porque la entrevista aún no ha terminado."

    summary_parts = ["Resumen de la anamnesis:"]
    for qid, question_text in ANAMNESIS_QUESTIONS:
        if qid in qa["anamnesis"]:
            summary_parts.append(f"- {question_text}: {qa['anamnesis'][qid]}")
    
    return "\n".join(summary_parts)

# --- 6. CONFIGURACIÓN DEL AGENTE HÍBRIDO ---
all_tools = [eval_consent, qa_next_question, qa_submit_answer, summarize_interview]

MEDICAL_INTERVIEW_PROTOCOL = f"""
# PROTOCOLO DE ENTREVISTA MÉDICA
Hoy es {datetime.now().strftime('%Y-%m-%d')}.

Si el usuario indica que tiene una pregunta o problema médico, DEBES seguir estos pasos estrictamente:

1.  **Solicitar Consentimiento**: Responde ÚNICAMENTE con: "Para poder ayudarte, necesito hacerte algunas preguntas sobre tus síntomas. Esta conversación no reemplaza un diagnóstico médico profesional. ¿Aceptas continuar?"

2.  **Evaluar Respuesta**: Una vez que el usuario responda, usa la herramienta `eval_consent`.

3.  **Iniciar Entrevista**: Si el consentimiento es otorgado, DEBES iniciar la anamnesis llamando a la herramienta `qa_next_question`. Esta te dará la primera pregunta.

4.  **Continuar Entrevista**: Después de cada respuesta del usuario, DEBES llamar a la herramienta `qa_submit_answer` con la respuesta que te dieron. La herramienta registrará la respuesta y te dará la siguiente pregunta.

5.  **Finalizar Entrevista**: La herramienta `qa_submit_answer` te informará cuando la entrevista haya terminado. En ese momento, DEBES llamar a la herramienta `summarize_interview` para generar y presentar el resumen final al usuario.
"""

model = init_chat_model(
    "gemini-2.0-flash",
    model_provider="google_genai",
    temperature=0,
)

agent = create_react_agent(
    model,
    tools=all_tools,
    prompt=MEDICAL_INTERVIEW_PROTOCOL,
    state_schema=CombinedAgentState
)

# --- 7. EJECUCIÓN ---
def run_turn(agent_executor, state: dict, user_input: str = ""):
    if user_input:
        state["messages"].append(HumanMessage(content=user_input))
        print(f"👤 Usuario: {user_input}")

    new_state = agent_executor.invoke(state)
    
    agent_response = new_state["messages"][-1].content
    print(f"🤖 Agente: {agent_response}\n")
    return new_state

if __name__ == "__main__":
    # Inicializa el estado, que se mantendrá durante toda la conversación
    current_state = {
        "messages": [],
        "qa": {"anamnesis": {}, "pending_qid": None, "done": False}
    }
    
    print("--- Asistente Médico Interactivo ---")
    print("Inicia la conversación con tu consulta médica (o escribe 'salir' para terminar).\n")
    
    # Bucle principal de la conversación
    while True:
        # Pide la entrada del usuario desde la consola
        user_input = input("👤 Usuario: ")
        
        if user_input.lower() == 'salir':
            print("🤖 Agente: Adiós.")
            break

        # Cada llamada a run_turn actualiza el estado, guardando la información
        current_state = run_turn(agent, current_state, user_input)

        # Si el agente marca la entrevista como finalizada, le damos un último turno 
        # para que pueda generar el resumen y luego rompemos el bucle.
        if current_state["qa"].get("done"):
            print("--- ENTREVISTA FINALIZADA, GENERANDO RESUMEN ---\n")
            current_state = run_turn(agent, current_state)
            break
            
    # --- VERIFICACIÓN FINAL ---
    # Al final de la conversación, imprimimos el estado para demostrar que todo se guardó.
    print("="*50)
    print("✅ VERIFICACIÓN DEL ESTADO FINAL GUARDADO")
    print("="*50)
    
    import json
    final_anamnesis_data = current_state.get("qa", {}).get("anamnesis", {})
    
    if final_anamnesis_data:
        print("¡Éxito! La información de tu conversación fue guardada:")
        print(json.dumps(final_anamnesis_data, indent=2, ensure_ascii=False))
    else:
        print("La conversación terminó antes de que se guardara información de la anamnesis.")
    