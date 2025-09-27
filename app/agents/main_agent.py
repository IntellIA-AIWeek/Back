import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import TypedDict

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv(os.path.join("..", ".env"), override=True)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or "AIzaSyAPDNhsimSdZEb5A8hs9HVC7MhUXfBzkig"

# --- 2. DEFINICIÓN DE ESTADO ---
class QAState(TypedDict):
    anamnesis: Dict[str, str]
    pending_qid: Optional[str]
    done: bool
    # Campo opcional para almacenar lo detectado por el symptom agent
    symptoms: List[str]

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
    if current_qid is None:
        return ANAMNESIS_QUESTIONS[0]
    try:
        i = ids.index(current_qid)
        return ANAMNESIS_QUESTIONS[i + 1] if i + 1 < len(ids) else None
    except ValueError:
        return ANAMNESIS_QUESTIONS[0]

def _ensure_qa(state: Dict[str, Any]) -> Dict[str, Any]:
    qa = state.get("qa") or {}
    qa.setdefault("anamnesis", {})
    qa.setdefault("pending_qid", None)
    qa.setdefault("done", False)
    qa.setdefault("symptoms", [])
    return qa

# --- 5. TOOLS DE ENTREVISTA ---
@tool
def eval_consent(user_response: str) -> str:
    """Evalúa la respuesta del usuario para otorgar o negar el consentimiento para la entrevista médica."""
    if user_response.lower().strip() in ["acepto", "si", "sí", "de acuerdo", "ok"]:
        return "Consentimiento otorgado. El agente ahora debe iniciar la entrevista llamando a la herramienta `qa_next_question`."
    else:
        return "Consentimiento no otorgado. El agente debe informar al usuario que no puede continuar."

@tool
def qa_next_question(
    state: Annotated[CombinedAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
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
    return Command(update={"qa": qa, "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

@tool
def qa_submit_answer(
    answer: str,
    state: Annotated[CombinedAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Registra la respuesta a la pregunta pendiente y avanza a la siguiente."""
    qa = _ensure_qa(state)
    qid = qa.get("pending_qid")
    if not qid:
        return Command(update={"qa": qa, "messages": [ToolMessage("No hay ninguna pregunta pendiente. Inicia la entrevista primero.", tool_call_id=tool_call_id)]})

    qa["anamnesis"][qid] = answer

    nxt = get_next_question(qid)
    if nxt is None:
        qa["pending_qid"] = None
        qa["done"] = True
        msg = f"Registrado: {answer}. La anamnesis ha terminado. Resume y extrae síntomas ahora."
    else:
        next_qid, question = nxt
        qa["pending_qid"] = next_qid
        msg = f"Registrado: {answer}. Siguiente pregunta: {question}"
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

# --- 6. SYMPTOM AGENT ---
SYMPTOM_ORDER = ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload.1','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

import re, json

def _safe_json_loads(text: str) -> Any:
    m = re.search(r'\{.*\}', text, flags=re.S)
    candidate = m.group(0) if m else text
    try:
        return json.loads(candidate)
    except Exception:
        return {"symptoms": []}

def _make_llm_symptoms():
    return init_chat_model(
        "gemini-2.0-flash",
        model_provider="google_genai",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )

def _symptom_prompt(anamnesis_text: str) -> str:
    vocab = "\n".join(SYMPTOM_ORDER)
    return f"""
Eres un asistente clínico. Tienes un texto de anamnesis del paciente (respuestas libres).
Tu tarea es identificar ÚNICAMENTE los síntomas presentes mapeando sinónimos al siguiente vocabulario canónico (lista cerrada):

VOCABULARIO (uno por línea):
{vocab}

INSTRUCCIONES:
- Devuelve SOLO un JSON con la forma:
  {{"symptoms": ["<symptom_1>", "<symptom_2>", ...]}}
- Cada elemento DEBE ser exactamente uno de los términos del vocabulario (case-sensitive tal cual).
- Si no hay síntomas, devuelve {{"symptoms": []}}.

ANAMNESIS:
\"\"\"{anamnesis_text.strip()}\"\"\"
    """.strip()

def _anamnesis_dict_to_text(anam: Dict[str, str]) -> str:
    ordered: List[str] = []
    qmap = dict(ANAMNESIS_QUESTIONS)
    for qid, _ in ANAMNESIS_QUESTIONS:
        if qid in anam:
            ordered.append(f"{qid}: {anam[qid]}")
    for k, v in anam.items():
        if k not in qmap:
            ordered.append(f"{k}: {v}")
    return " ".join(ordered)

def _extract_symptoms_from_qa(qa: Dict[str, Any]) -> List[str]:
    anam_data = qa.get("anamnesis") or ""
    if isinstance(anam_data, dict):
        anam_text = _anamnesis_dict_to_text(anam_data)
    else:
        anam_text = str(anam_data)

    llm = _make_llm_symptoms()
    resp = llm.invoke(_symptom_prompt(anam_text))
    data = _safe_json_loads(getattr(resp, "content", "") or str(resp))
    raw_list = data.get("symptoms", [])
    vocab_set = set(SYMPTOM_ORDER)
    return [s for s in raw_list if isinstance(s, str) and s in vocab_set]

@tool
def symptom_extractor(qa_json: str) -> Dict[str, Any]:
    """Extrae síntomas canónicos desde un JSON de QA (string). Retorna { "symptoms": [<str>...] }."""
    try:
        qa = json.loads(qa_json)
    except Exception:
        return {"symptoms": [], "error": "qa_json inválido"}
    return {"symptoms": _extract_symptoms_from_qa(qa)}

@tool
def symptoms_from_state(
    state: Annotated[CombinedAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Lee qa del estado, extrae síntomas canónicos y los guarda en qa['symptoms']."""
    qa = _ensure_qa(state)
    symptoms = _extract_symptoms_from_qa(qa)
    qa["symptoms"] = symptoms
    msg = f"Síntomas detectados: {symptoms}" if symptoms else "No se detectaron síntomas."
    return Command(update={"qa": qa, "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})

# --- 7. CONFIGURACIÓN DEL AGENTE HÍBRIDO ---
all_tools = [
    eval_consent,
    qa_next_question,
    qa_submit_answer,
    summarize_interview,
    symptom_extractor,
    symptoms_from_state,
]

MEDICAL_INTERVIEW_PROTOCOL = f"""
# PROTOCOLO DE ENTREVISTA MÉDICA
Hoy es {datetime.now().strftime('%Y-%m-%d')}.

Si el usuario indica que tiene una pregunta o problema médico, DEBES seguir estos pasos estrictamente:

1.  **Solicitar Consentimiento**: Responde ÚNICAMENTE con:
    "Para poder ayudarte, necesito hacerte algunas preguntas sobre tus síntomas.
     Esta conversación no reemplaza un diagnóstico médico profesional. ¿Aceptas continuar?"

2.  **Evaluar Respuesta**: Una vez que el usuario responda, usa la herramienta `eval_consent`.

3.  **Iniciar Entrevista**: Si el consentimiento es otorgado, DEBES iniciar la anamnesis llamando a `qa_next_question`.

4.  **Continuar Entrevista**: Después de cada respuesta del usuario, DEBES llamar a `qa_submit_answer(answer=...)`.
    Esta herramienta registrará la respuesta y te dará la siguiente pregunta.

5.  **Finalizar Entrevista**: Cuando `qa_submit_answer` indique que la entrevista terminó (`qa.done = True`),
    DEBES llamar a `summarize_interview` para presentar un resumen claro de lo recopilado.

6.  **Extraer Síntomas**: Tras el resumen, DEBES llamar a `symptoms_from_state` para extraer una lista de síntomas canónicos
    (vocabulario cerrado) a partir de `qa["anamnesis"]`. La lista quedará en `qa["symptoms"]` y se mostrará al usuario.
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

# --- 8. EJECUCIÓN (CLI de prueba opcional) ---
def run_turn(agent_executor, state: dict, user_input: str = ""):
    if user_input:
        state["messages"].append(HumanMessage(content=user_input))
        print(f"👤 Usuario: {user_input}")

    new_state = agent_executor.invoke(state)
    agent_response = new_state["messages"][-1].content
    print(f"🤖 Agente: {agent_response}\n")
    return new_state

if __name__ == "__main__":
    current_state = {
        "messages": [],
        "qa": {"anamnesis": {}, "pending_qid": None, "done": False, "symptoms": []}
    }
    print("--- Asistente Médico Interactivo ---")
    print("Inicia la conversación con tu consulta médica (o escribe 'salir' para terminar).\n")
    while True:
        user_input = input("👤 Usuario: ")
        if user_input.lower() == 'salir':
            print("🤖 Agente: Adiós.")
            break
        current_state = run_turn(agent, current_state, user_input)
        if current_state["qa"].get("done"):
            print("--- ENTREVISTA FINALIZADA: RESUMEN Y EXTRACCIÓN DE SÍNTOMAS ---\n")
            # Un turno adicional para que el agente, siguiendo el protocolo, invoque summarize_interview y symptoms_from_state
            current_state = run_turn(agent, current_state)
            break

    # --- VERIFICACIÓN FINAL ---
    print("="*50)
    print("✅ VERIFICACIÓN DEL ESTADO FINAL GUARDADO")
    print("="*50)
    import json
    print(json.dumps(current_state.get("qa", {}), indent=2, ensure_ascii=False))
