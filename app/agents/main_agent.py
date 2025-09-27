import os
# from datetime import datetime
# from typing import Dict, Any, List, Optional, Tuple, Annotated

from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model
# from langchain_core.messages import HumanMessage, ToolMessage
# from langchain_core.tools import tool, InjectedToolCallId
# from langgraph.prebuilt import create_react_agent, InjectedState
# from langgraph.types import Command
# from langgraph.prebuilt.chat_agent_executor import AgentState
# from typing_extensions import TypedDict

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv(os.path.join("..", ".env"), override=True)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or "AIzaSyAPDNhsimSdZEb5A8hs9HVC7MhUXfBzkig"
# medical_agent_bedrock.py
import os
import re
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Annotated
import httpx

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import TypedDict
import asyncio


# ============= 0) CONFIG =============
# Credenciales AWS deben estar en entorno (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION)
BEDROCK_REGION = "us-west-2"
BEDROCK_MODEL_MAIN = "anthropic.claude-3-sonnet-20240229-v1:0"

# ============= Helpers impresión =============
def to_text(msg: BaseMessage) -> str:
    """Convierte un mensaje LangChain a texto imprimible (maneja respuestas en partes)."""
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        # Claude/Bedrock puede retornar [{"type":"text","text":"..."}]
        parts = []
        for p in c:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(c)

def last_nonempty_text(messages: List[BaseMessage]) -> str:
    """Busca hacia atrás el último mensaje con texto no vacío."""
    for m in reversed(messages):
        t = to_text(m).strip()
        if t:
            return t
    return ""

# ============= 1) ESTADO =============
class QAState(TypedDict, total=False):
    anamnesis: Dict[str, str]
    pending_qid: Optional[str]
    done: bool
    symptoms: List[str]
    structured: Dict[str, Any]

class CombinedAgentState(AgentState):
    qa: QAState

# ============= 2) PREGUNTAS =============
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
    ("antecedentes", "¿Antecedentes médicos relevantes (hipertensión, diabetes, cirugías, etc.)?"),
    ("medicamentos", "¿Estás tomando medicamentos actualmente?"),
    ("alergias", "¿Tienes alergias a medicamentos o alimentos?"),
    ("habitos", "¿Fumas, consumes alcohol u otras sustancias?"),
    ("red_flags", "¿Has tenido desmayos, dolor torácico intenso o dificultad para respirar?"),
    ("sintomas_asociados", "¿Tienes otros síntomas acompañantes (fiebre, náuseas, tos, etc.)?"),
]

# ============= 3) UTILIDADES =============
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
    qa.setdefault("structured", {})
    return qa

# ============= 4) TOOLS ENTREVISTA =============
@tool
def eval_consent(user_response: str) -> str:
    """Evalúa la respuesta del usuario para otorgar o negar el consentimiento para la entrevista médica."""
    if user_response.lower().strip() in {"acepto", "si", "sí", "de acuerdo", "ok"}:
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

import re

# --- Heurísticas sencillas para enrutar respuestas ---
def _classify_answer(answer: str) -> Optional[str]:
    """
    Dada una respuesta corta, intenta detectar a cuál QID encaja mejor.
    Devuelve el qid sugerido o None si no detecta.
    """
    a = answer.lower().strip()

    # Intensidad 1-10
    if re.fullmatch(r"(10|[1-9])(\.0)?", a):
        return "intensidad"

    # Duración (horas/días/semanas/meses)
    if re.search(r"\b(\d+|\bun[oa]s?)\s*(hora|horas|día|dias|días|semana|semanas|mes|meses)\b", a):
        return "duracion"

    # Inicio (gradual/repentino + referencias “anoche”, “hoy”, etc.)
    if any(w in a for w in ["gradual", "repentino", "de golpe", "de forma súbita", "anoche", "hoy", "ayer"]):
        return "inicio"

    # Curso
    if any(w in a for w in ["mejorado", "empeorado", "igual", "estable", "peor", "mejor"]):
        return "curso"

    # Localización / irradiación
    if any(w in a for w in ["pecho","torác","abdomen","espalda","cabeza","garganta","pierna","brazo","mano","pie","cuello","lumba","costado"]):
        # Si menciona 'hacia', 'al', 'se va a', tomamos irradiación
        if any(w in a for w in ["hacia", "al ", "se va a", "irradia", "irradiado"]):
            return "irradiacion"
        return "localizacion"

    # Factores
    if any(w in a for w in ["empeora", "peor con", "con esfuerzo", "al subir", "al toser", "al comer", "al acostarme"]):
        return "factores_agravantes"
    if any(w in a for w in ["mejora", "alivia", "calma", "con reposo", "respiración profunda", "analgésico", "ibuprofeno", "paracetamol"]):
        return "factores_aliviantes"

    # Síntomas asociados comunes
    if any(w in a for w in ["fiebre","náusea","nausea","vómito","tos","disnea","falta de aire","sudor","diaforesis","mareo","cefalea","diarrea"]):
        return "sintomas_asociados"

    # Antecedentes / medicamentos / alergias / hábitos / red flags
    if any(w in a for w in ["hipertensión","diabetes","asma","epoc","cirugía","infarto","acv"]):
        return "antecedentes"
    if any(w in a for w in ["mg","tomando","pastilla","tableta","losartán","metformina","aspirina","ibuprofeno","paracetamol"]):
        return "medicamentos"
    if any(w in a for w in ["alerg","penicilina","nsaid","ibuprofeno me da","erupción"]):
        return "alergias"
    if any(w in a for w in ["no fumo","fumo","cigarro","alcohol","cerveza","social","droga","marihuana"]):
        return "habitos"
    if any(w in a for w in ["desmayo","hemoptisis","sangrado","dolor intenso súbito","dolor insoportable","dificultad para respirar"]):
        return "red_flags"

    return None

def _smart_route_answer(pending_qid: str, answer: str) -> str:
    """
    Si la respuesta no cuadra con la pregunta pendiente, intenta redirigir a la
    que sí cuadra (por ejemplo, “7” → intensidad). Si no detecta, devuelve pending_qid.
    """
    guess = _classify_answer(answer)
    if not guess:
        return pending_qid
    # Si coincide con la pendiente, perfecto
    if guess == pending_qid:
        return pending_qid
    # Permitimos saltar hacia adelante si la clasificación lo sugiere
    ids = [qid for qid, _ in ANAMNESIS_QUESTIONS]
    try:
        cur = ids.index(pending_qid)
        tgt = ids.index(guess)
        # Permitimos reordenar (capturar la que el usuario contestó)
        if tgt != cur:
            return guess
    except ValueError:
        pass
    return pending_qid


@tool
def qa_submit_answer(
    answer: str,
    state: Annotated[CombinedAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Registra la respuesta (acepta fuera de orden) y avanza."""
    qa = _ensure_qa(state)
    pending = qa.get("pending_qid")
    if not pending:
        return Command(update={"qa": qa, "messages": [ToolMessage("No hay ninguna pregunta pendiente. Inicia la entrevista primero.", tool_call_id=tool_call_id)]})

    # 1) Enrutamiento inteligente
    target_qid = _smart_route_answer(pending, answer)

    # 2) Guarda la respuesta en el target detectado
    qa["anamnesis"][target_qid] = answer

    # 3) Siguiente pregunta: si saltamos, ponemos como pendiente la que sigue al target
    ids = [qid for qid, _ in ANAMNESIS_QUESTIONS]
    try:
        idx = ids.index(target_qid)
    except ValueError:
        idx = ids.index(pending)

    nxt = ANAMNESIS_QUESTIONS[idx + 1] if idx + 1 < len(ANAMNESIS_QUESTIONS) else None

    if nxt is None:
        qa["pending_qid"] = None
        qa["done"] = True
        msg = f"Registrado: {answer}. La anamnesis ha terminado. Resume y extrae síntomas ahora."
    else:
        qa["pending_qid"] = nxt[0]
        msg = f"Registrado: {answer}. Siguiente pregunta: {nxt[1]}"

    return Command(update={"qa": qa, "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]})


# ============= 5) STRUCTURER + SYMPTOM AGENT =============
TARGET_SCHEMA_EXAMPLE = {
    "motivo_consulta": "dolor en el pecho",
    "enfermedad_actual": {
        "sintoma_principal": "dolor torácico",
        "inicio": "3 días",
        "caracteristicas": "opresivo, constante, 7/10, se agrava con esfuerzo y mejora con reposo"
    },
    "antecedentes_personales": ["hipertensión"],
    "antecedentes_familiares": [],
    "habitos": {"tabaquismo": "no", "alcohol": "social ocasional"},
    "sintomas_asociados": ["disnea leve", "diaforesis"],
    "medicamentos": ["losartán 50 mg diarios"],
    "alergias": ["ninguna conocida"],
    "red_flags": []
}

# Corrige términos con espacio
SYMPTOM_ORDER = [
 'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills',
 'joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting',
 'burning_micturition','spotting_urination','fatigue','weight_gain','anxiety',
 'cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
 'patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes',
 'breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin',
 'dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain',
 'constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
 'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
 'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',
 'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion',
 'chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements',
 'pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness',
 'cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes',
 'enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger',
 'extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain',
 'hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness',
 'spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side',
 'loss_of_smell','bladder_discomfort','foul_smell_of_urine','continuous_feel_of_urine',
 'passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability',
 'muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation',
 'dischromic_patches','watering_from_eyes','increased_appetite','polyuria','family_history',
 'mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
 'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding',
 'distention_of_abdomen','history_of_alcohol_consumption','fluid_overload.1','blood_in_sputum',
 'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads',
 'scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails',
 'blister','red_sore_around_nose','yellow_crust_ooze'
]

def _safe_json_only(text: str) -> Any:
    m = re.search(r'\{.*\}', text, flags=re.S)
    candidate = m.group(0) if m else text
    try:
        return json.loads(candidate)
    except Exception:
        return {}

def _safe_json_loads(text: str) -> Any:
    m = re.search(r'\{.*\}', text, flags=re.S)
    candidate = m.group(0) if m else text
    try:
        return json.loads(candidate)
    except Exception:
        return {"symptoms": []}

def _make_llm_structurer():
    return init_chat_model(
        BEDROCK_MODEL_MAIN,
        model_provider="bedrock",
        temperature=0.0,
        region_name=BEDROCK_REGION,
    )

def _make_llm_symptoms():
    return init_chat_model(
        BEDROCK_MODEL_MAIN,
        model_provider="bedrock",
        temperature=0.0,
        region_name=BEDROCK_REGION,
    )

def _anamnesis_dict_to_text(anam: Dict[str, str]) -> str:
    ordered: List[str] = []
    qmap = dict(ANAMNESIS_QUESTIONS)
    for qid, _ in ANAMNESIS_QUESTIONS:
        if qid in anam:
            ordered.append(f"{qid}: {anam[qid]}")
    for k, v in anam.items():
        if k not in qmap:
            ordered.append(f"{k}: {v}")
    return " | ".join(ordered)

def _structured_to_text(structured: Dict[str, Any]) -> str:
    parts = []
    mc = structured.get("motivo_consulta")
    if mc: parts.append(f"motivo_consulta: {mc}")
    ea = structured.get("enfermedad_actual", {}) or {}
    for k in ["sintoma_principal", "inicio", "caracteristicas"]:
        if ea.get(k): parts.append(f"{k}: {ea[k]}")
    for sec in ["sintomas_asociados","antecedentes_personales","antecedentes_familiares","medicamentos","alergias","red_flags"]:
        val = structured.get(sec, [])
        if isinstance(val, list) and val:
            parts.append(f"{sec}: {', '.join(map(str, val))}")
    hab = structured.get("habitos", {}) or {}
    if hab:
        parts.append("habitos: " + ", ".join([f"{k}={v}" for k, v in hab.items()]))
    return " | ".join(parts)

def _structure_prompt_from_anam_dict(anam: Dict[str, str]) -> str:
    anam_text = _anamnesis_dict_to_text(anam)
    schema_hint = json.dumps(TARGET_SCHEMA_EXAMPLE, ensure_ascii=False, indent=2)
    return f"""
Eres un asistente clínico. Recibirás respuestas crudas de una anamnesis (español).
Devuelve UN JSON válido (sin texto adicional) siguiendo este ESQUEMA objetivo, normalizando y resumiendo lo esencial:

ESQUEMA (ejemplo orientativo, adapta valores):
{schema_hint}

INSTRUCCIONES DE NORMALIZACIÓN:
- "motivo_consulta": frase breve.
- "enfermedad_actual": "sintoma_principal", "inicio" (p. ej. "3 días") y "caracteristicas" (calidad, 1-10, curso, factores, localización/irradiación).
- "antecedentes_personales"/"familiares": listas en minúsculas.
- "habitos": objeto con "tabaquismo" y "alcohol" (no/social/diario...).
- "sintomas_asociados": lista corta estandarizada; si no hay, lista vacía.
- "medicamentos": lista con fármaco y dosis si aplica.
- "alergias": lista (si dice "ninguna conocida", inclúyelo tal cual o vacía si corresponde).
- "red_flags": lista (si niega, lista vacía).

RESTRICCIONES:
- DEVUELVE SOLO JSON, sin explicaciones, sin bloques de código, sin comentarios.

ANAMNESIS CRUDA:
\"\"\"{anam_text}\"\"\"
""".strip()

def _symptom_prompt(anamnesis_text: str) -> str:
    vocab = "\n".join(SYMPTOM_ORDER)
    return f"""
Eres un asistente clínico. Tienes un texto de anamnesis del paciente.
Identifica ÚNICAMENTE síntomas presentes mapeando sinónimos al siguiente vocabulario canónico (lista cerrada).

VOCABULARIO (uno por línea):
{vocab}

INSTRUCCIONES:
- Devuelve SOLO JSON:
  {{"symptoms": ["<symptom_1>", "<symptom_2>", ...]}}
- Cada elemento DEBE ser exactamente uno de los términos del vocabulario.
- Si no hay síntomas, devuelve {{"symptoms": []}}.

ANAMNESIS:
\"\"\"{anamnesis_text.strip()}\"\"\"
""".strip()

def _extract_symptoms_from_qa(qa: Dict[str, Any]) -> List[str]:
    # Prioriza structured si existe:
    structured = qa.get("structured")
    if isinstance(structured, dict) and structured:
        anam_text = _structured_to_text(structured)
    else:
        anam = qa.get("anamnesis") or {}
        anam_text = _anamnesis_dict_to_text(anam) if isinstance(anam, dict) else str(anam)

    llm = _make_llm_symptoms()
    resp = llm.invoke(_symptom_prompt(anam_text))
    data = _safe_json_loads(to_text(resp))
    raw_list = data.get("symptoms", [])
    vocab_set = set(SYMPTOM_ORDER)
    return [s for s in raw_list if isinstance(s, str) and s in vocab_set]

@tool
def structure_interview(
    state: Annotated[CombinedAgentState, InjectedState]
) -> Dict[str, Any]:
    """Estructura qa['anamnesis'] → qa['structured'] (JSON clínico)."""
    qa = _ensure_qa(state)
    if isinstance(qa.get("structured"), dict) and qa["structured"]:
        return {"ok": True, "structured": qa["structured"], "cached": True}

    anam = qa.get("anamnesis", {})
    if not isinstance(anam, dict) or not anam:
        return {"ok": False, "error": "No hay anamnesis para estructurar."}

    llm = _make_llm_structurer()
    resp = llm.invoke(_structure_prompt_from_anam_dict(anam))
    data = _safe_json_only(to_text(resp))
    if not isinstance(data, dict) or not data:
        return {"ok": False, "error": "No se pudo estructurar con el LLM."}

    qa["structured"] = data
    return {"ok": True, "structured": data, "cached": False}

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

@tool
def summarize_interview(state: Annotated[CombinedAgentState, InjectedState]) -> str:
    """Genera un resumen (prefiere estructurado si existe)."""
    qa = _ensure_qa(state)
    if not qa.get("done"):
        return "No puedo generar un resumen porque la entrevista aún no ha terminado."

    if isinstance(qa.get("structured"), dict) and qa["structured"]:
        s = qa["structured"]
        ea = s.get("enfermedad_actual", {}) or {}
        partes = [
            "Resumen (estructurado):",
            f"- Motivo de consulta: {s.get('motivo_consulta','')}",
            f"- Síntoma principal: {ea.get('sintoma_principal','')}",
            f"- Inicio: {ea.get('inicio','')}",
            f"- Características: {ea.get('caracteristicas','')}",
            f"- Síntomas asociados: {', '.join(s.get('sintomas_asociados', [])) or 'ninguno'}",
            f"- Antecedentes personales: {', '.join(s.get('antecedentes_personales', [])) or 'ninguno'}",
            f"- Medicamentos: {', '.join(s.get('medicamentos', [])) or 'ninguno'}",
            f"- Alergias: {', '.join(s.get('alergias', [])) or 'ninguna'}",
            f"- Red flags: {', '.join(s.get('red_flags', [])) or 'negadas'}",
        ]
        return "\n".join(partes)

    # Fallback por preguntas
    summary_parts = ["Resumen de la anamnesis:"]
    for qid, question_text in ANAMNESIS_QUESTIONS:
        if qid in qa["anamnesis"]:
            summary_parts.append(f"- {question_text}: {qa['anamnesis'][qid]}")
    return "\n".join(summary_parts)

# ============= 6) AGENTE =============
all_tools = [
    eval_consent,
    qa_next_question,
    qa_submit_answer,
    summarize_interview,
    structure_interview,
    symptom_extractor,
    symptoms_from_state,
]

MEDICAL_INTERVIEW_PROTOCOL = f"""
# PROTOCOLO DE ENTREVISTA MÉDICA
Hoy es {datetime.now().strftime('%Y-%m-%d')}.

Si el usuario indica que tiene una pregunta o problema médico, DEBES seguir estos pasos estrictamente:

1.  **Solicitar Consentimiento**: Responde ÚNICAMENTE con:
    "Antes de continuar, necesito su consentimiento.
    Declaro que comprendo que este agente conversacional tiene fines informativos y
    educativos, no sustituye la atención médica profesional, y que los resultados son
    estimaciones probabilísticas sujetas a error. La información que proporcione se usará solo
    durante esta sesión con fines de demostración y no se almacenará de forma permanente ni
    se compartirá con terceros. En caso de presentar síntomas de urgencia, debo buscar
    atención inmediata.
    Si está de acuerdo en continuar bajo estas condiciones, responda: “Acepto”.
    Si no está de acuerdo, responda: “No acepto” y finalizaré la conversación."

Luego:
2) Usa `eval_consent` con la respuesta del usuario.
3) Si acepta, `qa_next_question`; tras cada respuesta, `qa_submit_answer`.
4) Al final, `summarize_interview`.
5) Después, `structure_interview` para normalizar datos clínicos en qa["structured"].
6) Finalmente, `symptoms_from_state` para extraer síntomas canónicos.
"""

# Modelo principal (Bedrock)
model = init_chat_model(
    BEDROCK_MODEL_MAIN,
    model_provider="bedrock",     # o "bedrock_converse" si prefieres la API nueva
    temperature=0,
    region_name=BEDROCK_REGION,
)

agent = create_react_agent(
    model,
    tools=all_tools,
    prompt=MEDICAL_INTERVIEW_PROTOCOL,
    state_schema=CombinedAgentState
)

def run_turn(agent_executor, state: dict, user_input: str = ""):
    # 👇 Empujón si no hay input
    if not user_input:
        user_input = "continuar"

    state["messages"].append(HumanMessage(content=user_input))
    print(f"👤 Usuario: {user_input}")

    new_state = agent_executor.invoke(state)

    # Imprimir el último mensaje con texto útil
    printed = False
    for m in reversed(new_state["messages"]):
        txt = to_text(m).strip()
        # Ignora vacíos/puntuación suelta
        if not txt or txt in {".", "…"}: 
            continue
        print(f"🤖 Agente: {txt}\n")
        printed = True
        break
    if not printed:
        print("🤖 Agente: [silencio]\n")

    return new_state




if __name__ == "__main__":
    current_state = {
        "messages": [],
        "qa": {"anamnesis": {}, "pending_qid": None, "done": False, "symptoms": [], "structured": {}}
    }
    print("--- Asistente Médico Interactivo (Bedrock) ---")
    print("Escribe tu consulta médica. (o 'salir' para terminar)\n")
    while True:
        try:
            user_input = input("👤 Usuario: ")
        except EOFError:
            break
        if user_input.lower().strip() == 'salir':
            print("🤖 Agente: Adiós.")
            break
        current_state = run_turn(agent, current_state, user_input)

        # Si la entrevista terminó, damos un turno extra para que el agente
        # ejecute summarize_interview -> structure_interview -> symptoms_from_state
        if current_state["qa"].get("done"):
            print("--- ENTREVISTA FINALIZADA: RESUMEN + ESTRUCTURA + SÍNTOMAS ---\n")
            current_state = run_turn(agent, current_state)
            anamnesis = current_state["qa"]["anamnesis"]
            symptoms = current_state["qa"]["symptoms"]
            print(f"Síntomas detectados: {symptoms}")
            print(f"Anamnesis: {anamnesis}")
            anamnesis.pop("sintomas_asociados")
            anamnesis["symptoms_present"] = symptoms

            
            async def call_generate_recommendation(data):
                url = "http://localhost:8000/generate-recommendation/"
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=data)
                    if response.status_code == 200:
                        recommendation = response.json()
                        print(f"Recomendación generada: {recommendation['recommendation']}")
                    else:
                        print(f"Error al obtener la recomendación: {response.status_code}")

            asyncio.run(call_generate_recommendation(anamnesis))

            async def call_predict_probabilities(data):
                url = "http://localhost:8000/predict-probabilities/"
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=data)
                    if response.status_code == 200:
                        recommendation = response.json()
                        print(f"Recomendación generada: {recommendation['recommendation']}")
                    else:
                        print(f"Error al obtener la recomendación: {response.status_code}")

            asyncio.run(call_predict_probabilities(symptoms))

            

