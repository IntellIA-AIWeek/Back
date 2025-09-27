from typing import Dict, List, Any, Tuple
import json
import re
from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv(os.path.join("..", ".env"), override=True)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAr-18BxO1xmvvvk2hrao_KQogkrUaNNpM"

SYMPTOM_ORDER = ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload.1','blood_in_sputum','prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze']

def _safe_json_loads(text: str) -> Any:
    match = re.search(r'\{.*\}', text, flags=re.S)
    candidate = match.group(0) if match else text
    try:
        return json.loads(candidate)
    except Exception:
        return {"symptoms": []}
    
def _make_llm():
    # Puedes cambiar a "gpt-4o" u otro si deseas
    return init_chat_model(
        "gemini-2.5-flash",
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

def extract_symptoms_from_qa(qa: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    # Soporta claves 'amnsesis' (como indicaste) y 'anamnesis'
    anamnesis_data = qa.get("anamnesis") or ""
    
    # Si es dict, lo convertimos a string concatenando respuestas
    if isinstance(anamnesis_data, dict):
        anamnesis_text = " ".join(f"{k}: {v}" for k, v in anamnesis_data.items())
    else:
        anamnesis_text = str(anamnesis_data)

    llm = _make_llm()
    prompt = _symptom_prompt(anamnesis_text)

    resp = llm.invoke(prompt)  # LangChain: devuelve un AIMessage
    data = _safe_json_loads(getattr(resp, "content", "") or str(resp))

    raw_list = data.get("symptoms", [])
    # Normaliza y filtra a vocabulario exacto (por seguridad)
    vocab_set = set(SYMPTOM_ORDER)
    detected = [s for s in raw_list if isinstance(s, str) and s in vocab_set]

    return detected


@tool("symptom_extractor")
def symptom_extractor(qa_json: str) -> Dict[str, Any]:
    """
    Extrae síntomas canónicos en formato de lista.
    Retorna:
      { "symptoms": [<str>...] }
    """
    try:
        qa = json.loads(qa_json)
    except Exception:
        return {"symptoms": [], "error": "qa_json inválido"}

    symptoms = extract_symptoms_from_qa(qa)
    return {"symptoms": symptoms}

# qa = {
#     "anamnesis": {
#         "sx_ppal": "Fiebre alta con tos y dolor de garganta.",
#         "inicio": "Inicio repentino, comenzó hace tres días en la noche.",
#         "duracion": "Tres días continuos.",
#         "curso": "Ha empeorado ligeramente desde el primer día.",
#         "intensidad": "7/10",
#         "localizacion": "Molestia principal en la garganta y cabeza; malestar general.",
#         "irradiacion": "El dolor de garganta no irradia; el dolor de cabeza se siente hacia la frente.",
#         "factores_agravantes": "Aire frío y hablar mucho empeoran la garganta; el esfuerzo físico aumenta la tos.",
#         "factores_aliviantes": "Líquidos tibios y reposo; paracetamol baja la fiebre temporalmente.",
#         "sintomas_asociados": "Cansancio, sudoración nocturna, congestión nasal, tos seca, náuseas leves y pérdida de apetito.",
#         "antecedentes": "Sin enfermedades crónicas conocidas; resfriados frecuentes en invierno.",
#         "medicamentos": "Paracetamol 500 mg cada 8 horas desde ayer.",
#         "alergias": "Niega alergias a medicamentos y alimentos.",
#         "habitos": "No fuma; alcohol social ocasional; duerme 6 horas promedio.",
#         "red_flags": "Niega dolor torácico intenso, desmayos, sangrado o disnea severa."
#     }
# }

# symptoms = extract_symptoms_from_qa(qa)

# print(symptoms)
# print("\n=== Vector binario (primeros 40) ===")
# print(symptoms)
# print("Longitud total:", len(symptoms))