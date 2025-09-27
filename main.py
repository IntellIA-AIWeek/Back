from openai import AzureOpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
from transformers import pipeline

# Cargar el pipeline de HuggingFace
diagnosis_pipe = pipeline(
    "text-generation", 
    model="alpha-ai/Medical-Diagnosis-COT-Gemma3-270M"
)

# Cargar el modelo de predicción de enfermedades
model = joblib.load(
    hf_hub_download("AWeirdDev/human-disease-prediction", "sklearn_model.joblib")
)

# Configuración de Azure OpenAI
endpoint = "https://invuniandesai-2.openai.azure.com/"
model_name = "gpt-4o-mini"
deployment = "gpt"
api_key = "DivIIrUFG4NvQitVNb6XTmDe34PGGwrsp4piIKJDCPXfHfeLwhfAJQQJ99BFACHYHv6XJ3w3AAABACOGn4Td" 


client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version="2024-12-01-preview"
)

# Lista de síntomas
SYMPTOM_ORDER = ['itching',
 'skin_rash',
 'nodal_skin_eruptions',
 'continuous_sneezing',
 'shivering',
 'chills',
 'joint_pain',
 'stomach_pain',
 'acidity',
 'ulcers_on_tongue',
 'muscle_wasting',
 'vomiting',
 'burning_micturition',
 'spotting_ urination',
 'fatigue',
 'weight_gain',
 'anxiety',
 'cold_hands_and_feets',
 'mood_swings',
 'weight_loss',
 'restlessness',
 'lethargy',
 'patches_in_throat',
 'irregular_sugar_level',
 'cough',
 'high_fever',
 'sunken_eyes',
 'breathlessness',
 'sweating',
 'dehydration',
 'indigestion',
 'headache',
 'yellowish_skin',
 'dark_urine',
 'nausea',
 'loss_of_appetite',
 'pain_behind_the_eyes',
 'back_pain',
 'constipation',
 'abdominal_pain',
 'diarrhoea',
 'mild_fever',
 'yellow_urine',
 'yellowing_of_eyes',
 'acute_liver_failure',
 'fluid_overload',
 'swelling_of_stomach',
 'swelled_lymph_nodes',
 'malaise',
 'blurred_and_distorted_vision',
 'phlegm',
 'throat_irritation',
 'redness_of_eyes',
 'sinus_pressure',
 'runny_nose',
 'congestion',
 'chest_pain',
 'weakness_in_limbs',
 'fast_heart_rate',
 'pain_during_bowel_movements',
 'pain_in_anal_region',
 'bloody_stool',
 'irritation_in_anus',
 'neck_pain',
 'dizziness',
 'cramps',
 'bruising',
 'obesity',
 'swollen_legs',
 'swollen_blood_vessels',
 'puffy_face_and_eyes',
 'enlarged_thyroid',
 'brittle_nails',
 'swollen_extremeties',
 'excessive_hunger',
 'extra_marital_contacts',
 'drying_and_tingling_lips',
 'slurred_speech',
 'knee_pain',
 'hip_joint_pain',
 'muscle_weakness',
 'stiff_neck',
 'swelling_joints',
 'movement_stiffness',
 'spinning_movements',
 'loss_of_balance',
 'unsteadiness',
 'weakness_of_one_body_side',
 'loss_of_smell',
 'bladder_discomfort',
 'foul_smell_of urine',
 'continuous_feel_of_urine',
 'passage_of_gases',
 'internal_itching',
 'toxic_look_(typhos)',
 'depression',
 'irritability',
 'muscle_pain',
 'altered_sensorium',
 'red_spots_over_body',
 'belly_pain',
 'abnormal_menstruation',
 'dischromic _patches',
 'watering_from_eyes',
 'increased_appetite',
 'polyuria',
 'family_history',
 'mucoid_sputum',
 'rusty_sputum',
 'lack_of_concentration',
 'visual_disturbances',
 'receiving_blood_transfusion',
 'receiving_unsterile_injections',
 'coma',
 'stomach_bleeding',
 'distention_of_abdomen',
 'history_of_alcohol_consumption',
 'fluid_overload.1',
 'blood_in_sputum',
 'prominent_veins_on_calf',
 'palpitations',
 'painful_walking',
 'pus_filled_pimples',
 'blackheads',
 'scurring',
 'skin_peeling',
 'silver_like_dusting',
 'small_dents_in_nails',
 'inflammatory_nails',
 'blister',
 'red_sore_around_nose',
 'yellow_crust_ooze']

# FastAPI para servir al frontend
app = FastAPI()

origins = [
    "http://localhost:3000",   # React local
    "http://127.0.0.1:3000",
    "http://localhost:5173",   # Vite local (si usas)
    "http://localhost:5174",   # Nuevo puerto que mencionaste
]

# Agregar el middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Permitir solicitudes desde estos orígenes
    allow_credentials=True,
    allow_methods=["*"],            # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],            # Permitir todos los headers
)


# Definir las clases de solicitud
class SymptomRequest(BaseModel):
    symptoms: List[str]

class AnamnesisRequest(BaseModel):
    sx_ppal: str
    inicio: str
    duracion: str
    curso: str
    intensidad: str
    localizacion: str
    irradiacion: str
    factores_agravantes: str
    factores_aliviantes: str
    antecedentes: str
    medicamentos: str
    alergias: str
    habitos: str
    red_flags: str
    symptoms_present: List[str]

class RecommendationRequest(BaseModel):
    anamnesis: AnamnesisRequest
    symptoms_present: List[str]  # Lista de síntomas presentes

# Endpoint para predecir probabilidades (mantienes la lógica de predicción de enfermedades)
@app.post("/predict-probabilities")
def predict_probabilities(request: SymptomRequest):
    x = np.zeros(len(SYMPTOM_ORDER))

    for symptom in request.symptoms:
        if symptom in SYMPTOM_ORDER:
            idx = SYMPTOM_ORDER.index(symptom)
            x[idx] = 1.0

    x = np.expand_dims(x, axis=0)
    probs = model.predict_proba(x)[0]

    top_idx = np.argsort(probs)[-3:][::-1]
    top_preds = [
        {"disease": model.classes_[i], "probability": float(probs[i])}
        for i in top_idx
    ]

    return {"top_predictions": top_preds}

# Endpoint para generar recomendaciones usando Azure OpenAI (GPT-4o-mini)
@app.post("/generate-recommendation/")
async def generate_recommendation(data: AnamnesisRequest):
    try:
        symptoms_present = data.symptoms_present
        
        # Construir el prompt
        prompt = f"""
        A continuación se presentan los síntomas, historial médico y diagnóstico de un paciente:

        Síntomas principales: {data.sx_ppal}
        Inicio: {data.inicio}
        Duración: {data.duracion}
        Curso: {data.curso}
        Intensidad: {data.intensidad}
        Localización: {data.localizacion}
        Irradiación: {data.irradiacion}
        Factores agravantes: {data.factores_agravantes}
        Factores aliviantes: {data.factores_aliviantes}
        Antecedentes médicos: {data.antecedentes}
        Medicamentos: {data.medicamentos}
        Alergias: {data.alergias}
        Hábitos: {data.habitos}
        Red flags: {data.red_flags}
        Sintomas presentes: {', '.join(symptoms_present)}

        Proporcione un resumen breve y conciso sobre la situación de este paciente, incluyendo únicamente la evaluación general y la conclusión sobre su estado. 
        No mencione nombres de medicamentos ni dosis específicas. Solo haga un resumen y una conclusión clara y directa.
        """

        # Llamada a la API de OpenAI (chat completions)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asistente médico que da recomendaciones breves."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.7,
        )

        recommendation = response.choices[0].message.content.strip()

        return {"recommendation": recommendation}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/call-agent/")
def call_agent():
    pass
