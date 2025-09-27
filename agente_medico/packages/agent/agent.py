# packages/agent/agent.py

import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

# Carga las variables de entorno (opcional pero recomendado)
load_dotenv()

# --- 1. LÓGICA DE LA ENTREVISTA (ANAMNESIS) ---
ANAMNESIS_QUESTIONS = {
    "sx_ppal": "¿Cuál es tu síntoma principal?",
    "duracion": "¿Desde hace cuánto tiempo tienes este síntoma?",
    "intensidad": "En una escala del 1 al 10, ¿qué tan intenso es el síntoma?",
    "localizacion": "¿En qué parte del cuerpo se localiza el síntoma?",
    "antecedentes": "¿Tienes antecedentes médicos relevantes (como alergias, cirugías previas, etc.)?",
}

# --- 2. DEFINICIÓN DE HERRAMIENTAS (TOOLS) ---
@tool
def eval_consent(user_response: str) -> str:
    """Evalúa la respuesta del usuario para otorgar o negar el consentimiento para la entrevista médica."""
    if user_response.lower().strip() in ["acepto", "si", "de acuerdo", "ok", "sí"]:
        return "Consentimiento otorgado. El agente ahora debe iniciar la entrevista llamando a la herramienta `get_anamnesis_question` con el tema 'sx_ppal'."
    else:
        return "Consentimiento no otorgado. El agente debe informar al usuario que no puede continuar sin su consentimiento."

@tool
def get_anamnesis_question(topic: Literal["sx_ppal", "duracion", "intensidad", "localizacion", "antecedentes"]) -> str:
    """Obtiene la pregunta para un tema específico de la anamnesis. Usar un tema a la vez."""
    return ANAMNESIS_QUESTIONS.get(topic, "Tema no encontrado. Por favor, elige uno de los temas válidos.")

@tool
def finish_interview(summary: str) -> str:
    """Llama a esta herramienta cuando la entrevista haya finalizado para confirmar y resumir la información recopilada."""
    return f"Anamnesis finalizada. La información ha sido registrada. Resumen proporcionado por el agente: {summary}"

@tool
def think_tool(reflection: str) -> str:
    """Permite al agente reflexionar sobre el progreso y planificar los siguientes pasos."""
    return f"Reflexión registrada: {reflection}"

# --- 3. CONFIGURACIÓN DEL AGENTE PRINCIPAL ---
all_tools = [eval_consent, get_anamnesis_question, finish_interview, think_tool]

CHAT_PROMPT = "Eres un asistente de IA conversacional. Tu objetivo principal es ayudar al usuario. Sin embargo, tienes un protocolo especial para consultas médicas."
MEDICAL_INTERVIEW_PROTOCOL = f"""
# PROTOCOLO DE ENTREVISTA MÉDICA
Si el usuario indica que tiene una pregunta o problema médico, DEBES seguir estos pasos estrictamente:
1.  **Solicitar Consentimiento**: NO hagas ninguna pregunta médica. Primero, debes pedir consentimiento de forma explícita. Responde únicamente con el siguiente texto:
    "Para poder ayudarte, necesito hacerte algunas preguntas sobre tus síntomas. Esta conversación no reemplaza un diagnóstico médico profesional. ¿Aceptas continuar?"
2.  **Evaluar Respuesta**: Una vez que el usuario responda, utiliza la herramienta `eval_consent` para verificar si aceptó.
3.  **Iniciar Entrevista (si hay consentimiento)**:
    - Si `eval_consent` devuelve "Consentimiento otorgado", DEBES iniciar la anamnesis.
    - Para ello, llama a la herramienta `get_anamnesis_question` UNA SOLA VEZ con el primer tema: `sx_ppal`.
    - Envía la pregunta resultante al usuario.
4.  **Continuar la Entrevista**:
    - Después de que el usuario responda a una pregunta, llama a `get_anamnesis_question` con el siguiente tema en el orden establecido.
    - El orden es: {', '.join(ANAMNESIS_QUESTIONS.keys())}.
    - Haz UNA pregunta a la vez. Espera la respuesta del usuario antes de pasar al siguiente tema.
5.  **Finalizar Entrevista**:
    - Después de recibir la respuesta a la última pregunta ('antecedentes'), utiliza la herramienta `think_tool` para hacer un resumen de toda la información que recopilaste.
    - Finalmente, llama a la herramienta `finish_interview` con el resumen que creaste.
    - Informa al usuario que la recopilación de datos ha terminado.
- Si en el paso 2 el consentimiento es negado, informa al usuario que no puedes proceder y espera sus instrucciones.
- Para cualquier otra consulta no médica, conversa normalmente.
- Hoy es {datetime.now().strftime('%Y-%m-%d')}.
"""
INSTRUCTIONS = CHAT_PROMPT + "\n" + "="*50 + "\n" + MEDICAL_INTERVIEW_PROTOCOL

model = init_chat_model(
    "gemini-1.5-flash",
    model_provider="google_genai",
    temperature=0,
)

# --- 4. EXPOSICIÓN DEL AGENTE ---
# ¡IMPORTANTE! La variable DEBE llamarse 'agent' para que Agent Core la detecte.
agent = create_react_agent(model, all_tools, prompt=INSTRUCTIONS, state_schema=AgentState)