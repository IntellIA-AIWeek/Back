import os
from dotenv import load_dotenv
from datetime import datetime
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from app.utils.utils import format_messages
from app.utils.prompts import CHAT_PROMPT, TODO_USAGE_INSTRUCTIONS, SUBAGENT_USAGE_INSTRUCTIONS
from app.utils.todo_tools import write_todos, read_todos
from app.utils.state import DeepAgentState

# Cargar configuraciones y claves API desde el archivo .env
load_dotenv(os.path.join("..", ".env"), override=True)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAr-18BxO1xmvvvk2hrao_KQogkrUaNNpM"
os.environ["TAVILY_API_KEY"] = "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"

# Crear modelo de chat (Gemini)
model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# Definir las herramientas que usará el agente
sub_agent_tools = []  # Si usas sub-agentes, añádelos aquí
built_in_tools = [write_todos, read_todos]

# Crear el agente utilizando las herramientas definidas
all_tools = sub_agent_tools + built_in_tools

# Construir las instrucciones para el agente
INSTRUCTIONS = (
    "# Chat Agent\n"
    + CHAT_PROMPT
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# TODO MANAGEMENT\n"
    + TODO_USAGE_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + "# SUB-AGENT DELEGATION\n"
    + SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=3, max_researcher_iterations=3, date=datetime.now()
    )
)

# Crear el agente con React
chat_agent = create_react_agent(
    model, all_tools, prompt=INSTRUCTIONS, state_schema=DeepAgentState
)

# Función para interactuar con el agente
def chat_with_agent(user_input):
    if not user_input.strip():
        raise ValueError("El contenido del mensaje está vacío.")

    # Enviar el mensaje al agente y obtener la respuesta
    response = chat_agent.invoke({
        "messages": [
            {"role": "user", "content": user_input}
        ],
    })

    # Formatear y mostrar la respuesta del agente
    format_messages(response["messages"])
    return response

# Función principal que ejecuta la lógica de conversación
def main():
    print("Bienvenido al agente de chat. Escribe 'exit' para terminar.")
    
    while True:
        user_input = input("Tú: ")
        
        if user_input.lower() == 'exit':
            print("Cerrando la sesión de chat...")
            break

        # Obtener respuesta del agente
        response = chat_with_agent(user_input)
        print(response)
        
        

if __name__ == "__main__":
    main()
