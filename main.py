from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# 1. Purana import hata kar Naya Import lagayein
from google import genai
from google.genai import types

load_dotenv()

# 2. Purani 'genai.configure' line HATA dein.
# Ab hum seedha modern client banayenge jo auto-key pick karega:
client = genai.Client()

app = FastAPI(title="AI Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str
    message_count: int

@app.get("/")
def root():
    return {"status": "AI Chat API is running"}

@app.post("/chat")
def chat(request: ChatRequest):
    # 3. Chat Session logic ko naye SDK ke mutabiq update karein
    if request.session_id not in sessions:
        # Naye client ke zariye chats manage hoti hain
        sessions[request.session_id] = client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful AI assistant. Be concise and clear."
            )
        )
    
    chat_session = sessions[request.session_id]
    response = chat_session.send_message(request.message)
    
    # Naye SDK mein history check karne ka tareeqa:
    history_list = list(chat_session.get_history())
    
    return ChatResponse(
        reply=response.text,
        message_count=len(history_list)
    )

@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session cleared"}