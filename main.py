
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import shutil
import uuid

from rag_engine import build_rag_pipeline, ask_question

load_dotenv()

app = FastAPI(title="Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store sessions: session_id -> {retriever, llm, full_text, filename}
sessions = {}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class AskRequest(BaseModel):
    session_id: str
    question: str

class AskResponse(BaseModel):
    answer: str
    source_count: int

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks_created: int
    message: str

@app.get("/")
def root():
    return {"status": "Document Q&A API is running"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    # Validate file type
    allowed_extensions = (".pdf", ".docx", ".txt")
    if not file.filename.endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported")

    # Save uploaded file temporarily
    session_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        retriever, llm, full_text = build_rag_pipeline(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    # Store session
    sessions[session_id] = {
        "retriever": retriever,
        "llm": llm,
        "full_text": full_text,
        "filename": file.filename
    }

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        chunks_created=len(retriever.invoke("test")),
        message="Document processed successfully. You can now ask questions."
    )

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    session = sessions[request.session_id]
    answer, sources = ask_question(
        request.question,
        session["retriever"],
        session["llm"],
        session["full_text"]
    )

    return AskResponse(answer=answer, source_count=len(sources))

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session cleared"}
    raise HTTPException(status_code=404, detail="Session not found")