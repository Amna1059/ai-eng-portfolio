# AI Chat API

A FastAPI backend powered by Google Gemini with conversation memory.

## Features
- Multi-turn conversation with memory
- Session management
- REST API with auto-generated docs

## Tech Stack
Python · FastAPI · Google Gemini API · Pydantic

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://localhost:8000/docs to test.

## Endpoints
- `GET /` — health check
- `POST /chat` — chat with memory
- `DELETE /chat/{session_id}` — clear session