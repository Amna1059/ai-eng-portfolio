# AI Engineering Portfolio

A collection of AI-powered backend systems built with Python, FastAPI, and Google Gemini — built while learning to ship production-style LLM applications.

## Projects in this repo

### 1. AI Chat API (Day 1)
A FastAPI backend that wraps the Gemini API with multi-turn conversation memory, using per-session history so a frontend can hold a real back-and-forth conversation instead of one-off prompts.

**Files:** `main.py`, `chat.py`

**Tech:** Python · FastAPI · Google Gemini API · Pydantic

**Endpoints:**
- `GET /` — health check
- `POST /chat` — send a message, get an AI reply with memory
- `DELETE /chat/{session_id}` — clear a session

### 2. Document Q&A System — RAG (Day 2)
A Retrieval-Augmented Generation pipeline that lets a user upload any PDF, DOCX, or TXT file and ask natural-language questions about it, answered with sources pulled from the actual document — not the model's general knowledge.

**Files:** `rag_engine.py`, `main.py`

**Tech:** Python · LangChain · ChromaDB · Google Gemini (embeddings + generation) · FastAPI

**How it works:**
1. Document is loaded and split into overlapping chunks
2. Each chunk is embedded and stored in a local ChromaDB vector store
3. A question is embedded and matched against the most relevant chunks
4. The matched chunks are passed to Gemini as context to generate a grounded answer

**Endpoints:**
- `POST /upload` — upload a document, get back a `session_id`
- `POST /ask` — ask a question about an uploaded document
- `DELETE /session/{session_id}` — clear a session

**A real engineering tradeoff I ran into:** naive top-k retrieval (only fetching the 3 most similar chunks) fails on broad questions like "summarize this document," since no single chunk is semantically closest to that kind of query. Fixed by detecting broad/summary-style questions and routing them to the full document text instead of the retriever.

**Frontend:** a matching React interface for this project lives in a separate repo → [rag-frontend](https://github.com/Amna1059/rag-frontend)

## Setup

```bash
git clone https://github.com/Amna1059/ai-eng-portfolio.git
cd ai-eng-portfolio
pip install -r requirements.txt
```

Create a `.env` file in the root:
```
GEMINI_API_KEY=your-gemini-api-key-here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

## Run locally

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API docs (Swagger UI) to test endpoints directly.

## What's next

- Agentic AI system with tool-calling (web search + calculator) using LangGraph
- Evaluation framework (RAGAS) to measure RAG answer quality
- Deployment (Railway/Render)

## Author

**Amna Iftikhar** — BSc Computer Science, LUMS
[LinkedIn](https://www.linkedin.com/in/amna1059/) · [GitHub](https://github.com/Amna1059)
