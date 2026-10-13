from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.generate import answer
from app.retrieve import retrieve

app = FastAPI(title="RAG Docs Assistant")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    passages = retrieve(req.question)
    try:
        text = answer(req.question, passages)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Erreur du LLM : {exc}")
    return {
        "answer": text,
        "sources": [
            {"page": p["page"], "snippet": p["text"][:300], "score": round(p["score"], 3)}
            for p in passages
        ],
    }


@app.get("/")
def index():
    return FileResponse("app/static/index.html")