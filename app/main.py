import json
import os
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app import config
from app.generate import answer
from app.ingest import index_document
from app.retrieve import get_resources, retrieve

app = FastAPI(title="RAG Docs Assistant")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
INDEX_LOCK = threading.RLock()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    with INDEX_LOCK:
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


def _current_document():
    if config.ACTIVE_DOCUMENT_META_PATH.exists():
        try:
            return json.loads(config.ACTIVE_DOCUMENT_META_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"name": Path(config.PDF_PATH).name, "source": "default"}


def _write_metadata(metadata):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=config.UPLOAD_DIR, delete=False
    ) as temporary:
        json.dump(metadata, temporary)
        metadata_path = Path(temporary.name)
    os.replace(metadata_path, config.ACTIVE_DOCUMENT_META_PATH)


def _index_uploaded_document(content, filename):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pdf", dir=config.UPLOAD_DIR, delete=False
        ) as temporary:
            temporary.write(content)
            temp_path = Path(temporary.name)

        model, client = get_resources()
        with INDEX_LOCK:
            result = index_document(temp_path, model=model, client=client)
            os.replace(temp_path, config.ACTIVE_DOCUMENT_PATH)
            temp_path = None
            metadata = {"name": filename, "source": "upload", **result}
            _write_metadata(metadata)
        return metadata
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.get("/documents/current")
def current_document():
    return _current_document()


@app.post("/documents")
async def upload_document(document: UploadFile = File(...)):
    filename = Path(document.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Please upload a PDF file.")

    content = await document.read(config.MAX_UPLOAD_BYTES + 1)
    await document.close()
    if len(content) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"The PDF must be smaller than {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="The uploaded file is not a valid PDF.")

    try:
        return await run_in_threadpool(_index_uploaded_document, content, filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to index the PDF: {exc}") from exc


@app.get("/")
def index():
    return FileResponse("app/static/index.html")
