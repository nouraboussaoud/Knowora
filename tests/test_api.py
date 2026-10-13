from fastapi.testclient import TestClient
from pathlib import Path

import app.main as main

client = TestClient(main.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(main, "retrieve", lambda q: [{"page": 3, "text": "texte", "score": 0.9}])
    monkeypatch.setattr(main, "answer", lambda q, p: "Réponse [p. 3]")
    r = client.post("/ask", json={"question": "Une question ?"})
    assert r.status_code == 200
    assert r.json()["sources"][0]["page"] == 3


def test_question_too_short():
    assert client.post("/ask", json={"question": "a"}).status_code == 422


def test_upload_replaces_active_document(monkeypatch, tmp_path):
    monkeypatch.setattr(main.config, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(main.config, "ACTIVE_DOCUMENT_PATH", tmp_path / "active.pdf")
    monkeypatch.setattr(main.config, "ACTIVE_DOCUMENT_META_PATH", tmp_path / "active.json")
    monkeypatch.setattr(main, "get_resources", lambda: (object(), object()))

    def fake_index(path, model, client):
        assert Path(path).read_bytes().startswith(b"%PDF-")
        return {"chunks": 3, "pages": 2}

    monkeypatch.setattr(main, "index_document", fake_index)
    response = client.post(
        "/documents",
        files={"document": ("spring-guide.pdf", b"%PDF-1.7 test", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "spring-guide.pdf"
    assert (tmp_path / "active.pdf").exists()
    assert client.get("/documents/current").json()["name"] == "spring-guide.pdf"


def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents",
        files={"document": ("notes.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 415
