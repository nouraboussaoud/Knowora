from fastapi.testclient import TestClient

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