import json
import re
import time

from groq import Groq

from app import config
from app.generate import REFUSAL, answer
from app.retrieve import retrieve

JUDGE = """Tu compares une réponse à une réponse de référence.
Réponds en JSON : {"correct": true} si la réponse contient la même information essentielle, sinon {"correct": false}."""


def judge(client, question, expected, got):
    r = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE},
            {"role": "user", "content": f"Question : {question}\nRéférence : {expected}\nRéponse : {got}"},
        ],
    )
    try:
        return bool(json.loads(r.choices[0].message.content)["correct"])
    except Exception:
        return False


def main():
    client = Groq(api_key=config.GROQ_API_KEY)
    questions = json.load(open("eval/questions.json", encoding="utf-8"))
    rows, hits, correct, citations, refusals = [], [], [], [], []

    for item in questions:
        passages = retrieve(item["q"])
        got = answer(item["q"], passages)
        expected = set(item["expected_pages"])
        retrieved = {p["page"] for p in passages}
        cited = {int(n) for n in re.findall(r"\[p\.\s*(\d+)\]", got)}

        if item["type"] == "out_of_scope":
            ok = REFUSAL in got
            refusals.append(ok)
            rows.append((item["q"], item["type"], "-", "-", "OK" if ok else "KO"))
        else:
            hit = bool(expected & retrieved)
            ok = judge(client, item["q"], item["expected_answer"], got)
            cite_ok = bool(expected & cited)
            hits.append(hit); correct.append(ok); citations.append(cite_ok)
            rows.append((item["q"], item["type"], "OK" if hit else "KO",
                         "OK" if cite_ok else "KO", "OK" if ok else "KO"))
        time.sleep(1.5)  # limites du plan gratuit

    pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f} %" if xs else "n/a"
    lines = [
        f"Configuration : chunk={config.CHUNK_WORDS} mots, overlap={config.CHUNK_OVERLAP}, top_k={config.TOP_K}, "
        f"embeddings={config.EMBED_MODEL}, llm={config.GROQ_MODEL}", "",
        "| Métrique | Score |", "|---|---|",
        f"| Hit@k (bonne page récupérée) | {pct(hits)} |",
        f"| Réponses correctes (juge LLM) | {pct(correct)} |",
        f"| Citations valides | {pct(citations)} |",
        f"| Refus corrects (hors sujet) | {pct(refusals)} |", "",
        "| Question | Type | Retrieval | Citation | Réponse |", "|---|---|---|---|---|",
    ]
    lines += [f"| {q} | {t} | {a} | {b} | {c} |" for q, t, a, b, c in rows]
    open("eval/results.md", "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()