from groq import Groq

from app import config

REFUSAL = "I could not find this information in the document."

SYSTEM = f"""You are an assistant that answers only from the provided excerpts.
Rules:
- Use no knowledge from outside the excerpts.
- Cite the pages you used in the form [p. N].
- If the excerpts do not contain the answer, reply exactly: "{REFUSAL}"
- Answer in the same language as the question, concisely."""

def build_context(passages):
    return "\n\n".join(f"[p. {p['page']}]\n{p['text']}" for p in passages)


def answer(question, passages):
    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Extraits :\n{build_context(passages)}\n\nQuestion : {question}",
            },
        ],
    )
    return response.choices[0].message.content.strip()