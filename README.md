# Knowora

**Your knowledge. One intelligent conversation.**

Knowora is a compact Retrieval-Augmented Generation (RAG) assistant for asking natural-language questions about a PDF knowledge base. It retrieves the most relevant document passages from a local Qdrant index, then asks Groq to generate an answer grounded only in those passages.

The application includes a responsive web interface, page citations, retrieved-source previews, and an evaluation harness for factual, multi-step, and out-of-scope questions.

## Features

- Modern responsive Knowora chat interface, including keyboard-friendly controls and mobile navigation.
- PDF extraction with PyMuPDF and overlapping text chunks.
- Local vector search using FastEmbed and Qdrant's embedded local storage.
- Grounded answers generated with Groq; the model is instructed to cite pages as `[p. N]` and refuse unsupported answers.
- Returned source previews with page number and similarity score.
- Evaluation script for retrieval hit rate, citation validity, answer correctness, and out-of-scope refusals.
- Docker and Docker Compose support.

## Architecture

```text
PDF → PyMuPDF extraction → chunking → FastEmbed vectors → local Qdrant
                                                         ↓
Browser → FastAPI POST /ask → vector retrieval → Groq → answer + sources
```

## Requirements

- Python 3.11 or newer
- A Groq API key
- Internet access for the first FastEmbed model download and for Groq requests

## Quick start

1. Create and activate a virtual environment.

   ```bash
   python -m venv .venv
   # PowerShell
   .\.venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install dependencies.

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root.

   ```env
   GROQ_API_KEY=your_groq_api_key
   # Optional
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

4. Put the source PDF at `data/document.pdf`, or set `PDF_PATH` to another file.

5. Build the local vector index.

   ```bash
   python -m app.ingest
   ```

6. Start the application.

   ```bash
   uvicorn app.main:app --reload --port 7860
   ```

Open [http://localhost:7860](http://localhost:7860).

## Docker

Docker builds the index while building the image. Ensure that `data/document.pdf` is present before building, and create a `.env` file containing `GROQ_API_KEY` before starting the service.

```bash
docker compose up --build
```

The application is available on [http://localhost:7860](http://localhost:7860).

## Deploy to Render with GitHub Actions

The repository includes a Render Blueprint in `render.yaml` and a CI/CD workflow in `.github/workflows/ci.yml`.

1. Push this repository to GitHub, using `main` as the production branch.
2. In Render, create a new **Blueprint** service from the repository. Render detects `render.yaml` and creates the Docker web service. The Blueprint sets `/health` as the health check and configures Render to use the container's `PORT` environment variable.
3. In the Render service's **Environment** settings, set `GROQ_API_KEY` to your Groq key. Do not commit it to the repository.
4. In the Render service's **Settings**, copy its Deploy Hook URL.
5. In GitHub, open **Settings → Secrets and variables → Actions** and add a repository secret named `RENDER_DEPLOY_HOOK_URL` containing that URL.
6. Push or merge to `main`. GitHub Actions runs the tests, builds the Docker image, and then invokes the protected Render deploy hook.

Pull requests run tests only. Deployments occur only after a successful push to `main`. The `production` environment in the workflow can optionally be protected in GitHub with required reviewers before Render is triggered.

`autoDeployTrigger` is set to `off` in the Blueprint to ensure that deployments are gated by the GitHub Actions checks rather than starting immediately when a commit is pushed.

## Configuration

All configuration is read from environment variables or `.env`.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required Groq API key. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq chat-completions model. |
| `PDF_PATH` | `data/document.pdf` | PDF to ingest. |
| `QDRANT_PATH` | `qdrant_data` | Local Qdrant storage directory. |
| `COLLECTION` | `docs` | Qdrant collection name. |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed embedding model. |
| `EMBED_DIM` | `384` | Embedding vector dimension; must match the embedding model. |
| `CHUNK_WORDS` | `250` | Maximum words per indexed chunk. |
| `CHUNK_OVERLAP` | `50` | Word overlap between chunks. |
| `TOP_K` | `5` | Number of passages retrieved for each question. |

For a French or multilingual PDF, use a compatible multilingual embedding model and keep `EMBED_DIM` aligned with that model.

## API

### Health check

```http
GET /health
```

Response:

```json
{"status":"ok"}
```

### Ask a question

```http
POST /ask
Content-Type: application/json

{"question":"What is the default scope of a Spring bean?"}
```

Response shape:

```json
{
  "answer": "The default scope is singleton: one shared instance [p. 5].",
  "sources": [
    {"page": 5, "snippet": "...", "score": 0.91}
  ]
}
```

Questions must be between 3 and 500 characters. If the provided excerpts do not contain the answer, the generation prompt instructs the model to return: `I could not find this information in the document.`

## Evaluation

Define evaluation cases in `eval/questions.json`. Each entry uses this shape:

```json
{
  "q": "What three annotations does @SpringBootApplication combine?",
  "expected_answer": "@Configuration, @EnableAutoConfiguration and @ComponentScan.",
  "expected_pages": [3],
  "type": "factual"
}
```

Supported test types are:

- `factual`: one direct fact from the document.
- `reasoning`: synthesis across one or more passages.
- `out_of_scope`: a question that should receive the refusal response.

Run the evaluator after ingestion:

```bash
python eval/run_eval.py
```

It calls Groq to generate the answer and uses Groq again as a semantic judge. It writes the report to `eval/results.md` and reports:

- `Hit@k`: whether an expected page was retrieved;
- answer correctness judged against the expected answer;
- valid page citations;
- correct refusals for out-of-scope questions.

Because this is an LLM-based evaluation, it requires a valid Groq key, makes external API calls, and may incur usage costs.

## Tests

Run the test suite with:

```bash
python -m pytest -q
```

## Project structure

```text
app/
  main.py          FastAPI routes and static frontend serving
  ingest.py        PDF extraction, chunking, embeddings, Qdrant indexing
  retrieve.py      Vector retrieval
  generate.py      Grounded Groq generation prompt and client
  config.py        Environment-based configuration
  static/          Knowora frontend and SVG brand mark
data/document.pdf  Default knowledge-base document
eval/              Evaluation data, runner, and generated report
tests/             API and chunking tests
```

## Current limitations

- The default workflow indexes one PDF at a time and recreates the configured collection during ingestion.
- There is no document upload, document list, server-side chat history, or authentication endpoint yet.
- The source metadata currently includes page, excerpt, and score; it does not retain document title or section metadata.
- A retrieval score threshold is not yet applied, so out-of-scope behavior relies on the generation prompt rather than a retrieval-confidence gate.

## Development notes

Do not commit `.env`, `qdrant_data/`, model caches, or virtual environments. They are already excluded in `.gitignore`.

