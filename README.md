# Document Intelligence Agent

Document-grounded question answering for PDF and JSON files.

Upload a source document and a JSON list of questions. The service indexes the document, retrieves evidence, and answers with **OpenAI `gpt-4o-mini` only**. If the document does not support an answer, it refuses instead of using general knowledge.

## What it does

1. Accept two inputs: a source document (`PDF` or `JSON`) and a questions file (`JSON`).
2. Parse, chunk, embed, and index the document.
3. Answer each question from retrieved evidence only.
4. Return structured JSON pairing every question with an answer, plus citations.

The assignment example `[“question”: ”answer”]` is not valid JSON. The API returns a list of objects:

```json
{
  "document_id": "doc_abc123",
  "filename": "vendor_security.pdf",
  "results": [
    {
      "question": "Which cloud providers do you rely on?",
      "answer": "Acme Cloud relies on AWS as the primary provider and GCP for analytics.",
      "supported": true,
      "citations": [
        { "source": "vendor_security.pdf", "page": 1 }
      ]
    }
  ]
}
```

## Quick start

### 1. Configure the API key

```bash
cp .env.example .env
```

Put your key in `.env`. Never commit that file.

```
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Use `gpt-4o-mini` only. The key budget is small: cheap embeddings, temperature `0`, at most two retrievals, and at most one answer generation per question.

### 2. Install

Python 3.11+ and Node 18+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cd frontend && npm install && cd ..
python scripts/generate_sample_pdf.py
```

### 3. Run

Terminal 1:

```bash
cd backend
PYTHONPATH=. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
cd frontend
npm run dev
```

UI: http://localhost:5173 · API docs: http://localhost:8000/docs

### Docker

```bash
docker compose up --build
```

## Overall architecture

Ingestion is deterministic (no LLM). Question answering is a bounded LangGraph agent. Evidence is graded **before** generation so a weak first retrieve does not spend a `gpt-4o-mini` completion.

```
                    ┌──────────────────────────────────────────────┐
                    │                 React UI                     │
                    │     upload · batch questions · chat          │
                    └──────────────────────┬───────────────────────┘
                                           │ HTTP
                    ┌──────────────────────▼───────────────────────┐
                    │                   FastAPI                    │
                    │   /documents  /qa  /qa/batch  /chat  /health │
                    └──────────────┬───────────────┬───────────────┘
                                   │               │
                     ┌─────────────▼──┐     ┌──────▼──────┐
                     │ IngestionService│     │  QAService  │
                     │ parse → chunk   │     │  semaphore  │
                     │ embed → index   │     │  per question│
                     └───────┬────────┘     └──────┬───────┘
                             │                     │
              ┌──────────────▼──────────┐   ┌──────▼───────────────┐
              │ PDF / JSON loaders      │   │ DocumentQAAgent      │
              │ questions parser        │   │ (LangGraph)          │
              └──────────────┬──────────┘   └──────┬───────────────┘
                             │                     │
              ┌──────────────▼──────────┐          │
              │ DocumentCatalog (JSON)  │          │
              │ VectorIndex (Chroma)    │◄─────────┘
              └─────────────────────────┘          │
                                             ┌─────▼──────┐
                                             │ gpt-4o-mini │
                                             │ rewrite /   │
                                             │ generate    │
                                             └─────────────┘
```

### Agent loop (per question)

```
retrieve
    │
    ▼
evaluate evidence?          ← lexical check, no LLM
    ├─ yes ──────────────────────────► generate ──► END
    └─ no, attempts < 2
           │
           ▼
        rewrite query               ← short LLM call
           │
           ▼
        retrieve again
           │
           ▼
        generate ──► END            ← at most one answer call
```

`MAX_RETRIEVAL_ATTEMPTS = 2`, then the graph always generates or returns the grounded refusal. Generation is never called twice for the same question.

- **PDF**: PyMuPDF, page numbers kept on every chunk.
- **JSON**: recursive flatten; citations use paths such as `$.security.cloud_providers[0]`.
- **Vector DB**: Chroma on disk. Tests use an in-memory store.
- **Grounding**: unsupported answers are forced to `I could not find this information in the provided document.`
- **Isolation**: retrieval always filters by `document_id`.
- **Batch**: independent questions run concurrently (default 3).

## Repository layout and how files relate

```
backend/app/                 HTTP + agent
  main.py                  FastAPI factory (uvicorn target)
  cli.py                   same pipeline from the command line
  runtime.py               builds the shared AppContainer
  api/routes/              thin HTTP handlers
  api/uploads.py           multipart size checks
  agents/graph.py          LangGraph retrieve → evaluate → rewrite → generate
  agents/evidence.py       cheap “is this evidence related?” gate
  agents/llm.py            gpt-4o-mini structured answer + query rewrite
  agents/prompts.py        system / rewrite prompts
  loaders/                 PDF, JSON, questions, chunking
  services/                ingestion, retrieval, batch QA
  repositories/            catalog JSON, embeddings, Chroma / memory index
  models/                  domain objects and API schemas
  core/                    settings, errors, logging, API-key check
frontend/                  React + Vite UI (proxies /api to the backend)
sample_data/               sample questions + JSON/PDF policy
scripts/                   generate the sample PDF
```

Call chain for one-shot `POST /api/v1/qa`:

```
routes/qa.py
  → IngestionService.ingest
       → loaders (pdf/json) → chunking → VectorIndex.add_chunks
       → DocumentCatalog.add
  → QAService.answer_batch
       → DocumentQAAgent.answer
            → RetrievalService.search
            → evidence_is_sufficient
            → OpenAILLMClient.rewrite_query   (only if evidence is weak)
            → OpenAILLMClient.generate_answer (at most once)
```

| Path | Role |
| --- | --- |
| `backend/app/main.py` | App factory, CORS, request ids, error handler |
| `backend/app/runtime.py` | Constructs catalog, vector store, agent, services |
| `backend/app/api/routes/*.py` | HTTP surface; no business logic beyond parsing uploads |
| `backend/app/services/ingestion.py` | File type, parse, chunk, embed, persist |
| `backend/app/services/qa.py` | One question or a bounded concurrent batch |
| `backend/app/agents/graph.py` | Agent control flow |
| `backend/app/agents/evidence.py` | Skip generation when retrieved text is off-topic |
| `backend/app/repositories/vector_index.py` | Chroma in prod, `InMemoryVectorStore` in tests |

## API

### One-shot (assignment contract)

`POST /api/v1/qa` — multipart form:

- `document`: PDF or JSON source
- `questions`: JSON question list

```bash
curl -s -X POST http://localhost:8000/api/v1/qa \
  -F "document=@sample_data/vendor_security.json" \
  -F "questions=@sample_data/questions.json"
```

### Split flow (UI / chat)

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/documents` | Upload and index a PDF/JSON document |
| `GET` | `/api/v1/documents` | List indexed documents |
| `GET` | `/api/v1/documents/{id}` | Document metadata |
| `POST` | `/api/v1/qa/batch` | `document_id` + questions file |
| `POST` | `/api/v1/chat` | JSON `{ "document_id", "message" }` |
| `GET` | `/health` | Liveness |

Questions JSON may be `{ "questions": ["..."] }` or a top-level array of strings / `{ "question": "..." }` objects.

### CLI

```bash
cd backend
PYTHONPATH=. python -m app.cli \
  --document ../sample_data/vendor_security.json \
  --questions ../sample_data/questions.json \
  --out ../data/out.json
```

## Tests

Tests do not call OpenAI. They use deterministic fake embeddings and a stub LLM.

```bash
source .venv/bin/activate
pytest -q
```

Coverage: JSON/PDF loaders, question parsing, evidence gating, document-id filtered retrieval, agent retry/refusal (one generate), and API upload/QA/error paths.

## Sample files

- `sample_data/questions.json` — the challenge questions
- `sample_data/vendor_security.json` — structured policy that answers those questions
- `sample_data/vendor_security.pdf` — `python scripts/generate_sample_pdf.py`

You can also upload the public SOC 2 PDF from the challenge. If a fact is not in the file, the agent should refuse rather than invent.

## Design notes

- **Evaluate before generate.** Weak first-pass retrieval rewrites the query instead of paying for an answer completion.
- **One-shot + two-step APIs.** The brief asks for two files in; the UI is better with upload-then-ask.
- **Valid JSON output** with `supported` and `citations`.
- **Cost.** `text-embedding-3-small`, max two retrievals, one generate, no GPT-4 / 16k models.
- **No keys in git.** `.env` is ignored; `.env.example` has a placeholder only.
- **Not in MVP:** hybrid BM25, cross-encoders, SSE, multi-tenant auth, Kafka/Redis, Kubernetes.

## License

Private take-home submission. Not licensed for production use.
