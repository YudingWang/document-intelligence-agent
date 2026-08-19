# Document Intelligence Agent

Document-grounded question answering for PDF and JSON files.

**Repository:** [github.com/YudingWang/document-intelligence-agent](https://github.com/YudingWang/document-intelligence-agent)

Upload a source document and a JSON list of questions. The service indexes the document, retrieves evidence, and answers with **OpenAI `gpt-4o-mini` only**. If the document does not support an answer, it refuses instead of using general knowledge.

## What it does

1. Accept two inputs: a source document (`PDF` or `JSON`) and a questions file (`JSON`).
2. Parse, chunk, embed, and index the document.
3. Answer each question from retrieved evidence only (batch or chat), across one or more selected documents.
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

You need **Python 3.11+**, **Node 18+**, and an **OpenAI API key**. The sample PDF is already in the repo; generating it is optional.

### 1. Clone

```bash
git clone https://github.com/YudingWang/document-intelligence-agent.git
cd document-intelligence-agent
```

### 2. Configure the API key

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

### 3. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..
```

On Windows PowerShell, activate the venv with `.venv\Scripts\Activate.ps1`.

Shortcut if you prefer Make: `make install`.

### 4. Run

Terminal 1 — API (use the project venv, not a global/conda `uvicorn`):

```bash
cd backend
PYTHONPATH=. ../.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 — UI:

```bash
cd frontend
npm run dev
```

- UI: http://localhost:5173
- API docs: http://localhost:8000/docs (also linked from the UI header)
- Health: http://localhost:8000/health

Or: `make api` in one terminal and `make web` in another.

### 5. Try it

**UI.** Open http://localhost:5173. A **Sample** document (`sample_eval.json`) is already in the list with the 19 sheet questions loaded. Click documents on the left to include more than one in the search set, then Run questions. To try the small vendor policy, upload `sample_data/vendor_security.json` and select it (alone or together with Sample). Chat follow-ups such as “What region?” use recent history only to resolve the question, then retrieve from the selected documents.

**One-shot API** (assignment contract — document + questions in one request):

```bash
curl -s -X POST http://localhost:8000/api/v1/qa \
  -F "document=@sample_data/vendor_security.json" \
  -F "questions=@sample_data/questions.json"
```

**CLI:**

```bash
cd backend
PYTHONPATH=. ../.venv/bin/python -m app.cli \
  --document ../sample_data/vendor_security.json \
  --questions ../sample_data/questions.json \
  --out ../data/out.json
```

A supported answer includes citations. A fact that is not in the file returns `I could not find this information in the provided document.`

### Docker

Same clone and `.env` as above, then from the repo root:

```bash
docker compose up --build
```

UI: http://localhost:5173 · API: http://localhost:8000

## Overall architecture

Ingestion is deterministic (no LLM). Question answering is a bounded LangGraph agent. Evidence is graded **before** generation so a weak first retrieve does not spend a `gpt-4o-mini` completion.

```
                    ┌──────────────────────────────────────────────┐
                    │                 React UI                     │
                    │  upload · questions JSON · batch · chat      │
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
              ┌──────────────▼──────────┐   ┌──────▼───────────────┐
              │ DocumentCatalog (JSON)  │   │ DocumentTools        │
              │ VectorIndex (Chroma)    │◄──│ search / section     │
              └─────────────────────────┘   └──────┬───────────────┘
                                             ┌─────▼──────┐
                                             │ gpt-4o-mini │
                                             │ rewrite /   │
                                             │ generate    │
                                             └─────────────┘
```

### Agent loop (per question)

```
retrieve  (+ neighboring chunks; merge into evidence set)
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
        retrieve again              ← merged with first hits
           │
           ▼
        generate ──► END            ← at most one answer call
```

`MAX_RETRIEVAL_ATTEMPTS = 2`, then the graph always generates or returns the grounded refusal. Generation is never called twice for the same question.

- **PDF**: PyMuPDF, page numbers kept on every chunk.
- **JSON**: flatten to path/value leaves, then chunk by parent object. Citations are JSONPath section pointers such as `$.security.cloud_providers`.
- **Vector DB**: Chroma on disk. Tests use an in-memory store.
- **Grounding**: unsupported answers are forced to `I could not find this information in the provided document.`
- **Isolation**: retrieval always filters by the selected `document_id`s; other indexed files are not searched.
- **Multi-document**: the agent retrieves from every selected file, then merges evidence and cites the source filename.
- **Batch**: independent questions run concurrently (default 3).
- **Chat**: optional `history` is used only to rewrite a standalone question; answers still come from the selected documents.
- **Tools**: the graph calls `search_document` / `read_document_section` / `get_document_info` instead of the vector store directly.

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
  agents/prompts.py        system / rewrite / follow-up prompts
  tools/documents.py       search_document, read_document_section, get_document_info
  loaders/                 PDF, JSON, questions, chunking
  services/                ingestion and batch QA
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
            → DocumentTools.search_documents (+ neighboring section)
            → evidence_is_sufficient
            → OpenAILLMClient.rewrite_query   (only if evidence is weak)
            → merge/dedupe retrieved chunks
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
| `backend/app/tools/documents.py` | Document tools used by the graph |
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
| `DELETE` | `/api/v1/documents/{id}` | Remove a document and its vectors |
| `POST` | `/api/v1/qa/batch` | `document_id` and/or `document_ids` + questions file |
| `POST` | `/api/v1/chat` | JSON `{ "document_ids": [...], "message", "history?" }` |
| `GET` | `/health` | Liveness |

Questions JSON may be `{ "questions": ["..."] }` or a top-level array of strings / `{ "question": "..." }` objects.

Chat accepts `document_id` or `document_ids`. `history` is an optional list of `{ "role": "user"|"agent", "text": "..." }` (max 6). It resolves follow-ups such as “What region?”; it is not used as evidence.

Batch `document_ids` may be a comma-separated list. The assignment one-shot `POST /api/v1/qa` still takes a single uploaded file.

### CLI

```bash
cd backend
PYTHONPATH=. ../.venv/bin/python -m app.cli \
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

Coverage: JSON/PDF loaders, question parsing, evidence gating, document-id filtered retrieval, multi-document retrieve, retry evidence merge, filename sanitization, chat history rewrite, agent retry/refusal (one generate), and API upload/QA/error paths.

## Sample files

- `sample_data/sample_eval.json` — evaluation set converted from `Sample JSON.xlsx` (19 Q&A entries)
- `sample_data/sample_eval_questions.json` — the 19 questions from that sheet
- `sample_data/questions.json` — the original 5 challenge questions
- `sample_data/vendor_security.json` — small vendor policy JSON
- `sample_data/vendor_security.pdf` — `python scripts/generate_sample_pdf.py`

The sample eval JSON is **indexed on API startup** and shows up in the UI as **Sample**. Do not delete it unless you mean to; restarting the API restores it. Click documents to add or remove them from the search set, then Run questions. You can still upload any other PDF/JSON.

There is no remote sample-PDF URL in this repo. Use the local files above, or upload any PDF/JSON you have on disk (for example a SOC 2 report if the challenge gave you one). If a fact is not in the selected file, the agent should refuse rather than invent.

## Design notes

- **Evaluate before generate.** Weak first-pass retrieval rewrites the query instead of paying for an answer completion.
- **Merge retries.** A second retrieve adds to the first hit set instead of replacing it.
- **Chat history is query resolution, not memory-as-evidence.** Follow-ups are rewritten, then grounded again in the selected documents.
- **Multi-select in the UI.** Retrieval is the union of hits from each selected document, still isolated from everything else.
- **One-shot + two-step APIs.** The brief asks for two files in; the UI is better with upload-then-ask, including a questions JSON upload.
- **Valid JSON output** with `supported` and `citations`.
- **Cost.** `text-embedding-3-small`, max two retrievals, one generate, no GPT-4 / 16k models.
- **No keys in git.** `.env` is ignored; `.env.example` has a placeholder only.
- **Not in MVP:** hybrid BM25, cross-encoders, SSE, multi-tenant auth, Kafka/Redis, Kubernetes.

## License

Private take-home submission. Not licensed for production use.
