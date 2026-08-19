// Upload a PDF/JSON document, run a question list, or chat against the active doc.
import { useEffect, useMemo, useState } from "react";

const SAMPLE_QUESTIONS = [
  "Do you have formally defined criteria for notifying a client during an incident that might impact the security of their data or systems? What are your SLAs for notification?",
  "Is personal information transmitted, processed, stored, or disclosed to or retained by third parties? If yes, describe.",
  "Which cloud providers do you rely on?",
  "Please specify the primary data center location/region of the underlying cloud infrastructure used to host the service(s) as well as the backup location(s).",
  "Which of the following, if any, are performed as part of your monitoring process for the service: Application Performance Monitoring (APM), End User Monitoring (EUM), Digital Experience Monitoring (DEM)?",
];

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Request failed (${response.status})`);
  }
  return body;
}

function citationText(citation) {
  const bits = [citation.source];
  if (citation.page) bits.push(`p. ${citation.page}`);
  if (citation.json_path) bits.push(citation.json_path);
  return bits.filter(Boolean).join(" · ");
}

function autosize(el) {
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.max(el.scrollHeight, 52)}px`;
}

function questionsFromJson(text) {
  const payload = JSON.parse(text);
  const items = Array.isArray(payload)
    ? payload
    : payload.questions || payload.question;
  const list = (Array.isArray(items) ? items : [items])
    .map((item) => {
      if (typeof item === "string") return item.trim();
      if (item && typeof item === "object") {
        return String(item.question || item.q || item.text || "").trim();
      }
      return "";
    })
    .filter(Boolean);
  if (!list.length) {
    throw new Error("No questions were found in that JSON file.");
  }
  return list;
}

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [tab, setTab] = useState("batch");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [answering, setAnswering] = useState(false);
  const [questions, setQuestions] = useState(SAMPLE_QUESTIONS);
  const [results, setResults] = useState([]);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [dragging, setDragging] = useState(false);

  const selected = useMemo(
    () => documents.find((doc) => doc.document_id === selectedId),
    [documents, selectedId]
  );

  async function refreshDocuments(preferredId) {
    const data = await api("/api/v1/documents");
    setDocuments(data.documents || []);
    setSelectedId(preferredId || data.documents?.[0]?.document_id || "");
  }

  useEffect(() => {
    refreshDocuments().catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    setResults([]);
    setMessages([]);
  }, [selectedId]);

  async function uploadDocument(file) {
    if (!file) return;
    setError("");
    setIndexing(true);
    setStatus(`Indexing ${file.name}...`);
    try {
      const form = new FormData();
      form.append("document", file);
      const record = await api("/api/v1/documents", { method: "POST", body: form });
      await refreshDocuments(record.document_id);
      setStatus(`${record.filename} ready · ${record.chunks} chunks`);
    } catch (err) {
      setError(err.message);
      setStatus("");
    } finally {
      setIndexing(false);
    }
  }

  async function runBatch() {
    if (!selectedId) {
      setError("Upload a document first.");
      return;
    }
    const questionList = questions.map((line) => line.trim()).filter(Boolean);
    if (!questionList.length) {
      setError("Add at least one question.");
      return;
    }
    setError("");
    setAnswering(true);
    setStatus("Running grounded batch QA...");
    try {
      const blob = new Blob([JSON.stringify({ questions: questionList })], {
        type: "application/json",
      });
      const form = new FormData();
      form.append("document_id", selectedId);
      form.append("questions", blob, "questions.json");
      const data = await api("/api/v1/qa/batch", { method: "POST", body: form });
      setResults(data.results || []);
      setStatus(`Answered ${data.results.length} questions`);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnswering(false);
    }
  }

  async function sendChat(event) {
    event.preventDefault();
    if (!selectedId || !draft.trim()) return;
    const question = draft.trim();
    const history = messages.slice(-6).map((item) => ({
      role: item.role === "agent" ? "agent" : "user",
      text: item.text,
    }));
    setDraft("");
    setMessages((current) => [...current, { role: "user", text: question }]);
    setAnswering(true);
    setError("");
    try {
      const data = await api("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: selectedId, message: question, history }),
      });
      setMessages((current) => [
        ...current,
        {
          role: "agent",
          text: data.answer,
          citations: data.citations,
          supported: data.supported,
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setAnswering(false);
    }
  }

  function downloadResults() {
    const blob = new Blob([JSON.stringify({ document_id: selectedId, results }, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "qa-results.json";
    link.click();
    URL.revokeObjectURL(url);
  }

  function updateQuestion(index, value) {
    setQuestions((current) => current.map((item, i) => (i === index ? value : item)));
  }

  function addQuestion() {
    setQuestions((current) => [...current, ""]);
  }

  async function loadQuestionsFile(file) {
    if (!file) return;
    try {
      setQuestions(questionsFromJson(await file.text()));
      setStatus(`Loaded questions from ${file.name}`);
      setError("");
    } catch (err) {
      setError(err.message || "Could not parse questions JSON.");
    }
  }

  function removeQuestion(index) {
    setQuestions((current) => (current.length === 1 ? [""] : current.filter((_, i) => i !== index)));
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <div className="eyebrow">Document-grounded QA</div>
          <h1>Document Intelligence Agent</h1>
          <p className="lede">
            Upload a PDF or JSON source, then ask questions. Answers come only from the
            selected document, with citations and an explicit refusal when evidence is missing.
          </p>
        </div>
        <nav className="header-links" aria-label="Service links">
          <a href="/docs" target="_blank" rel="noreferrer">
            API docs
          </a>
          <a href="/health" target="_blank" rel="noreferrer">
            Health
          </a>
        </nav>
      </header>

      <div className="layout">
        <aside className="card sidebar">
          <div className="card-head">
            <h2>Documents</h2>
            <span className="count">{documents.length}</span>
          </div>
          <label
            className={`drop ${dragging ? "dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              uploadDocument(event.dataTransfer.files?.[0]);
            }}
          >
            <span className="drop-icon" aria-hidden="true">
              ⬆
            </span>
            <strong>Upload PDF or JSON</strong>
            <span>Drop a file here, or click to browse</span>
            <input
              className="hidden"
              type="file"
              accept=".pdf,.json,application/pdf,application/json"
              onChange={(event) => {
                uploadDocument(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
          </label>
          <div className="doc-list">
            {documents.length === 0 && (
              <p className="empty">No documents yet. Start with a sample JSON or PDF.</p>
            )}
            {documents.map((doc) => (
              <button
                key={doc.document_id}
                className={`doc-item ${doc.document_id === selectedId ? "active" : ""}`}
                onClick={() => setSelectedId(doc.document_id)}
              >
                <div className="doc-name">{doc.filename}</div>
                <small>
                  {doc.file_type.toUpperCase()} · {doc.chunks} chunks
                  {doc.pages ? ` · ${doc.pages} pages` : ""} · {doc.status}
                </small>
              </button>
            ))}
          </div>
          {status && <p className="status">{status}</p>}
          {error && <p className="status error">{error}</p>}
        </aside>

        <main className="card workspace">
          <div className="workspace-toolbar">
            <div className="tabs">
              <button className={tab === "batch" ? "active" : ""} onClick={() => setTab("batch")}>
                Batch questions
              </button>
              <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>
                Chat
              </button>
            </div>
            <p className="active-doc">
              Active document:{" "}
              <strong>{selected ? selected.filename : "none selected"}</strong>
            </p>
          </div>

          {tab === "batch" ? (
            <div className="batch">
              <section className="question-editor">
                {questions.map((question, index) => (
                  <div className="question-row" key={index}>
                    <span className="q-index">{index + 1}</span>
                    <textarea
                      rows={2}
                      value={question}
                      ref={autosize}
                      onChange={(event) => {
                        updateQuestion(index, event.target.value);
                        autosize(event.target);
                      }}
                      placeholder="Write one question"
                    />
                    <button
                      type="button"
                      className="icon-btn"
                      onClick={() => removeQuestion(index)}
                      aria-label={`Remove question ${index + 1}`}
                    >
                      ×
                    </button>
                  </div>
                ))}
                <div className="row actions">
                  <button type="button" className="ghost" onClick={addQuestion}>
                    Add question
                  </button>
                  <label className="ghost file-btn">
                    Upload questions JSON
                    <input
                      className="hidden"
                      type="file"
                      accept=".json,application/json"
                      onChange={(event) => {
                        loadQuestionsFile(event.target.files?.[0]);
                        event.target.value = "";
                      }}
                    />
                  </label>
                  <button className="primary" disabled={indexing || answering} onClick={runBatch}>
                    {answering ? "Running..." : indexing ? "Indexing..." : "Run questions"}
                  </button>
                  <button className="ghost" disabled={!results.length} onClick={downloadResults}>
                    Download JSON
                  </button>
                </div>
              </section>

              <section className="results">
                <h2 className="results-title">Answers</h2>
                {results.length === 0 ? (
                  <div className="empty-results">
                    <strong>No answers yet</strong>
                    <p>Upload a document, then run the question list to see grounded answers and citations.</p>
                  </div>
                ) : (
                  results.map((item) => (
                    <article className="result-card" key={item.question}>
                      <h3>{item.question}</h3>
                      <p className="answer">{item.answer}</p>
                      <div className="result-meta">
                        <span className={`badge ${item.supported ? "yes" : "no"}`}>
                          {item.supported ? "supported" : "not found"}
                        </span>
                        <span className="cite">
                          {(item.citations || []).map(citationText).join("; ") || "No citation"}
                        </span>
                      </div>
                    </article>
                  ))
                )}
              </section>
            </div>
          ) : (
            <div className="chat-panel">
              <div className="chat">
                {messages.length === 0 && (
                  <div className="empty-results">
                    <strong>Ask a follow-up</strong>
                    <p>
                      History resolves references like “what region?”, then the answer is grounded
                      only in the selected document.
                    </p>
                  </div>
                )}
                {messages.map((message, index) => (
                  <div key={index} className={`bubble ${message.role}`}>
                    <div>{message.text}</div>
                    {message.citations?.length ? (
                      <div className="cite">{message.citations.map(citationText).join("; ")}</div>
                    ) : null}
                  </div>
                ))}
              </div>
              <form className="composer" onSubmit={sendChat}>
                <input
                  type="text"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder="Ask a question about the selected document..."
                />
                <button className="primary" disabled={indexing || answering}>
                  Send
                </button>
              </form>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
