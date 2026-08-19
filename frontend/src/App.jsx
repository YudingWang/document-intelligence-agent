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

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [tab, setTab] = useState("batch");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [questionsText, setQuestionsText] = useState(SAMPLE_QUESTIONS.join("\n"));
  const [results, setResults] = useState([]);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");

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

  async function uploadDocument(file) {
    setError("");
    setBusy(true);
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
      setBusy(false);
    }
  }

  async function runBatch() {
    if (!selectedId) {
      setError("Upload a document first.");
      return;
    }
    const questions = questionsText.split("\n").map((line) => line.trim()).filter(Boolean);
    if (!questions.length) {
      setError("Add at least one question.");
      return;
    }
    setError("");
    setBusy(true);
    setStatus("Running grounded batch QA...");
    try {
      const blob = new Blob([JSON.stringify({ questions })], { type: "application/json" });
      const form = new FormData();
      form.append("document_id", selectedId);
      form.append("questions", blob, "questions.json");
      const data = await api("/api/v1/qa/batch", { method: "POST", body: form });
      setResults(data.results || []);
      setStatus(`Answered ${data.results.length} questions`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function sendChat(event) {
    event.preventDefault();
    if (!selectedId || !draft.trim()) return;
    const question = draft.trim();
    setDraft("");
    setMessages((current) => [...current, { role: "user", text: question }]);
    setBusy(true);
    setError("");
    try {
      const data = await api("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: selectedId, message: question }),
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
      setBusy(false);
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
      </header>

      <div className="layout">
        <aside className="card">
          <h2>Documents</h2>
          <label className="drop">
            <strong>Upload PDF or JSON</strong>
            Drop a file or click to browse
            <input
              className="hidden"
              type="file"
              accept=".pdf,.json,application/pdf,application/json"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) uploadDocument(file);
                event.target.value = "";
              }}
            />
          </label>
          <div className="doc-list">
            {documents.length === 0 && <p className="status">No documents yet.</p>}
            {documents.map((doc) => (
              <button
                key={doc.document_id}
                className={`doc-item ${doc.document_id === selectedId ? "active" : ""}`}
                onClick={() => setSelectedId(doc.document_id)}
              >
                <div>{doc.filename}</div>
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

        <main className="card">
          <div className="tabs">
            <button className={tab === "batch" ? "active" : ""} onClick={() => setTab("batch")}>
              Batch questions
            </button>
            <button className={tab === "chat" ? "active" : ""} onClick={() => setTab("chat")}>
              Chat
            </button>
          </div>

          {tab === "batch" ? (
            <>
              <p className="status">
                Active document: {selected ? selected.filename : "none selected"}
              </p>
              <textarea
                value={questionsText}
                onChange={(event) => setQuestionsText(event.target.value)}
                placeholder="One question per line"
              />
              <div className="row" style={{ marginTop: 12 }}>
                <button className="primary" disabled={busy} onClick={runBatch}>
                  {busy ? "Running..." : "Run questions"}
                </button>
                <button className="ghost" disabled={!results.length} onClick={downloadResults}>
                  Download JSON
                </button>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Question</th>
                      <th>Answer</th>
                      <th>Source</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((item) => (
                      <tr key={item.question}>
                        <td>{item.question}</td>
                        <td>
                          <div>{item.answer}</div>
                          <span className={`badge ${item.supported ? "yes" : "no"}`}>
                            {item.supported ? "supported" : "not found"}
                          </span>
                        </td>
                        <td className="cite">
                          {(item.citations || []).map(citationText).join("; ") || "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <>
              <div className="chat">
                {messages.length === 0 && (
                  <p className="status">Ask a follow-up against the selected document.</p>
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
              <form className="row" style={{ marginTop: 16 }} onSubmit={sendChat}>
                <input
                  type="text"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder="Ask a question..."
                />
                <button className="primary" disabled={busy}>
                  Send
                </button>
              </form>
            </>
          )}
        </main>
      </div>
    </div>
  );
}
