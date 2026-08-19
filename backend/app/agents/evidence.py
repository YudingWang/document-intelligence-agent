"""Cheap evidence gate used before calling gpt-4o-mini.

The grader is lexical on purpose: another model call would cancel the cost
saving of skipping generation when retrieved text is clearly off-topic.
"""

from __future__ import annotations

import re

from app.models.domain import RetrievedChunk

# Function words that should not count as topical overlap.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "part",
    "please",
    "than",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "what",
    "which",
    "who",
    "with",
    "you",
    "your",
}

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def evidence_is_sufficient(question: str, chunks: list[RetrievedChunk]) -> bool:
    """Return True when retrieved chunks look related enough to spend a generation call."""
    if not chunks:
        return False
    terms = content_terms(question)
    if not terms:
        return True
    blob = " ".join(chunk.text.lower() for chunk in chunks)
    overlap = sum(1 for term in terms if term in blob)
    required = 1 if len(terms) < 6 else 2
    return overlap >= required


def content_terms(question: str) -> set[str]:
    """Topic tokens from a question: content words, acronyms, and simple plurals."""
    terms: set[str] = set()
    for raw in _TOKEN.findall(question):
        token = raw.lower()
        if token in _STOPWORDS or len(token) < 3:
            continue
        terms.add(token)
        if token.endswith("s") and len(token) > 4:
            terms.add(token[:-1])
    return terms
