# app/main.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Vector store / embeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ──────────────────────────────────────────────────────────────────────────────
# Env & basic config
# ──────────────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# API keys (prefer GEMINI_API_KEY; allow GOOGLE_API_KEY fallback)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
if not GEMINI_API_KEY:
    # For production, fail fast is better than silent breakage
    raise RuntimeError("❌ Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Set it in Render/Vercel env.")

# Vector index path (persisted if your platform provides a disk)
INDEX_DIR = os.getenv("INDEX_DIR", str(ROOT_DIR / "storage" / "index"))

# Comma-separated origins, e.g. "http://localhost:5173,https://your-frontend.vercel.app"
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "FRONTEND_ORIGINS=https://policy-pilot.vercel.app,http://localhost:5173
")
ALLOWED_ORIGINS = [o.strip() for o in FRONTEND_ORIGINS.split(",") if o.strip()]

# ──────────────────────────────────────────────────────────────────────────────
# Embeddings & VectorStore
# ──────────────────────────────────────────────────────────────────────────────
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    # Google GenAI embeddings expect the fully-qualified model name
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_API_KEY,
    )

def get_vectorstore() -> Chroma:
    emb = get_embeddings()
    os.makedirs(INDEX_DIR, exist_ok=True)
    return Chroma(
        collection_name="policies",
        embedding_function=emb,
        persist_directory=INDEX_DIR,
    )

vs = get_vectorstore()

# ──────────────────────────────────────────────────────────────────────────────
# Clause extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

# Keywords grouped by topic to catch “Special Leaves”, “Referral”, “Probation”, etc.
PHRASE_GROUPS: Dict[str, List[str]] = {
    "special leaves": ["special leave", "special leaves", "marriage", "bereavement"],
    "probation": ["probation period", "probation"],
    "notice period": ["notice period"],
    "resignation": ["resignation process", "submit resignation"],
    "handover": ["handover", "knowledge transfer"],
    "full & final": ["full & final", "final settlement", "f&f"],
    "relieving": ["relieving", "experience letters"],
    "termination": ["termination for cause", "gross misconduct"],
    "communication": ["communication policy", "official channels", "response timelines", "social media"],
    "referral": ["referral bonus", "referral", "careers page", "bonus amounts"],
    "epf": ["epf", "provident fund"],
    "esi": ["esi", "state insurance"],
    "gratuity": ["gratuity"],
    "bonus act": ["bonus act", "statutory bonus"],
    "health insurance": ["health insurance"],
    "posh": ["posh", "safe workplace", "internal committee"],
    "leave": [
        "leave policy", "annual / earned leave", "earned leave", "annual leave",
        "sick leave", "casual leave", "compensatory off", "maternity", "paternity"
    ],
}

# Detect bullets / numbered items / paragraph breaks
BULLET_START = re.compile(r"^\s*(?:[\-\*•●]|(\d+(?:\.\d+)*)\s*[\.\)])\s*|^\s*$", re.UNICODE)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _split_lines(text: str) -> List[str]:
    return text.splitlines()

def _is_bullet_or_break(line: str) -> bool:
    return bool(BULLET_START.match(line))

def _expand_to_clause(lines: List[str], hit_index: int, max_up: int = 2, max_down: int = 6) -> Tuple[int, int, str]:
    """
    Expand around a matching line to include the whole bullet/paragraph.
    It climbs up to the previous bullet/break, and down until the next break.
    """
    n = len(lines)
    start = hit_index
    up_steps = 0
    while start > 0 and up_steps < max_up:
        if _is_bullet_or_break(lines[start - 1]):
            break
        start -= 1
        up_steps += 1

    end = hit_index
    down_steps = 0
    while end + 1 < n and down_steps < max_down:
        if _is_bullet_or_break(lines[end + 1]):
            break
        end += 1
        down_steps += 1

    clause = "\n".join(l.rstrip() for l in lines[start:end + 1]).strip()
    return start, end, clause

def _find_hits(question: str, lines: List[str]) -> List[int]:
    q = question.lower()

    # Prefer phrase groups if any phrase is present in the question
    for phrases in PHRASE_GROUPS.values():
        if any(p in q for p in phrases):
            hits = []
            for i, l in enumerate(lines):
                lower = l.lower()
                if any(p in lower for p in phrases):
                    hits.append(i)
            if hits:
                return hits

    # Fallback: overlap on keywords in the question
    words = [w for w in re.findall(r"[a-zA-Z]{3,}", q)]
    hits: List[int] = []
    for i, l in enumerate(lines):
        score = sum(1 for w in words if w in l.lower())
        if score >= max(1, len(words) // 4):
            hits.append(i)
    return hits

def extract_clauses(question: str, docs) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    From retrieved docs, extract full clauses that match the query.
    Returns (citations, policy_match_tags).
    """
    citations: List[Dict[str, Any]] = []
    tags: List[str] = []

    for d in docs:
        doc_id = d.metadata.get("doc_id") or d.metadata.get("source") or "document"
        page = d.metadata.get("page")
        content = d.page_content or ""
        lines = _split_lines(content)

        hit_indexes = _find_hits(question, lines)
        if not hit_indexes:
            continue

        # Expand around the first few hits (avoid returning too much)
        added = 0
        for hi in hit_indexes[:2]:
            s, e, clause = _expand_to_clause(lines, hi)
            if not clause:
                continue
            citations.append({
                "doc_id": doc_id,
                "section": None,
                "snippet": clause,
                "page": page
            })
            added += 1

        if added:
            tag = d.metadata.get("doc_id") or d.metadata.get("source")
            if tag and tag not in tags:
                tags.append(tag)

    return citations, tags

def compose_answer_from_citations(citations: List[Dict[str, Any]]) -> str:
    """
    Builds a readable answer by concatenating distinct clauses (first 2–3).
    Keeps bullets/paragraphs intact.
    """
    if not citations:
        return "I don’t have that in policy."

    # Deduplicate identical snippets (trimmed) and concatenate
    seen = set()
    parts: List[str] = []
    for c in citations[:3]:
        snip = c.get("snippet", "").strip()
        key = _norm(snip)
        if not snip or key in seen:
            continue
        seen.add(key)
        parts.append(snip)

    answer = "\n\n".join(parts).strip()
    return answer if answer else "I don’t have that in policy."

# ──────────────────────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────────────────────
class AskIn(BaseModel):
    question: str = Field(..., description="Natural language question")

class AskOut(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    policy_matches: List[str] = []
    confidence: str = "medium"
    follow_up_suggestions: List[str] = []

app = FastAPI(title="Company Policy Assistant (Clause Mode)", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "name": "Company Policy Assistant (Clause Mode)",
        "version": "2.0",
        "health": "/healthz",
        "ask": "POST /ask { question: string }",
        "index_dir": INDEX_DIR,
        "origins": ALLOWED_ORIGINS,
        "provider": "gemini",
    }

@app.get("/healthz")
def healthz():
    try:
        # Try to read collection count via chromadb collection
        count = None
        try:
            # LangChain's Chroma exposes underlying chromadb collection
            # count() exists on chromadb collection
            count = vs._collection.count()  # type: ignore[attr-defined]
        except Exception:
            # Fallback: attempt a small retrieval to ensure index is accessible
            _ = vs.as_retriever(search_type="mmr", search_kwargs={"k": 1})
        return {"ok": True, "model_provider": "gemini", "index_dir": INDEX_DIR, "chroma_count": count}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/ask", response_model=AskOut)
def ask(body: AskIn):
    try:
        q = (body.question or "").strip()
        if not q:
            return AskOut(answer="Please enter a question.", confidence="low")

        retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 6})
        docs = retriever.get_relevant_documents(q)

        citations, tags = extract_clauses(q, docs)
        if not citations:
            return AskOut(
                answer="I don’t have that in policy.",
                citations=[],
                policy_matches=[],
                confidence="low",
                follow_up_suggestions=["Please try again or contact HR."]
            )

        answer = compose_answer_from_citations(citations)

        # Heuristic confidence: more clauses == more confident
        conf = "high" if len(citations) >= 2 else "medium"

        # Simple follow-ups based on what was asked
        followups: List[str] = []
        lq = q.lower()
        if "leave" in lq:
            followups = [
                "Is there a carry-forward or encashment rule?",
                "Do sick leaves require a medical certificate?",
                "Can casual leave be clubbed with earned leave?"
            ]
        elif "referral" in lq:
            followups = [
                "What roles are eligible for referral?",
                "When is referral bonus paid?"
            ]
        elif "probation" in lq or "notice" in lq:
            followups = [
                "Does notice period differ after confirmation?",
                "What is the resignation submission process?"
            ]

        return AskOut(
            answer=answer,
            citations=citations,
            policy_matches=tags,
            confidence=conf,
            follow_up_suggestions=followups
        )
    except Exception as e:
        # Avoid leaking sensitive stack traces; keep message concise
        return AskOut(
            answer="I don’t have that in policy.",
            citations=[],
            policy_matches=[],
            confidence="low",
            follow_up_suggestions=[f"Error: {str(e)[:120]}"]
        )

# ──────────────────────────────────────────────────────────────────────────────
# Local run convenience (useful for testing on Windows)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD", "0") == "1"),
    )
