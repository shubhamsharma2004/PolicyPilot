from __future__ import annotations
import os, re
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from app.config import INDEX_DIR, OPENAI_API_KEY
from langchain.schema.runnable import RunnableLambda

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Add it to .env")

# Embeddings & LLM (configurable)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=OPENAI_API_KEY)

# Create or connect vector DB
def get_vectorstore() -> Chroma:
    os.makedirs(INDEX_DIR, exist_ok=True)
    return Chroma(collection_name="policies", embedding_function=embeddings, persist_directory=INDEX_DIR)

# -------- Ingestion --------
def load_one(path: str) -> List[Document]:
    path_lower = path.lower()
    meta = {
        "doc_id": os.path.basename(path),
        "title": os.path.basename(path),
        "source": path,
        "category": guess_category_from_filename(path_lower),
    }
    if path_lower.endswith(".pdf"):
        docs = PyPDFLoader(path).load()
    elif path_lower.endswith(".docx"):
        docs = Docx2txtLoader(path).load()
    elif path_lower.endswith(".txt") or path_lower.endswith(".md"):
        docs = TextLoader(path, encoding="utf-8").load()
    else:
        return []

    # Attach metadata to each page/section
    for d in docs:
        d.metadata.update(meta)
    return docs

def guess_category_from_filename(name: str) -> str:
    if "leave" in name: return "Leave"
    if "exit" in name: return "Exit"
    if "communication" in name: return "Communication"
    if "referral" in name: return "Referral Bonus"
    if "benefit" in name or "benefits" in name: return "Employee Benefits"
    return "General"

def sectionize(text: str) -> str:
    # Lightweight normalization: collapse spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_docs(raw_docs: List[Document]) -> List[Document]:
    # Normalize & split
    normed = []
    for d in raw_docs:
        normed.append(Document(page_content=sectionize(d.page_content), metadata=d.metadata))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(normed)
    return chunks

def ingest_folder(folder: str) -> Tuple[int, int]:
    vs = get_vectorstore()
    all_raw: List[Document] = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".pdf", ".docx", ".txt", ".md")):
                path = os.path.join(root, f)
                all_raw.extend(load_one(path))
    chunks = chunk_docs(all_raw)
    if not chunks:
        return 0, 0
    vs.add_documents(chunks)
    vs.persist()
    return len(all_raw), len(chunks)

# -------- Query --------
SYSTEM_PROMPT = """You are a precise HR policy assistant.
Follow STRICT rules:
- Answer ONLY using the provided policy context.
- Cite at least one source with doc_id, (section if obvious), and page if available.
- If not clearly in context, reply: "I don’t have that in policy." and suggest contacting HR.
- Keep answers under 200 words unless the question says 'details'.
- Never invent numbers or dates. Prefer exact language and effective dates.
Return JSON with keys:
answer, citations[], policy_matches[], confidence, follow_up_suggestions[], metadata
"""

def build_user_prompt(question: str, context_blocks: List[str]) -> str:
    ctx = "\n\n---\n\n".join(context_blocks)
    return f"""Question: {question}

Context (use this ONLY):
{ctx}

Respond in JSON with the schema described in the system prompt.
"""

def retrieve_context(question: str, k: int, filters: Dict[str, Any] | None) -> Tuple[List[Document], List[str]]:
    vs = get_vectorstore()
    search_kwargs = {"k": k}
    retriever = vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    # Simple server-side filtering by category if provided
    if filters and "policy" in filters:
        # Retrieve more then filter
        docs = retriever.get_relevant_documents(question)
        policy = str(filters["policy"]).strip().lower()
        docs = [d for d in docs if d.metadata.get("category", "").lower().startswith(policy.lower())]
    else:
        docs = retriever.get_relevant_documents(question)

    blocks = []
    for d in docs:
        # Add a short header for citation readability
        head = f"[doc_id: {d.metadata.get('doc_id','?')} | category: {d.metadata.get('category','?')} | page: {d.metadata.get('page', '?')}]"
        blocks.append(f"{head}\n{d.page_content}")
    return docs, blocks

def to_citations(docs: List[Document]) -> List[Dict[str, Any]]:
    out = []
    for d in docs[:3]:  # at most top-3 to keep output tidy
        out.append({
            "doc_id": d.metadata.get("doc_id"),
            "section": d.metadata.get("section"),
            "page": d.metadata.get("page"),
            "snippet": d.page_content[:240] + ("..." if len(d.page_content) > 240 else "")
        })
    return out

def policy_matches_from(docs: List[Document]) -> List[str]:
    seen = []
    for d in docs:
        c = d.metadata.get("category", "General")
        if c not in seen:
            seen.append(c)
    return seen

def ask_policies(question: str, k: int = 5, filters: Dict[str, Any] | None = None) -> Dict[str, Any]:
    docs, blocks = retrieve_context(question, k, filters)
    user_prompt = build_user_prompt(question, blocks)

    # Compose messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # Use LC ChatOpenAI directly
    res = llm.invoke(messages)  # returns AIMessage with .content
    raw = res.content

    # Try to coerce to JSON safely
    import json
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # fallback minimal wrapper
        data = {"answer": raw, "citations": [], "policy_matches": [], "confidence": "low",
                "follow_up_suggestions": [], "metadata": {}}

    # Enrich with citations & matches if model forgot
    if not data.get("citations"):
        data["citations"] = to_citations(docs)
    if not data.get("policy_matches"):
        data["policy_matches"] = policy_matches_from(docs)
    data.setdefault("metadata", {})
    data["metadata"].update({"retriever_k": k, "chunks_used": len(docs), "model": "gpt-4o-mini"})
    return data
