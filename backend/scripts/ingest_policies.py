# scripts/ingest_policies.py
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Vector store + embeddings (Gemini)
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# PDF loaders (we'll try the best, then fall back)
try:
    from langchain_community.document_loaders import UnstructuredPDFLoader  # best for bullets
    HAVE_UNSTRUCTURED = True
except Exception:
    HAVE_UNSTRUCTURED = False

from langchain_community.document_loaders import (
    PyPDFPlumberLoader,   # good accuracy
    PyPDFLoader,          # basic
    Docx2txtLoader,
    TextLoader,
)

# ---- config ----
ROOT = Path(__file__).resolve().parents[1]                    # .../backend
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH)

INDEX_DIR = os.getenv("INDEX_DIR", str((ROOT.parent / "storage" / "index")))
GOOGLE_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_KEY:
    raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in .env")

# small chunks so each bullet is its own piece
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def load_one_file(path: Path) -> List[Document]:
    """Load one file with the best available loader."""
    meta = {"doc_id": path.name, "source": str(path)}
    docs: List[Document] = []

    if path.suffix.lower() == ".pdf":
        # try the best loaders in order
        tried = []
        if HAVE_UNSTRUCTURED:
            try:
                docs = UnstructuredPDFLoader(str(path)).load()
                tried.append("UnstructuredPDFLoader ✓")
            except Exception:
                tried.append("UnstructuredPDFLoader ✗")
        if not docs:
            try:
                docs = PyPDFPlumberLoader(str(path)).load()
                tried.append("PyPDFPlumberLoader ✓")
            except Exception:
                tried.append("PyPDFPlumberLoader ✗")
        if not docs:
            docs = PyPDFLoader(str(path)).load()
            tried.append("PyPDFLoader ✓")
        # print which one worked (useful when debugging)
        print(f"[{path.name}] loader chain: {', '.join(tried)}")
    elif path.suffix.lower() == ".docx":
        docs = Docx2txtLoader(str(path)).load()
    elif path.suffix.lower() in {".txt", ".md"}:
        docs = TextLoader(str(path), encoding="utf-8").load()
    else:
        return []

    for d in docs:
        d.metadata.update(meta)

    return docs


def main(folder: str):
    in_dir = Path(folder).resolve()
    if not in_dir.exists():
        print(f"Folder not found: {in_dir}")
        return

    all_raw: List[Document] = []
    for p in sorted(in_dir.glob("*")):
        if p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"}:
            all_raw.extend(load_one_file(p))

    if not all_raw:
        print("No supported files found.")
        return

    chunks = SPLITTER.split_documents(all_raw)

    emb = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_KEY
    )

    vs = Chroma(
        collection_name="policies",
        embedding_function=emb,
        persist_directory=INDEX_DIR,
    )

    vs.add_documents(chunks)
    # Note: Chroma 0.4+ persists automatically, but this is harmless:
    vs.persist()

    print(f"Ingested files: {len(set(d.metadata['doc_id'] for d in all_raw))}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Index dir: {INDEX_DIR}")


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "app" / "policies")
    main(folder)
