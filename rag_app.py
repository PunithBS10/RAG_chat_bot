# rag_app.py
import os
import glob
import json
import numpy as np
import requests
import streamlit as st
import faiss
import fitz  # PyMuPDF
from io import BytesIO
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -----------------------
# Env / secrets
# -----------------------
load_dotenv()

def get_secret(name: str, default: str | None = None):
    """Prefer Streamlit Cloud secrets; fallback to env vars."""
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

# -----------------------
# Config
# -----------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"
DOCS_DIR = "docs"          # used only when not in ephemeral mode
INDEX_DIR = "vector_store" # used only when not in ephemeral mode
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K_DEFAULT = 5

DEFAULT_SYSTEM_PROMPT = (
    "You are a focused, concise RAG assistant.\n"
    "- Answer ONLY using the provided context.\n"
    "- If the answer is not in the context, say you don't know.\n"
    "- Include brief inline citations like (Source: filename.pdf).\n"
    "- Be factual and concise."
)

def get_system_prompt() -> str:
    """Resolve system prompt from file path or env/secret."""
    path = get_secret("SYSTEM_PROMPT_PATH")
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    return txt
        except Exception:
            pass
    env_val = get_secret("SYSTEM_PROMPT")
    return (env_val.strip() if env_val else DEFAULT_SYSTEM_PROMPT)

# -----------------------
# Utils
# -----------------------
def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def load_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text", sort=True))
    return "\n".join(parts)

def load_pdf_text_from_bytes(b: bytes) -> str:
    doc = fitz.open(stream=b, filetype="pdf")
    parts = []
    for page in doc:
        parts.append(page.get_text("text", sort=True))
    return "\n".join(parts)

def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

# -----------------------
# Vector store
# -----------------------
class VectorStore:
    def __init__(self, index_path: str | None, meta_path: str | None, model_name=EMBED_MODEL_NAME):
        self.index_path = index_path
        self.meta_path = meta_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta = None

    # ---------- Persistent (disk) ----------
    def load(self) -> bool:
        if not self.index_path or not self.meta_path:
            return False
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            self.meta = load_jsonl(self.meta_path)
            return True
        return False

    def build_from_docs_dir(self, docs_dir=DOCS_DIR):
        pdfs = sorted(glob.glob(os.path.join(docs_dir, "*.pdf")))
        if not pdfs:
            raise RuntimeError("No PDFs found in ./docs")
        corpus, meta = [], []
        for p in pdfs:
            try:
                text = load_pdf_text(p)
            except Exception as e:
                print(f"Failed to read {p}: {e}")
                continue
            chunks = chunk_text(text)
            base = os.path.basename(p)
            for i, ch in enumerate(chunks):
                corpus.append(ch)
                meta.append({"source": base, "chunk_id": i})
        self._fit_and_optionally_persist(corpus, meta)

    # ---------- Ephemeral (in-memory) ----------
    def build_from_inmemory_files(self, files: list[tuple[str, bytes]]):
        """
        files: list of (filename, bytes)
        """
        corpus, meta = [], []
        for fname, data in files:
            try:
                text = load_pdf_text_from_bytes(data)
            except Exception as e:
                print(f"Failed to read {fname}: {e}")
                continue
            chunks = chunk_text(text)
            for i, ch in enumerate(chunks):
                corpus.append(ch)
                meta.append({"source": fname, "chunk_id": i})
        if not corpus:
            raise RuntimeError("No text extracted from uploaded PDFs.")
        # Do NOT persist in ephemeral mode
        self._fit_and_optionally_persist(corpus, meta, persist=False)

    # ---------- Common fit ----------
    def _fit_and_optionally_persist(self, corpus, meta, persist=True):
        embs = self.model.encode(
            corpus, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        )
        embs = np.array(embs, dtype="float32")
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)  # cosine via normalized
        index.add(embs)
        self.index = index
        self.meta = [{"text": t, **m} for t, m in zip(corpus, meta)]
        if persist and self.index_path and self.meta_path:
            ensure_dir(os.path.dirname(self.index_path))
            faiss.write_index(self.index, self.index_path)
            save_jsonl(self.meta_path, self.meta)

    def search(self, query: str, top_k: int):
        if self.index is None:
            raise RuntimeError("Index not loaded.")
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.array(q, dtype="float32")
        scores, idxs = self.index.search(q, top_k)
        hits = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            m = self.meta[i]
            hits.append(
                {"score": float(s), "text": m["text"], "source": m["source"], "chunk_id": m["chunk_id"]}
            )
        return hits

# -----------------------
# OpenRouter
# -----------------------
def call_openrouter(messages, model=OPENROUTER_MODEL, temperature=0.1, max_tokens=700) -> str:
    api_key = get_secret("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    ref = get_secret("HTTP_REFERER")
    ttl = get_secret("X_TITLE")
    if ref:
        headers["HTTP-Referer"] = ref
    if ttl:
        headers["X-Title"] = ttl
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

# -----------------------
# Prompt
# -----------------------
def build_prompt(context_chunks, user_query: str):
    context_blob = "\n\n".join(
        f"[{i+1}] {c['text']}\n(Source: {c['source']} | Chunk {c['chunk_id']})"
        for i, c in enumerate(context_chunks)
    )
    system = get_system_prompt()
    user = (
        f"Context:\n{context_blob}\n\n"
        f"User question: {user_query}\n\n"
        "Follow the system instructions strictly."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.set_page_config(page_title="Simple RAG Chat", page_icon="🧠", layout="wide")
    st.title("🧠 Simple RAG Chat (FAISS + OpenRouter Mistral)")
    st.caption("Upload PDFs ephemerally or index files from ./docs. System prompt is backend-controlled.")

    ensure_dir(DOCS_DIR)
    ensure_dir(INDEX_DIR)

    index_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path  = os.path.join(INDEX_DIR, "meta.jsonl")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top-K context chunks", 2, 12, TOP_K_DEFAULT, 1)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        model = st.text_input("OpenRouter Model", OPENROUTER_MODEL)

        st.markdown("---")
        ephemeral = st.toggle("Ephemeral uploads (do NOT save files)", value=True,
                              help="If enabled, uploaded PDFs are processed in-memory and never written to disk.")
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        reindex = st.button("Build / Rebuild Index")

        st.markdown("---")
        if not ephemeral:
            st.write("📄 PDFs in ./docs:")
            for p in sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf"))):
                st.write("•", os.path.basename(p))
        with st.expander("ℹ️ Active System Prompt (read-only)"):
            st.code(get_system_prompt())

    # Init / Load store
    if "store" not in st.session_state:
        st.session_state.store = VectorStore(index_path=index_path, meta_path=meta_path)
    store: VectorStore = st.session_state.store

    # Build / Rebuild logic
    def build_index():
        if uploads and ephemeral:
            files = [(u.name, u.getvalue()) for u in uploads]
            store.build_from_inmemory_files(files)
            st.success(f"Indexed {len(store.meta)} chunks from {len(files)} uploaded file(s). (Ephemeral)")
        elif uploads and not ephemeral:
            # Persist files then build from docs dir
            for u in uploads:
                with open(os.path.join(DOCS_DIR, u.name), "wb") as f:
                    f.write(u.getbuffer())
            store.build_from_docs_dir(DOCS_DIR)
            st.success(f"Saved uploads and indexed {len(store.meta)} chunks from ./docs.")
        else:
            # No uploads: use existing persistent docs
            store.build_from_docs_dir(DOCS_DIR)
            st.success(f"Indexed {len(store.meta)} chunks from ./docs.")

    # Load existing persistent index if available and not ephemeral/no uploads
    loaded = False
    if not ephemeral and not uploads:
        loaded = store.load()
        if loaded:
            st.info("Loaded existing index from disk.")

    if reindex or (not loaded and not uploads and not ephemeral and not store.load()):
        with st.spinner("Building/Updating index…"):
            try:
                build_index()
            except Exception as e:
                st.error(str(e))
                return
    elif uploads and ephemeral and store.index is None:
        # If user uploaded files in ephemeral mode, build immediately for first run
        with st.spinner("Indexing uploaded files (ephemeral)…"):
            try:
                build_index()
            except Exception as e:
                st.error(str(e))
                return
    elif store.index is None:
        # As a last resort (fresh app, no uploads, ephemeral ON): warn user
        st.warning("No index loaded. Upload PDFs (ephemeral) and/or turn OFF 'Ephemeral uploads' to use ./docs, then click 'Build / Rebuild Index'.")
    
    # Chat
    st.markdown("## Ask a question")
    query = st.text_input("Your question", placeholder="e.g., What is Dosha?")
    ask = st.button("Ask")

    if ask and query.strip():
        if store.index is None:
            st.error("No index is available. Please build the index first.")
            return

        with st.spinner("Retrieving context…"):
            hits = store.search(query, top_k=top_k)

        with st.expander("🔍 Retrieved context"):
            for h in hits:
                st.markdown(f"- **{h['source']}** (chunk {h['chunk_id']}), score={h['score']:.3f}")
            st.code("\n\n".join(h["text"] for h in hits[:3]))

        messages = build_prompt(hits, query)
        try:
            with st.spinner("Calling OpenRouter…"):
                answer = call_openrouter(messages, model=model, temperature=temperature)
        except Exception as e:
            st.error(str(e))
            return

        st.markdown("## Answer")
        st.write(answer)

        st.markdown("#### Sources")
        by_src = {}
        for h in hits:
            by_src.setdefault(h["source"], set()).add(h["chunk_id"])
        for s, ids in by_src.items():
            st.write(f"- {s} (chunks: {sorted(list(ids))[:10]})")

if __name__ == "__main__":
    main()
