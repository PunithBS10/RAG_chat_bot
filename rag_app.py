# rag_app.py
import os
import glob
import json
import hashlib
import numpy as np
import requests
import streamlit as st
import faiss
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# -----------------------
# Load env (for OpenRouter + prompt config)
# -----------------------
load_dotenv()

# -----------------------
# Config
# -----------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # fast & free
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"
DOCS_DIR = "docs"               # put PDFs here
INDEX_DIR = "vector_store"      # FAISS + metadata stored here
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K_DEFAULT = 5

# -----------------------
# System prompt (backend-only)
# -----------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a focused, concise RAG assistant.\n"
    "- Answer ONLY using the provided context.\n"
    "- If the answer is not in the context, say you don't know.\n"
    "- Include brief inline citations like (Source: filename.pdf).\n"
    "- Be factual and concise."
)

def get_system_prompt() -> str:
    """
    Resolve system prompt in this priority:
    1) SYSTEM_PROMPT_PATH file (e.g., system_prompt.txt)
    2) SYSTEM_PROMPT env var
    3) DEFAULT_SYSTEM_PROMPT
    """
    path = os.getenv("SYSTEM_PROMPT_PATH")
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                if txt:
                    return txt
        except Exception:
            pass
    env_val = os.getenv("SYSTEM_PROMPT")
    return (env_val.strip() if env_val else DEFAULT_SYSTEM_PROMPT)

# -----------------------
# Utilities
# -----------------------
def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks = []
    start = 0
    n = len(text)
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
# Vector Store
# -----------------------
class VectorStore:
    def __init__(self, index_path: str, meta_path: str, model_name=EMBED_MODEL_NAME):
        self.index_path = index_path
        self.meta_path = meta_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta = None

    def load(self) -> bool:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            self.meta = load_jsonl(self.meta_path)
            return True
        return False

    def build(self, docs_dir=DOCS_DIR):
        # Gather PDFs
        pdfs = sorted(glob.glob(os.path.join(docs_dir, "*.pdf")))
        if not pdfs:
            raise RuntimeError("No PDFs found in ./docs")

        corpus = []
        meta = []
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

        if not corpus:
            raise RuntimeError("No text extracted from PDFs.")

        # Embed
        embs = self.model.encode(
            corpus, batch_size=64, normalize_embeddings=True, show_progress_bar=True
        )
        embs = np.array(embs, dtype="float32")
        d = embs.shape[1]

        # Cosine similarity via inner product on normalized vectors
        index = faiss.IndexFlatIP(d)
        index.add(embs)
        self.index = index
        self.meta = [{"text": t, **m} for t, m in zip(corpus, meta)]

        # Persist
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
# OpenRouter (chat completions)
# -----------------------
def call_openrouter(messages, model=OPENROUTER_MODEL, temperature=0.1, max_tokens=700) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Optional attribution (helps OpenRouter)
    ref = os.getenv("HTTP_REFERER")
    ttl = os.getenv("X_TITLE")
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
# Prompt builder
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
    st.caption("Put PDFs in the `docs/` folder. Index builds automatically. System prompt is loaded from backend.")

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
        reindex = st.button("Rebuild Index")

        st.markdown("---")
        st.write("📄 PDFs found:")
        for p in sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf"))):
            st.write("•", os.path.basename(p))

        st.markdown("---")
        with st.expander("ℹ️ Active System Prompt (read-only)"):
            st.code(get_system_prompt())

    # Init / Load store (cache model in session)
    if "store" not in st.session_state:
        st.session_state.store = VectorStore(index_path=index_path, meta_path=meta_path)

    store: VectorStore = st.session_state.store
    need_build = reindex or (not store.load())
    if need_build:
        with st.spinner("Building/Updating index…"):
            try:
                store.build(DOCS_DIR)
                st.success(f"Indexed {len(store.meta)} chunks.")
            except Exception as e:
                st.error(str(e))
                return

    # Chat input
    st.markdown("## Ask a question")
    query = st.text_input("Your question", placeholder="e.g., What is Dosha?")
    ask = st.button("Ask")

    if ask and query.strip():
        # Retrieve
        with st.spinner("Retrieving context…"):
            hits = store.search(query, top_k=top_k)

        with st.expander("🔍 Retrieved context"):
            for h in hits:
                st.markdown(f"- **{h['source']}** (chunk {h['chunk_id']}), score={h['score']:.3f}")
            st.code("\n\n".join(h["text"] for h in hits[:3]))

        # Generate
        messages = build_prompt(hits, query)
        try:
            with st.spinner("Calling OpenRouter…"):
                answer = call_openrouter(messages, model=model, temperature=temperature)
        except Exception as e:
            st.error(str(e))
            return

        st.markdown("## Answer")
        st.write(answer)

        # Sources list
        st.markdown("#### Sources")
        by_src = {}
        for h in hits:
            by_src.setdefault(h["source"], set()).add(h["chunk_id"])
        for s, ids in by_src.items():
            st.write(f"- {s} (chunks: {sorted(list(ids))[:10]})")

if __name__ == "__main__":
    main()
