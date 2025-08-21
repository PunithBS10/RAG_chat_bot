# Simple RAG Chat (FAISS + OpenRouter Mistral)

A minimal Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **FAISS**, and **sentence-transformers**, using **OpenRouter** (`mistralai/mistral-7b-instruct`) for answers.
**Ephemeral uploads by default** — uploaded PDFs are processed in memory and not saved to disk.

---

## Features

* Local vector search with FAISS (`all-MiniLM-L6-v2` embeddings)
* OpenRouter generation (Mistral 7B Instruct)
* Backend-only system prompt (`system_prompt.txt` or secret)
* Ephemeral uploads (toggleable); optional persistent indexing from `./docs`
* Single-file app: `rag_app.py`

---

## Quick Start (Local)

1. **Install**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure**
   Create `.env` in the project root:

   ```env
   OPENROUTER_API_KEY=sk-or-v1-...
   SYSTEM_PROMPT_PATH=system_prompt.txt   # or use SYSTEM_PROMPT instead
   ```

3. **Run**

   ```bash
   python -m streamlit run rag_app.py
   ```

4. **Use**

   * In the sidebar, leave **Ephemeral uploads** ON.
   * Upload PDFs → click **Build / Rebuild Index** → ask questions.

---

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On [https://share.streamlit.io](https://share.streamlit.io) create a new app:

   * **Repo:** `<your-username>/RAG_chat_bot`
   * **Branch:** `main`
   * **Main file:** `rag_app.py`
3. **Secrets** (Settings → Secrets):

   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."
   SYSTEM_PROMPT_PATH = "system_prompt.txt"   # or set SYSTEM_PROMPT instead
   ```
4. Deploy, upload PDFs in the sidebar, then **Build / Rebuild Index**.

> Streamlit Cloud storage is temporary. With Ephemeral uploads ON, nothing is written to disk.

---

## Configuration

| Key                  | Where            | Notes                           |
| -------------------- | ---------------- | ------------------------------- |
| `OPENROUTER_API_KEY` | `.env` / Secrets | Required                        |
| `SYSTEM_PROMPT_PATH` | `.env` / Secrets | Path to `system_prompt.txt`     |
| `SYSTEM_PROMPT`      | `.env` / Secrets | Inline prompt (used if no file) |

---

## Project Layout

```
.
├─ rag_app.py
├─ requirements.txt
├─ system_prompt.txt
├─ docs/           # optional (persistent mode)
└─ vector_store/   # auto-generated (persistent mode)
```

---

## Troubleshooting

* **Auth error (401):** check `OPENROUTER_API_KEY` in `.env` or Secrets.
* **No index loaded:** upload PDFs and click **Build / Rebuild Index**.
* **Windows:** use `python -m streamlit run rag_app.py` to start the app.

---

**License:** MIT
