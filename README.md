# PDF Chat — RAG Q&A

Ask questions about any PDF using a local RAG pipeline. Upload a document, get instant answers grounded in the text — no hallucinations, no guessing.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43-red)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-orange)

## Demo

**[Live demo →](https://rag-pdf-chat.streamlit.app)**

## How it works

```
PDF → text extraction → chunking (500 chars, 100 overlap)
    → sentence-transformers embeddings (all-MiniLM-L6-v2)
    → FAISS index (cosine similarity)

Question → embed → top-4 chunk retrieval
         → LLaMA 3.3 70B (Groq) answers using only retrieved context
```

1. **Parse** — `pdfplumber` extracts raw text from any PDF
2. **Chunk** — text split into 500-char overlapping chunks
3. **Embed** — `sentence-transformers` converts each chunk to a 384-dim vector
4. **Index** — `FAISS` stores vectors for fast cosine similarity search
5. **Retrieve** — at query time, top-4 most relevant chunks are fetched
6. **Generate** — `LLaMA 3.3 70B` via Groq answers using only those chunks

## Setup

**1. Get a free Groq API key**

Sign up at [console.groq.com](https://console.groq.com) — free tier is generous.

**2. Clone and install**

```bash
git clone https://github.com/sa1701/rag-pdf-chat
cd rag-pdf-chat
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Set your API key**

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

**4. Run**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

## Tech stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| PDF parsing | pdfplumber |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (CPU) |
| LLM | LLaMA 3.3 70B via Groq API |
| Orchestration | LangChain |

## Deploy to Streamlit Cloud (free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, set `app.py` as entrypoint
4. Add `GROQ_API_KEY` in Secrets settings
