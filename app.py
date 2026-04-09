import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import re

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chat — RAG Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks
TOP_K = 4              # number of chunks to retrieve
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def extract_text(pdf_file) -> str:
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]


def build_index(chunks: list[str], embedder: SentenceTransformer):
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(query: str, index, chunks: list[str], embedder: SentenceTransformer) -> list[str]:
    q_emb = embedder.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    _, indices = index.search(q_emb, TOP_K)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def answer(question: str, context_chunks: list[str], llm: ChatGroq) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    messages = [
        SystemMessage(content=(
            "You are a precise document assistant. Answer the user's question using ONLY "
            "the context provided below. If the answer is not in the context, say "
            "'I couldn\\'t find that in the document.' Be concise and cite relevant details.\n\n"
            f"CONTEXT:\n{context}"
        )),
        HumanMessage(content=question),
    ]
    return llm.invoke(messages).content


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("📄 PDF Chat")
st.caption("Upload a PDF, then ask anything about it. Powered by RAG + Groq LLaMA 3.")

# Sidebar — API key + upload
with st.sidebar:
    st.header("Setup")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Free at console.groq.com",
    )

    st.divider()
    st.header("Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded:
        st.success(f"✅ {uploaded.name}")

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. PDF is split into overlapping chunks\n"
        "2. Chunks are embedded with `all-MiniLM-L6-v2`\n"
        "3. Your question retrieves the top-4 relevant chunks via FAISS\n"
        "4. LLaMA 3 answers using only those chunks"
    )
    st.markdown("[Get free Groq API key →](https://console.groq.com)")

# Main area
if not api_key:
    st.info("Enter your Groq API key in the sidebar to get started.")
    st.stop()

if not uploaded:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

# Process PDF (cached per file)
@st.cache_data(show_spinner="Parsing and indexing PDF...")
def process_pdf(file_bytes: bytes, filename: str):
    import io
    embedder = load_embedder()
    text = extract_text(io.BytesIO(file_bytes))
    chunks = chunk_text(text)
    index, _ = build_index(chunks, embedder)
    return chunks, index, len(text), len(chunks)

file_bytes = uploaded.read()
chunks, index, char_count, chunk_count = process_pdf(file_bytes, uploaded.name)

with st.sidebar:
    st.divider()
    st.caption(f"📊 {char_count:,} characters · {chunk_count} chunks indexed")

# Chat interface
llm = ChatGroq(api_key=api_key, model=LLM_MODEL, temperature=0.1)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# Reset chat when PDF changes
if st.session_state.current_pdf != uploaded.name:
    st.session_state.messages = []
    st.session_state.current_pdf = uploaded.name

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if question := st.chat_input(f"Ask anything about {uploaded.name}..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            embedder = load_embedder()
            relevant = retrieve(question, index, chunks, embedder)
            reply = answer(question, relevant, llm)

        st.markdown(reply)

        with st.expander("📎 Source chunks used"):
            for i, chunk in enumerate(relevant, 1):
                st.markdown(f"**Chunk {i}:**")
                st.caption(chunk)

    st.session_state.messages.append({"role": "assistant", "content": reply})
