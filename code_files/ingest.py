import os
import re
from typing import List, Tuple

from pypdf import PdfReader

from vector_db import SimpleVectorStore

import openai


def _get_api_key_from_testpy(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        m = re.search(r"api_key\s*=\s*[\'\"]([A-Za-z0-9\-\_]+)[\'\"]", txt)
        if m:
            return m.group(1)
    except Exception:
        return None


def get_openai_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    # try reading test.py in same folder
    candidate = os.path.join(os.path.dirname(__file__), "test.py")
    if os.path.exists(candidate):
        k = _get_api_key_from_testpy(candidate)
        if k:
            return k
    # fallback: try parent folder
    candidate = os.path.join(os.path.dirname(__file__), "..", "test.py")
    candidate = os.path.abspath(candidate)
    if os.path.exists(candidate):
        k = _get_api_key_from_testpy(candidate)
        if k:
            return k
    return None


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    # naive chunking that preserves sentences where possible
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 <= chunk_size:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)

    # if overlap required, create overlapping chunks
    if overlap > 0 and len(chunks) > 1:
        new_chunks = []
        for i, c in enumerate(chunks):
            if i == 0:
                new_chunks.append(c)
            else:
                # overlap tokens approximate by characters
                prev = new_chunks[-1]
                ov = prev[-overlap:]
                new_chunks.append((ov + " " + c).strip())
        chunks = new_chunks

    return chunks


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    key = get_openai_key()
    if not key:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY or put key in test.py")
    openai.api_key = key
    # call embeddings in batch
    res = openai.Embedding.create(input=texts, model=model)
    return [r["embedding"] for r in res["data"]]


def ingest_pdf(file_path: str, store: SimpleVectorStore, doc_id_prefix: str = "doc", chunk_size: int = 1000, overlap: int = 200):
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    ids = [f"{doc_id_prefix}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(file_path), "chunk_index": i, "text": chunks[i][:1000]} for i in range(len(chunks))]
    store.upsert(ids, embeddings, metadatas)
    return len(chunks)
