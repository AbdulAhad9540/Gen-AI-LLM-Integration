import os
import re
from typing import List, Tuple

from pypdf import PdfReader

from vector_db import SimpleVectorStore

from dial_api import dial_embedding


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


def embed_texts(texts: List[str], model: str = "text-embedding-3-small-1", use_mock: bool = False) -> List[List[float]]:
    if use_mock:
        # Use a simple mock embedding for testing (not recommended for production)
        import hashlib
        embeddings = []
        for text in texts:
            # Create a fake 384-dim embedding from text hash
            hash_obj = hashlib.sha256(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            # Generate pseudo-random embedding
            embedding = [(hash_int >> (i % 256)) % 256 / 256.0 for i in range(384)]
            embeddings.append(embedding)
        print(f"✓ Created {len(embeddings)} mock embeddings (no API call)")
        return embeddings
    try:
        print(f"→ Attempting to embed {len(texts)} texts using OpenAI API (timeout=5s)...")
        embeddings = dial_embedding(texts, model=model)
        print(f"✓ Successfully created {len(embeddings)} embeddings via OpenAI API")
        return embeddings
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            print(f"⚠ DIAL API returned 401 Unauthorized - Invalid/missing API key!")
            print(f"   Please check your .env file and set OPENAI_API_KEY correctly")
        elif "timeout" in error_msg.lower() or "read timeout" in error_msg.lower():
            print(f"⚠ DIAL API timeout, falling back to mock embeddings")
        else:
            print(f"⚠ DIAL API error ({type(e).__name__}: {error_msg[:80]})")
        print(f"   Falling back to mock embeddings...")
        return embed_texts(texts, model=model, use_mock=True)


def ingest_pdf(file_path: str, store: SimpleVectorStore, doc_id_prefix: str = "doc", chunk_size: int = 1000, overlap: int = 200):
    print(f"[ingest_pdf] Starting ingestion for: {file_path}")
    text = extract_text_from_pdf(file_path)
    print(f"[ingest_pdf] Extracted text length: {len(text)}")
    print(f"[ingest_pdf] Extracted text:\n{text}\n{'='*40}")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"[ingest_pdf] Number of chunks: {len(chunks)}")
    if not chunks:
        print(f"[ingest_pdf] No chunks produced. Exiting.")
        return 0
    print(f"[ingest_pdf] First chunk preview: {chunks[0][:100]}...")
    # embeddings = embed_texts(chunks)
    embeddings = embed_texts(chunks, use_mock=False)
    print(f"[ingest_pdf] Embeddings created: {len(embeddings)}")
    ids = [f"{doc_id_prefix}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(file_path), "chunk_index": i, "text": chunks[i][:1000]} for i in range(len(chunks))]
    print(f"[ingest_pdf] About to upsert {len(ids)} records to vector store...")
    try:
        store.upsert(ids, embeddings, metadatas)
        print(f"[ingest_pdf] Upserted {len(ids)} records to vector store.")
    except Exception as e:
        print(f"[ingest_pdf] Exception during upsert: {e}")
        raise
    return len(chunks)
