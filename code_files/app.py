from fastapi.responses import JSONResponse
from nl2sql import get_db_connection, introspect_schema, generate_sql, execute_sql, result_to_nl

# Example: set DB path here (or use env var)
DB_PATH = os.getenv("NL2SQL_DB_PATH", os.path.join(DATA_DIR, "demo.db"))

@app.post("/nl2sql")
async def nl2sql_query(payload: dict):
    """
    Receives: {"question": "..."}
    Returns: {"sql": ..., "result": ..., "table": ...}
    """
    question = payload.get("question")
    if not question:
        return JSONResponse(status_code=400, content={"error": "Missing question"})
    try:
        conn = get_db_connection(DB_PATH)
        schema = introspect_schema(conn)
        sql = generate_sql(question, schema)
        rows = execute_sql(conn, sql)
        nl = result_to_nl(rows)
        if nl is not None:
            return {"sql": sql, "result": nl, "table": None}
        else:
            # Table view: return as list of dicts
            return {"sql": sql, "result": None, "table": rows}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from vector_db import SimpleVectorStore
from ingest import ingest_pdf
from agent import RAGAgent


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Simple RAG Chatbot Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# singletons
STORE_PATH = os.path.join(DATA_DIR, "vector_store.pkl")
store = SimpleVectorStore(path=STORE_PATH)
agent = RAGAgent(store=store)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    try:
        n = ingest_pdf(dest, store, doc_id_prefix=os.path.splitext(file.filename)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "file": file.filename, "chunks_indexed": n}


@app.post("/chat")
async def chat(question: dict):
    # Expect JSON with keys: question, conversation_id (optional), top_k (optional)
    q = question.get("question")
    if not q:
        raise HTTPException(status_code=400, detail="question field is required")
    conv = question.get("conversation_id")
    top_k = int(question.get("top_k", 5))
    try:
        resp = agent.ask(q, conversation_id=conv, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return resp


@app.get("/conversations")
def list_conversations():
    return {"conversations": list(agent.conversations.keys())}


@app.get("/conversation/{conv_id}")
def get_conversation(conv_id: str):
    conv = agent.conversations.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"conversation_id": conv_id, "messages": conv}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
