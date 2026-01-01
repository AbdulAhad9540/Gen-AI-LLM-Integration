import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vector_db import SimpleVectorStore
from ingest import ingest_pdf
from agent import RAGAgent
from nl2sql import get_db_connection, introspect_schema, generate_sql, execute_sql, result_to_nl

from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    top_k: int = 5


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Simple RAG Chatbot Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# singletons
STORE_PATH = os.path.join(DATA_DIR, "vector_store.pkl")
store = SimpleVectorStore(path=STORE_PATH)

# Initialize agent with error handling
try:
    agent = RAGAgent(store=store)
except Exception as e:
    print(f"Warning: RAGAgent initialization failed: {e}")
    agent = None

# Example: set DB path here (or use env var)
DB_PATH = os.getenv("NL2SQL_DB_PATH", os.path.join(DATA_DIR, "demo.db"))


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
        print(f"[upload_pdf] Ingestion complete for {file.filename}, chunks indexed: {n}")
    except Exception as e:
        print(f"[upload_pdf] Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    try:
        response = {"status": "ok", "file": file.filename, "chunks_indexed": n}
        print(f"[upload_pdf] Returning response: {response}")
        return response
    except Exception as e:
        print(f"[upload_pdf] Exception during response: {e}")
        raise HTTPException(status_code=500, detail=f"Response error: {e}")


@app.post("/chat")
async def chat(payload: ChatRequest):
    resp = agent.ask(
        payload.question,
        payload.conversation_id,
        payload.top_k
    )
    return resp



@app.get("/conversations")
def list_conversations():
    if agent is None:
        return {"conversations": []}
    return {"conversations": list(agent.conversations.keys())}


@app.get("/conversation/{conv_id}")
def get_conversation(conv_id: str):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    conv = agent.conversations.get(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"conversation_id": conv_id, "messages": conv}


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


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)
#What is the project code of Project Orion?
#uvicorn app:app --host 0.0.0.0 --port 8000 --reload 