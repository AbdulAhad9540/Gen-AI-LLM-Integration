import os
import time
from typing import List, Dict, Any, Optional

from dial_api import dial_embedding, dial_chat

from vector_db import SimpleVectorStore
from ingest import get_openai_key
import sqlite3
from typing import List, Dict, Any
from nl2sql import get_db_connection, introspect_schema, generate_sql

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "chinook.db")


class RAGAgent:
    def __init__(self, store: SimpleVectorStore, chat_model: str = None, embed_model: str = "text-embedding-3-small-1"):
        self.store = store
        self.embed_model = embed_model
        # Use the correct DIAL deployment name for the chat model
        self.chat_model = chat_model or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini-2024-07-18"
        key = get_openai_key()
        if not key:
            raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY or put key in test.py")
        # in-memory conversations
        self.conversations: Dict[str, List[Dict[str, str]]] = {}

    def _embed_query(self, query: str):
        return dial_embedding([query], model=self.embed_model)[0]

    def _call_chat(self, messages: List[Dict[str, str]], max_tokens: int = 512):
        return dial_chat(messages, model=self.chat_model, max_tokens=max_tokens, temperature=0.0)

    def _build_context_from_retrievals(self, retrievals: List[Dict[str, Any]]) -> str:
        # create a single context string with source citations
        parts = []
        for rid, score, meta in retrievals:
            txt = meta.get("text", "")
            src = meta.get("source", "unknown")
            idx = meta.get("chunk_index", None)
            parts.append(f"Source: {src} | chunk: {idx}\n{txt}\n---")
        return "\n".join(parts)

    def ask(self, question: str, conversation_id: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        # record question in conversation
        if conversation_id is None:
            conversation_id = f"conv-{int(time.time()*1000)}"
        self.conversations.setdefault(conversation_id, [])
        self.conversations[conversation_id].append({"role": "user", "content": question})

        q_vec = self._embed_query(question)
        retrievals = self.store.search(q_vec, top_k=top_k)
        print(f"[RAGAgent] Retrievals for question '{question}':")
        for rid, score, meta in retrievals:
            print(f"  id={rid}, score={score}, text={meta.get('text','')[:80]}")
        if not retrievals:
            print("[RAGAgent] No relevant chunks retrieved.")

        # Build a grounded system prompt that prevents hallucination
        context = self._build_context_from_retrievals(retrievals)

        system = (
                        "You are a retrieval-grounded assistant.\n"
                        "You MUST answer using ONLY the provided excerpts.\n"
                        "If the answer is not explicitly present, respond with:\n"
                        "'I don't know based on the provided document.'\n"
                        "Do NOT use prior knowledge.\n"
                        "Always cite sources like [filename:chunk_index]."
                    )


        # include recent conversation for context following
        messages = [{"role": "system", "content": system}]

        # include retrieved context as a system/context message
        if context.strip():
            messages.append({"role": "system", "content": f"Here are relevant excerpts:\n\n{context}"})

        # include last 6 turns of conversation (user+assistant)
        history = self.conversations.get(conversation_id, [])[-12:]
        messages.extend(history)

        # ask LLM to first summarize the retrieved context (short) then answer
        prompt = (
            "Given the excerpts above, first write a one-paragraph summary of the relevant information. "
            "Then answer the user's question below and cite sources in square brackets.\n\nQuestion: " + question
        )
        messages.append({"role": "user", "content": prompt})

        answer = self._call_chat(messages)

        # save assistant reply
        self.conversations[conversation_id].append({"role": "assistant", "content": answer})

        # return answer and the retrieval metadata used
        return {
            "conversation_id": conversation_id,
            "answer": answer,
            "retrieved": [
                {"id": r[0], "score": r[1], "metadata": r[2]} for r in retrievals
            ],
        }



def execute_sql(sql: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()

def generate_nl_answer(question: str, sql: str, rows):
    """
    Uses LLM to generate a natural language answer
    based on the user question and SQL result.
    """

    system_prompt = (
        "You are a helpful assistant that explains database query results "
        "to users in clear, natural language."
    )

    user_prompt = f"""
User question:
{question}

SQL query:
{sql}

Query result:
{rows}

Generate a concise and clear natural language answer.
"""

    answer = dial_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=150,
    )

    return answer.strip()


def format_response(question: str, sql: str, rows: List[Dict[str, Any]]):
    if not rows:
        nl_answer = generate_nl_answer(question, sql, rows)
        return {
            "sql": sql,
            "response_type": "text",
            "answer": nl_answer
        }

    if len(rows) == 1:
        nl_answer = generate_nl_answer(question, sql, rows)
        return {
            "sql": sql,
            "response_type": "text",
            "answer": nl_answer
        }

    # >1 row → table (NO LLM)
    return {
        "sql": sql,
        "response_type": "table",
        "columns": list(rows[0].keys()),
        "rows": [list(row.values()) for row in rows]
    }


def answer_question(question: str):
    conn = get_db_connection(DB_PATH)
    schema = introspect_schema(conn)
    sql = generate_sql(question, schema)
    conn.close()

    # Safety check
    if any(word in sql.lower() for word in ["insert", "update", "delete", "drop", "alter"]):
        raise ValueError("Only SELECT queries are allowed")

    rows = execute_sql(sql)
    return format_response(question, sql, rows)



