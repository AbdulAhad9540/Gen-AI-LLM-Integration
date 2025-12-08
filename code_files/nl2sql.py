import os
import sqlite3
from typing import List, Dict, Any, Optional
import openai

def get_openai_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    return None

def get_db_connection(db_path: str):
    # For demo: use SQLite. For prod, swap to psycopg2/MySQL connector.
    return sqlite3.connect(db_path)

def introspect_schema(conn) -> str:
    # Returns a string describing all tables and columns in the DB
    cursor = conn.cursor()
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    schema = []
    for (table_name,) in tables:
        cols = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
        col_str = ", ".join([f"{col[1]} {col[2]}" for col in cols])
        schema.append(f"Table {table_name}: {col_str}")
    return "\n".join(schema)

def generate_sql(question: str, schema: str, dialect: str = "sqlite") -> str:
    key = get_openai_key()
    if not key:
        raise RuntimeError("OpenAI API key not found.")
    openai.api_key = key
    system = (
        f"You are a helpful assistant that translates natural language to SQL for a {dialect} database. "
        "Given the schema below, write a single SQL query that answers the user's question. "
        "Only use tables and columns that exist. Do not hallucinate. Return only the SQL query, no explanation."
    )
    prompt = f"Schema:\n{schema}\n\nQuestion: {question}\nSQL:"
    resp = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo"),
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    sql = resp["choices"][0]["message"]["content"].strip()
    # Remove code block markers if present
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[-1].strip()
        if sql.endswith("```"):
            sql = sql[:-3].strip()
    return sql

def execute_sql(conn, sql: str) -> List[Dict[str, Any]]:
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        return [{"error": str(e)}]

def result_to_nl(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No results found."
    if "error" in rows[0]:
        return f"Error: {rows[0]['error']}"
    if len(rows) == 1:
        # Describe the row in NL
        items = [f"{k}: {v}" for k, v in rows[0].items()]
        return "Result: " + ", ".join(items)
    return None  # For >1 row, show table
