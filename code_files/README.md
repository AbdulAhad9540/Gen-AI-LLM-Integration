# Simple RAG Chatbot Backend

This is a minimal Retrieval-Augmented Generation (RAG) backend that:

- Accepts PDF uploads and ingests them by chunking and embedding text.
- Stores vectors and metadata in a simple on-disk numpy/pickle store.
- Provides a `/chat` endpoint that retrieves relevant chunks, grounds the LLM answer to them, and returns a grounded answer.

Quick start

1. Install dependencies (recommended inside a virtualenv):

```powershell
python -m pip install -r code_files/requirements.txt
```

2. Set your OpenAI API key (or put it in `test.py` as `api_key = "..."` — not recommended for production):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

3. Run the app:

```powershell
python code_files/app.py
```

4. Upload a PDF:

POST `http://localhost:8000/upload` multipart with `file` field.

5. Ask a question (JSON POST):

POST `http://localhost:8000/chat` with body:

```json
{ "question": "What does the document say about X?" }
```

Notes and features

- The agent is conservative: it is instructed to only answer from retrieved sources and to say "I don't know" when the sources don't contain the answer.
- Conversation history is preserved in memory and returned by `/conversation/{id}`.
- This demo uses the OpenAI API embeddings + chat completions. Ensure your key has access to the models.

Files added

- `app.py` — FastAPI app and endpoints
- `ingest.py` — PDF parsing, chunking, embedding
- `vector_db.py` — simple persistent vector store
- `agent.py` — RAG agent and conversation management
- `requirements.txt`, `README.md`
