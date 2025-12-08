import os
import time
from typing import List, Dict, Any, Optional

import openai

from vector_db import SimpleVectorStore
from ingest import get_openai_key


class RAGAgent:
    def __init__(self, store: SimpleVectorStore, chat_model: str = None, embed_model: str = "text-embedding-3-small"):
        self.store = store
        self.embed_model = embed_model
        self.chat_model = chat_model or os.getenv("OPENAI_CHAT_MODEL") or "gpt-3.5-turbo"
        key = get_openai_key()
        if not key:
            raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY or put key in test.py")
        openai.api_key = key
        # in-memory conversations
        self.conversations: Dict[str, List[Dict[str, str]]] = {}

    def _embed_query(self, query: str):
        res = openai.Embedding.create(input=[query], model=self.embed_model)
        return res["data"][0]["embedding"]

    def _call_chat(self, messages: List[Dict[str, str]], max_tokens: int = 512):
        resp = openai.ChatCompletion.create(model=self.chat_model, messages=messages, max_tokens=max_tokens, temperature=0.0)
        return resp["choices"][0]["message"]["content"]

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

        # Build a grounded system prompt that prevents hallucination
        context = self._build_context_from_retrievals(retrievals)

        system = (
            "You are an assistant that answers questions using ONLY the provided sources. "
            "If the answer is not contained in the sources, say you don't know. "
            "Cite sources inline using [source:chunk_index] and include a short excerpt if useful. "
            "Be concise and avoid inventing facts."
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
