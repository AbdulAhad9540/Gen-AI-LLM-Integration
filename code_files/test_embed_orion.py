# import os
# from dial_api import dial_embedding

# # Text to embed (from your Project Orion document)
# doc_text = '''
# Project Orion – Internal Technical Brief
# This document describes Project Orion, an internal experimental system created solely for testing
# Retrieval-Augmented Generation (RAG) pipelines. The information in this document does not exist
# outside this file.
# Project Identification
# Project Name: Orion
# Project Code: ORN-4729
# Launch Date: 14 March 2024
# Primary Owner: Data Systems Group
# System Architecture
# Project Orion uses a three-layer architecture:
# 1. Ingestion Layer – responsible for document loading and chunking.
# 2. Retrieval Layer – uses vector embeddings stored in an internal database named VegaDB.
# 3. Generation Layer – powered by a lightweight transformer model named Atlas-7B.
# Operational Constraints
# • Maximum supported document size: 18 MB
# • Chunk size used during ingestion: 512 tokens
# • Overlap between chunks: 64 tokens
# Known Limitations
# Project Orion does not support image-based PDFs or scanned documents. If a document exceeds
# the size limit, ingestion will fail silently
# '''

# # Split into chunks (simulate chunking)
# chunks = [doc_text[i:i+1000] for i in range(0, len(doc_text), 1000)]

# print(f"Embedding {len(chunks)} chunk(s)...")
# try:
#     embeddings = dial_embedding(chunks, model="text-embedding-3-small-1")
#     print(f"Successfully embedded {len(embeddings)} chunk(s). Example embedding (first 5 dims):\n{embeddings[0][:5]}")

#     # Now test the /chat endpoint with a relevant question
#     import requests
#     chat_url = "http://localhost:8000/chat"
#     question = "What is the project code of Project Orion?"
#     payload = {"question": question}
#     print(f"\nTesting /chat endpoint with question: {question}")
#     try:
#         resp = requests.post(chat_url, json=payload, timeout=30)
#         print(f"Status: {resp.status_code}")
#         try:
#             print(f"Response: {resp.json()}")
#         except Exception:
#             print(f"Raw response: {resp.text}")
#     except Exception as e:
#         print(f"Error calling /chat endpoint: {e}")
# except Exception as e:
#     print(f"Embedding failed: {e}")
