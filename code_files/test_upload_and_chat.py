# import requests
# import sys
# import os
# import time

# # Path to the text file to upload as a PDF (simulate PDF upload)
# pdf_path = r"C:\Users\abdul_ahad\Downloads\rag_strict_source_test.pdf"

# # Upload endpoint
# upload_url = "http://localhost:8000/upload"
# chat_url = "http://localhost:8000/chat"

# if not os.path.exists(pdf_path):
#     print(f"File not found: {pdf_path}")
#     sys.exit(1)

# print(f"Uploading {pdf_path} to {upload_url} ...")
# try:
#     with open(pdf_path, "rb") as f:
#         files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
#         resp = requests.post(upload_url, files=files, timeout=120)
#     print(f"Upload Status: {resp.status_code}")
#     print(f"Upload Response: {resp.json()}")
#     if resp.status_code != 200:
#         print("Upload failed. Exiting.")
#         sys.exit(1)
# except Exception as e:
#     print(f"Upload error: {e}")
#     sys.exit(1)

# # Wait a moment to ensure vector store is updated
# print("Waiting 2 seconds for vector store update...")
# time.sleep(2)

# # Now test the /chat endpoint with a relevant question
# question = "What is the project code of Project Orion?"
# payload = {"question": question}
# print(f"\nTesting /chat endpoint with question: {question}")
# try:
#     resp = requests.post(chat_url, json=payload, timeout=30)
#     print(f"Chat Status: {resp.status_code}")
#     try:
#         print(f"Chat Response: {resp.json()}")
#     except Exception:
#         print(f"Raw chat response: {resp.text}")
# except Exception as e:
#     print(f"Error calling /chat endpoint: {e}")
#     sys.exit(1)
