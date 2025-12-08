import streamlit as st
import requests
import tempfile
import os
from typing import Optional

st.set_page_config(page_title="RAG Chatbot UI", layout="wide")

st.title("Simple RAG Chatbot — UI")

# Backend base URL (adjust if needed)
BASE_URL = st.sidebar.text_input("Backend URL", value="http://localhost:8000")

st.sidebar.markdown("---")
st.sidebar.markdown("Upload a PDF to ingest into the vector store:")

uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Uploading and ingesting..."):
        # write to a temp file then upload via requests
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            files = {"file": (os.path.basename(tmp_path), open(tmp_path, "rb"), "application/pdf")}
            resp = requests.post(f"{BASE_URL}/upload", files=files, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                st.sidebar.success(f"Indexed {data.get('chunks_indexed',0)} chunks from {data.get('file')}")
            else:
                st.sidebar.error(f"Upload failed: {resp.status_code} {resp.text}")
        except Exception as e:
            st.sidebar.error(f"Upload error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

st.sidebar.markdown("---")
st.sidebar.markdown("Conversation controls")
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

conv_col1, conv_col2 = st.sidebar.columns([2, 1])
with conv_col1:
    if st.button("New Conversation"):
        st.session_state.conversation_id = None
with conv_col2:
    if st.button("List Conversations"):
        try:
            r = requests.get(f"{BASE_URL}/conversations", timeout=10)
            if r.status_code == 200:
                convs = r.json().get("conversations", [])
                st.sidebar.write(convs)
            else:
                st.sidebar.error(f"Error: {r.status_code}")
        except Exception as e:
            st.sidebar.error(str(e))

st.header("Chat")

placeholder = st.empty()

with placeholder.container():
    question = st.text_input("Your question:")
    top_k = st.slider("Retrieval top_k", min_value=1, max_value=10, value=5)
    submit = st.button("Ask")

    if submit and question:
        with st.spinner("Querying the agent..."):
            payload = {"question": question, "top_k": top_k}
            if st.session_state.conversation_id:
                payload["conversation_id"] = st.session_state.conversation_id
            try:
                r = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60)
            except Exception as e:
                st.error(f"Request failed: {e}")
                r = None
            if r is not None:
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.conversation_id = data.get("conversation_id")
                    st.subheader("Answer")
                    st.write(data.get("answer"))
                    st.subheader("Retrieved sources")
                    for item in data.get("retrieved", []):
                        md = item.get("metadata", {})
                        st.markdown(f"**{md.get('source','unknown')}** — chunk {md.get('chunk_index')}, score {item.get('score'):.4f}")
                        text = md.get("text","")
                        st.write(text)
                else:
                    st.error(f"Server error: {r.status_code} {r.text}")

    # show conversation messages if available
    if st.session_state.conversation_id:
        try:
            r = requests.get(f"{BASE_URL}/conversation/{st.session_state.conversation_id}")
            if r.status_code == 200:
                msgs = r.json().get("messages", [])
                st.subheader("Conversation History")
                for m in msgs:
                    role = m.get("role")
                    content = m.get("content")
                    if role == "user":
                        st.markdown(f"**User:** {content}")
                    else:
                        st.markdown(f"**Assistant:** {content}")
        except Exception:
            pass

st.sidebar.markdown("---")
st.sidebar.markdown("Server: " + BASE_URL)
