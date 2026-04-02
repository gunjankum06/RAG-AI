"""Streamlit chat UI for RAG AI."""

import streamlit as st
import httpx

st.set_page_config(page_title="RAG AI Chat", page_icon="🔍", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.title("RAG AI")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    api_key = st.text_input("API Key", type="password")
    collection = st.text_input("Collection", value="default")
    top_k = st.slider("Top K results", 1, 20, 5)
    use_rerank = st.checkbox("Enable reranking", value=True)

    st.divider()

    # File upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "docx"],
    )
    if st.button("Ingest", disabled=not uploaded_files):
        if not api_key:
            st.error("Enter an API key first")
        else:
            with st.spinner("Ingesting documents..."):
                files = [
                    ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                    for f in uploaded_files
                ]
                try:
                    resp = httpx.post(
                        f"{api_url.rstrip('/')}/api/v1/ingest",
                        headers={"X-API-Key": api_key},
                        files=files,
                        data={"collection": collection},
                        timeout=300.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"Ingested {data.get('chunks_stored', 0)} chunks "
                            f"from {data.get('documents_loaded', 0)} document(s)"
                        )
                    else:
                        st.error(f"Ingestion failed: {resp.text}")
                except httpx.HTTPError as exc:
                    st.error(f"Connection error: {exc}")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Chat UI ───────────────────────────────────────────────────────────

st.title("Chat with your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src.get('filename', 'unknown')}** (score: {src.get('score', 0):.3f})")
                    st.text(src.get("content", "")[:300])
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not api_key:
        st.error("Please enter your API key in the sidebar")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build chat history for API
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # Exclude current question
    ]

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = httpx.post(
                    f"{api_url.rstrip('/')}/api/v1/query",
                    headers={"X-API-Key": api_key, "Content-Type": "application/json"},
                    json={
                        "question": prompt,
                        "collection": collection,
                        "top_k": top_k,
                        "rerank": use_rerank,
                        "stream": False,
                        "chat_history": chat_history,
                    },
                    timeout=300.0,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer received")
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.markdown(
                                    f"**{src.get('filename', 'unknown')}** "
                                    f"(score: {src.get('score', 0):.3f})"
                                )
                                st.text(src.get("content", "")[:300])
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    error_msg = f"API error ({resp.status_code}): {resp.text}"
                    st.error(error_msg)

            except httpx.HTTPError as exc:
                st.error(f"Connection error: {exc}")
