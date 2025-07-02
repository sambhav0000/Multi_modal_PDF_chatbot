import os
import datetime
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")


st.set_page_config(
    page_title="Multi-PDF Q&A",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* App background gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    /* Header styling */
    .header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        background-color: #4A90E2;
        padding: 10px 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #4A90E2 !important;
        color: white !important;
        font-weight: bold;
    }
    /* User bubble */
    .user-bubble {
        background-color: #9DE0AD;
        padding: 12px;
        border-radius: 15px 15px 0 15px;
        margin-bottom: 10px;
        text-align: right;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* Bot bubble */
    .bot-bubble {
        background-color: #FFFFFF;
        padding: 12px;
        border-radius: 15px 15px 15px 0;
        margin-bottom: 15px;
        border: 1px solid #BDE0FE;
        text-align: left;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    /* Citation styling */
    .citation {
        font-size: 0.75em;
        color: #555555;
        margin-top: 4px;
    }
    /* Timestamp styling */
    .timestamp {
        font-size: 0.75em;
        color: #888888;
        margin-top: 4px;
    }
    /* Button hover effect */
    button:hover {
        background-color: #4A90E2 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header section
st.markdown('<div class="header">📄 Multi-PDF Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown("_Select PDFs, sync with backend, then chat! Your chat history persists._")

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []  # entries: {question, answer, citations, images, timestamp}

# Upload section in expander
with st.expander("📁 Upload PDFs for Chat", expanded=True):
    uploaded = st.file_uploader(
        "Select one or more PDF files:",
        type=["pdf"],
        accept_multiple_files=True
    )
    if uploaded:
        if st.button("🔄 Sync & Index PDFs"):
            files_payload = [
                ("files", (f.name, f.read(), "application/pdf")) for f in uploaded
            ]
            with st.spinner("Syncing PDFs..."):
                res = requests.post(
                    f"{API_URL}/upload",
                    files=files_payload,
                    timeout=300
                )
            if res.status_code == 200:
                data = res.json()
                st.success(f"Indexed {data['chunks_indexed']} chunks.")
                if data.get('errors'):
                    st.warning("Errors:\n" + "\n".join(data['errors']))
                st.session_state.uploaded_files = [f.name for f in uploaded]
                st.session_state.chat_history = []
            else:
                st.error(f"Upload failed: {res.text}")

# Display active PDFs
if st.session_state.uploaded_files:
    st.markdown("**Active PDFs:** " + ", ".join(st.session_state.uploaded_files))

st.markdown("---")

# Chat history display
st.header("💬 Chat History")
for entry in st.session_state.chat_history:
    # User message
    cols_user = st.columns([3, 1])
    with cols_user[1]:
        st.markdown(
            f"""
            <div class='user-bubble'>
            <strong>You:</strong> {entry['question']}<br>
            <div class='timestamp'>{entry['timestamp']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    # Bot response
    cols_bot = st.columns([1, 3])
    with cols_bot[0]:
        st.markdown(
            f"""
            <div class='bot-bubble'>
            <strong>Bot:</strong> {entry['answer']}<br>
            <div class='citation'>Cited: {', '.join(entry['citations'])}</div>
            <div class='timestamp'>{entry['timestamp']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        # Display images if present
        for img in entry.get('images', []):
            st.image(
                f"data:image/png;base64,{img['img_b64']}",
                caption=f"{img['source']} (page {img['page']})",
                use_container_width=True
            )

st.markdown("---")

# Question input
st.subheader("Ask your PDFs 📝")
query = st.text_input("Your question:")

if st.button("💡 Ask"):
    if not query.strip():
        st.warning("Please type a question.")
    elif not st.session_state.uploaded_files:
        st.warning("Upload at least one PDF first.")
    else:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{API_URL}/ask",
                json={"text": query},
                timeout=60
            )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.chat_history.append({
                "question": query,
                "answer": data.get("answer", ""),
                "citations": data.get("citations", []),
                "images": data.get("images", []),
                "timestamp": timestamp
            })
            st.rerun()
        else:
            st.error(f"Error: {resp.text}")
