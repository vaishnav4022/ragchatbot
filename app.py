"""
app.py
------
Streamlit UI — Bonus requirement
A beautiful chat interface for the RAG chatbot.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import build_rag_pipeline

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot 📄",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Background */
  .stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
  }

  /* Chat messages */
  .user-bubble {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 75%;
    margin-left: auto;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
    font-size: 15px;
    line-height: 1.5;
  }

  .bot-bubble {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    color: #e0e0e0;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 80%;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    font-size: 15px;
    line-height: 1.6;
  }

  .source-badge {
    display: inline-block;
    background: rgba(102,126,234,0.2);
    border: 1px solid rgba(102,126,234,0.4);
    color: #a78bfa;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    margin: 4px 3px 0 0;
  }

  .not-found {
    color: #f97316;
    font-style: italic;
  }

  /* Title */
  h1 {
    background: linear-gradient(135deg, #667eea, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
  }

  /* Input box */
  .stChatInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
    border-radius: 25px !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102,126,234,0.5);
  }

  /* Metrics */
  [data-testid="stMetricValue"] {
    color: #a78bfa !important;
    font-weight: 700 !important;
  }

  /* Divider */
  hr {
    border-color: rgba(255,255,255,0.1) !important;
  }

  /* Status messages */
  .status-ok {
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 8px;
    padding: 10px 14px;
    color: #6ee7b7;
    font-size: 13px;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0
if "found_answers" not in st.session_state:
    st.session_state.found_answers = 0


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 RAG Chatbot")
    st.markdown("*Powered by LangChain + FAISS + Gemini*")
    st.divider()

    # API Key input
    st.markdown("### 🔑 API Key")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Get your free key at https://aistudio.google.com/app/apikey",
        placeholder="AIza..."
    )
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    # PDF Upload
    st.markdown("### 📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF — lecture notes, textbooks, articles, etc."
    )

    if uploaded_file and api_key:
        if st.button("🚀 Process PDF", use_container_width=True):
            with st.spinner("⚙️ Building RAG pipeline..."):
                # Save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    st.session_state.qa_chain = build_rag_pipeline(tmp_path)
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.messages = []   # Reset chat
                    st.session_state.total_questions = 0
                    st.session_state.found_answers = 0
                    st.success("✅ Pipeline ready! Start chatting.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                finally:
                    os.unlink(tmp_path)
    elif uploaded_file and not api_key:
        st.warning("⚠️ Please enter your Gemini API key first.")

    st.divider()

    # Stats
    if st.session_state.qa_chain:
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", st.session_state.total_questions)
        with col2:
            st.metric("Found", st.session_state.found_answers)

        st.markdown(f"""
        <div class="status-ok">
          ✅ Active: <strong>{st.session_state.pdf_name}</strong>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_questions = 0
            st.session_state.found_answers = 0
            st.rerun()

    st.divider()

    # Pipeline info
    st.markdown("### ⚙️ Pipeline")
    st.markdown("""
    | Step | Tool |
    |------|------|
    | Load | PyPDFLoader |
    | Split | RecursiveChar... |
    | Embed | all-MiniLM-L6-v2 |
    | Store | FAISS (local) |
    | LLM | Gemini 1.5 Flash |
    """)


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("# 🤖 RAG Chatbot")
st.markdown("Ask questions about your uploaded PDF document. Answers are grounded **only** in the document — no hallucination.")
st.divider()

# Welcome / empty state
if not st.session_state.qa_chain:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px;">
          <div style="font-size: 80px; margin-bottom: 20px;">📄</div>
          <h3 style="color: #a78bfa; margin-bottom: 10px;">Upload a PDF to get started</h3>
          <p style="color: rgba(255,255,255,0.5); font-size: 14px;">
            1. Enter your Gemini API key in the sidebar<br>
            2. Upload any PDF document<br>
            3. Click "Process PDF"<br>
            4. Ask questions!
          </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                answer = msg["content"]
                css_class = "not-found" if "Not found in document" in answer else ""
                icon = "⚠️" if "Not found in document" in answer else "🤖"
                sources_html = ""
                if msg.get("sources"):
                    sources_html = "<br><div style='margin-top:8px'>"
                    for s in msg["sources"]:
                        sources_html += f'<span class="source-badge">📄 Page {s}</span>'
                    sources_html += "</div>"
                st.markdown(
                    f'<div class="bot-bubble">{icon} <span class="{css_class}">{answer}</span>{sources_html}</div>',
                    unsafe_allow_html=True
                )

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.total_questions += 1

        # Get answer from RAG chain
        with st.spinner("🔍 Searching document..."):
            try:
                result = st.session_state.qa_chain.invoke({"query": prompt})
                answer = result["result"].strip()
                source_docs = result.get("source_documents", [])

                # Extract page numbers from source documents
                pages = sorted(set(
                    doc.metadata.get("page", 0) + 1
                    for doc in source_docs
                ))

                if "Not found in document" not in answer:
                    st.session_state.found_answers += 1

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": pages,
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {str(e)}",
                    "sources": [],
                })

        st.rerun()

    # Suggested questions (when chat is empty)
    if st.session_state.qa_chain and len(st.session_state.messages) == 0:
        st.markdown("#### 💡 Try asking:")
        suggestions = [
            "What is the main topic of this document?",
            "Summarize the key points",
            "What are the most important concepts?",
            "What examples are mentioned?",
        ]
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(f"💬 {suggestion}", key=f"sug_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    st.session_state.total_questions += 1
                    with st.spinner("🔍 Searching document..."):
                        result = st.session_state.qa_chain.invoke({"query": suggestion})
                        answer = result["result"].strip()
                        source_docs = result.get("source_documents", [])
                        pages = sorted(set(
                            doc.metadata.get("page", 0) + 1
                            for doc in source_docs
                        ))
                        if "Not found in document" not in answer:
                            st.session_state.found_answers += 1
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": pages,
                        })
                    st.rerun()
