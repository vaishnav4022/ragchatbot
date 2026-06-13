"""
rag_pipeline.py
---------------
Core RAG pipeline:
  Step 2 — Load PDF with PyPDFLoader
  Step 3 — Chunk text with RecursiveCharacterTextSplitter
  Step 4 — Embed & Store in FAISS vector index
  Step 5 — Wire retriever into RetrievalQA chain (Gemini LLM)
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Load environment variables ────────────────────────────────────────────────
load_dotenv()


# ── Step 2: Load PDF ──────────────────────────────────────────────────────────
def load_pdf(pdf_path: str):
    """Load a PDF file and return a list of Document objects (one per page)."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[Load] Loaded {len(documents)} page(s) from '{pdf_path}'")
    return documents


# ── Step 3: Split into chunks ─────────────────────────────────────────────────
def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    chunk_size=1000 chars, chunk_overlap=200 chars for context continuity.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[Split] Created {len(chunks)} chunk(s) from {len(documents)} page(s)")
    return chunks


# ── Step 4: Embed & Store in FAISS ────────────────────────────────────────────
def create_vectorstore(chunks):
    """
    Build a FAISS vector index from document chunks using
    HuggingFace sentence-transformers (runs 100% locally, no API key needed).
    Model: all-MiniLM-L6-v2 — fast, lightweight, great semantic search quality.
    """
    print("[Embed] Loading HuggingFace embeddings model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("[Store] Building FAISS vector index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"[Store] FAISS index built with {vectorstore.index.ntotal} vectors")
    return vectorstore


# ── Step 5: Build RetrievalQA Chain ──────────────────────────────────────────
def build_qa_chain(vectorstore, k: int = 4):
    """
    Wire the FAISS retriever into a RetrievalQA chain backed by Gemini 1.5 Flash.
    
    Anti-hallucination prompt:
      - Instructs the model to answer ONLY from retrieved context
      - If the answer isn't in the document, return 'Not found in document'
    """
    # Custom anti-hallucination prompt
    PROMPT_TEMPLATE = """You are a helpful assistant that answers questions strictly based on the provided document context.

RULES:
1. Answer ONLY using the information from the context below.
2. If the answer is not found in the context, respond with exactly: "Not found in document"
3. Do NOT make up or infer information beyond what is in the context.
4. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    # Gemini Flash Lite — free tier, confirmed working
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",
        temperature=0.1,          # Low temp for factual accuracy
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Retriever: fetch top-k most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",           # Stuff all chunks into one prompt
        retriever=retriever,
        return_source_documents=True, # Return source chunks for transparency
        chain_type_kwargs={"prompt": prompt},
    )
    print("[Chain] RetrievalQA chain ready!")
    return qa_chain


# ── Main pipeline builder ─────────────────────────────────────────────────────
def build_rag_pipeline(pdf_path: str):
    """
    Full pipeline:
      PDF → Load → Split → Embed → FAISS → RetrievalQA chain
    Returns the ready-to-query qa_chain.
    """
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    vectorstore = create_vectorstore(chunks)
    qa_chain = build_qa_chain(vectorstore)
    return qa_chain


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_notes.pdf"

    print("\n" + "="*60)
    print("  RAG Chatbot — CLI Mode")
    print("="*60)

    chain = build_rag_pipeline(pdf_path)

    # Step 6: Test with 5 sample questions
    test_questions = [
        "What is the main topic of this document?",
        "What are the key concepts explained?",
        "Summarize the most important points.",
        "What examples are provided?",
        "What is machine learning?",          # May or may not be in doc
    ]

    print("\n--- Running 5 test questions ---\n")
    for i, q in enumerate(test_questions, 1):
        print(f"Q{i}: {q}")
        result = chain.invoke({"query": q})
        print(f"A{i}: {result['result']}")
        print(f"    [Sources: {len(result['source_documents'])} chunk(s) retrieved]")
        print()
