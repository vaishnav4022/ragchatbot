# 📄 RAG Chatbot with LangChain

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions strictly from an uploaded PDF document — no hallucination.

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| Framework | LangChain |
| Vector Store | FAISS (runs locally) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | Google Gemini Flash Lite (free tier) |
| PDF Loader | PyPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| UI | Streamlit |

---

## ⚙️ Pipeline Explained

```
PDF File
   │
   ▼
[Step 2] PyPDFLoader ──── Loads all pages as Document objects
   │
   ▼
[Step 3] RecursiveCharacterTextSplitter
         chunk_size=1000, chunk_overlap=200
         ──── Splits into smaller overlapping chunks
   │
   ▼
[Step 4] HuggingFaceEmbeddings (all-MiniLM-L6-v2)
         ──── Converts each chunk → 384-dim dense vector
         FAISS.from_documents ──── Builds local vector index
   │
   ▼
[Step 5] RetrievalQA Chain
         User Query → Embed query → Search FAISS (top-4 chunks)
         → Stuff chunks into prompt → Gemini 1.5 Flash → Answer
   │
   ▼
[Anti-hallucination] Custom prompt:
  "If not found in context, say 'Not found in document'"
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Gemini API key
Visit [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — it's free!

### 3. Set your API key
```bash
cp .env.example .env
# Edit .env and paste your key:
# GOOGLE_API_KEY=AIza...
```

### 4. (Optional) Generate sample PDF
```bash
python create_sample_pdf.py
```

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

### 6. (Optional) CLI mode
```bash
python rag_pipeline.py sample_notes.pdf
```

---

## 💬 Verified Q&A Results (Step 6 — grounded on sample_notes.pdf)

| # | Question | Answer | Status |
|---|----------|--------|--------|
| 1 | What is Machine Learning and who coined the term? | ML is a subset of AI... Arthur Samuel coined the term in 1959. | ✅ GROUNDED |
| 2 | What is RAG and who introduced it? | RAG enhances LLMs by retrieving from an external knowledge base... introduced by Lewis et al. from Facebook AI Research in 2020. | ✅ GROUNDED |
| 3 | What are the types of AI mentioned in the document? | Narrow AI (ANI), General AI (AGI), and Super AI (ASI). | ✅ GROUNDED |
| 4 | What is the F1-Score used for? | Harmonic mean of precision and recall. Good for imbalanced datasets. | ✅ GROUNDED |
| 5 | What is the capital of France? | **Not found in document** | ✅ NO HALLUCINATION |

---

## 📁 Project Structure

```
RAG agent/
├── app.py                  # Streamlit UI
├── rag_pipeline.py         # Core RAG pipeline
├── create_sample_pdf.py    # Sample PDF generator
├── requirements.txt        # Dependencies
├── .env.example            # API key template
├── .env                    # Your API key (not committed)
├── sample_notes.pdf        # Sample document (generated)
└── README.md               # This file
```

---

## 🧩 Key Design Decisions

- **HuggingFace Embeddings** (not OpenAI) → Runs 100% locally, no cost
- **FAISS** → Fast in-memory vector search, no external DB needed
- **Gemini 1.5 Flash** → Free tier, fast responses, great quality
- **chunk_overlap=200** → Prevents answers from being cut off at chunk boundaries
- **temperature=0.1** → Low temperature for factual, consistent answers
- **Custom prompt** → Strict instruction to not hallucinate; return "Not found in document"

---

## 📸 Demo

Run `streamlit run app.py`, upload any PDF, and start asking questions!

---

*Built for Day 3 Take Home Project — RAG Chatbot Assignment*
