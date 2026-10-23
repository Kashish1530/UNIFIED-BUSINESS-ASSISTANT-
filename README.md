# Unified Business Assistant - Role A (Knowledge Service)

Enterprise Retrieval-Augmented Generation (RAG) platform and Policy Assistant powered by **Neon PostgreSQL**, **ChromaDB**, and **FastAPI**.

---

## 🌟 Key Features

1. **50 Real Corporate Policy Documents**:
   - Ingests and indexes 50 multi-page policies across **Susan Electricals**, **Jayaswal Neco**, **Reckitt**, **Google/Alphabet**, and **GitLab**.
   - Spans 6 enterprise domains:
     - *Finance, Securities (UPSI), Audit & Risk Management*
     - *Governance, Ethics & Anti-Corruption*
     - *Cloud, Security, IT & Modern Operations*
     - *HR, Workplace Conduct & POSH*
     - *CSR, ESG & Sustainability*
     - *Supply Chain & Vendor Standards*

2. **Neon PostgreSQL Cloud Database Integration**:
   - Stores all chunks in the `rag_documents` table with page-level provenance.
   - **Incremental Deduplication**: Re-running ingestion never duplicates records.
   - **Deterministic Sequential Numbering**: Guarantees gapless `sr_no` (1, 2, 3... N).
   - **Chat Telemetry & Audit**: Logs every user query, answer, source citation, and confidence score to the `rag_chat_logs` table.

3. **High-Precision Multi-Query Vector Retrieval**:
   - Built on native `chromadb` using local ONNX embeddings (`all-MiniLM-L6-v2`) — **100% free, runs offline on CPU**.
   - Query expansion and reciprocal rank fusion to maximize search recall.

4. **Interactive Web Chatbot & API Service**:
   - **Modern Web Dashboard**: Glassmorphic UI with live Neon/Chroma badges, quick prompts, document catalog modal, and citation drawers.
   - **API Specification Compliance**: Strictly implements the `POST /search` contract defined in `UNIFIED-BUSINESS-ASSISTAN.md` Section 3.4.

---

## 📁 Project Structure

```text
knowledge_service/
├── app.py                      # FastAPI server & interactive Web Chatbot UI
├── rag_engine.py               # Core enterprise RAG engine (history, fusion, citations)
├── neon_client.py              # Neon PostgreSQL client (tables, inserts, chat logging)
├── 1_ingestion_pipeline.py     # PDF parsing, chunking, and dual-sync (Neon & Chroma)
├── 2_retrieval_pipeline.py     # Standalone retrieval testing script
├── 3_answer_generation.py      # Free model answer generation testing script
├── requirements.txt            # Cross-platform dependencies
├── .env                        # Database URL & API configuration
├── static/
│   └── index.html              # Sleek Web Chatbot user interface
└── data/
    ├── documents/              # Exactly 50 active corporate policy PDFs
    └── documents_metadata.json # Master 50-document catalog & metadata
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Verify your `knowledge_service/.env` file contains your Neon database connection string:
```ini
DATABASE_URL=postgresql://user:password@endpoint.neon.tech/neondb?sslmode=require
```
*(Optional: Add `GEMINI_API_KEY` or `GROQ_API_KEY` for free generative narrative answers; defaults to local offline extractive RAG if no key is provided).*

---

## 🧪 Pipeline Execution Steps

### Step 1: Run Document Ingestion
Ingests all 50 PDFs, generates text chunks, and synchronizes them to Neon PostgreSQL and ChromaDB:
```bash
python 1_ingestion_pipeline.py
```

### Step 2: Test Retrieval
Test vector similarity search with page-level provenance:
```bash
python 2_retrieval_pipeline.py "What is the policy for reporting sexual harassment under POSH?"
```

### Step 3: Test Answer Generation
Generate policy answers with exact source citations:
```bash
python 3_answer_generation.py "What are the rules regarding gifts and anti-bribery?"
```

---

## 🌐 Launch the Web Chatbot Application

Start the production server:
```bash
python app.py
```
Or with Uvicorn:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Web Chatbot UI**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- **Interactive Swagger API Docs**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Interactive Web Chatbot Dashboard |
| `POST` | `/search` | Formal Role A Search Endpoint (`UNIFIED-BUSINESS-ASSISTAN.md` Section 3.4) |
| `POST` | `/chat` | Conversational RAG Chatbot (Session history + Citations) |
| `GET` | `/documents` | Master catalog of all 50 corporate policy documents |
| `GET` | `/stats` | Live Neon PostgreSQL and ChromaDB chunk count metrics |
| `GET` | `/health` | Service health probe |
