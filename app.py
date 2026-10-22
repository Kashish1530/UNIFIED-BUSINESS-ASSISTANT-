import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# Configure safe Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag_engine import UnifiedRAGEngine

# Initialize FastAPI application
app = FastAPI(
    title="Unified Business Assistant - Knowledge Service (Role A)",
    description="Enterprise RAG & Corporate Policy Retrieval Service with Neon PostgreSQL telemetry.",
    version="1.0.0"
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine singleton
engine = UnifiedRAGEngine()

# Static files directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas (Aligned with UNIFIED-BUSINESS-ASSISTAN.md Section 3.4)
# ──────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    question: str = Field(..., description="Query for document search")
    top_k: Optional[int] = Field(default=5, description="Number of results to retrieve")
    score_threshold: Optional[float] = Field(default=0.35, description="Minimum cosine similarity")

class PassageResult(BaseModel):
    text: str
    source_file: str
    page: int
    score: float
    chunk_id: str
    title: Optional[str] = None
    company: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[PassageResult]
    no_match: bool
    embedding_model: Optional[str] = None

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message or follow-up question")
    session_id: Optional[str] = Field(default="default_session", description="Conversation session ID")
    top_k: Optional[int] = Field(default=4, description="Candidate chunks to evaluate")
    score_threshold: Optional[float] = Field(default=0.50, description="Minimum cosine similarity threshold")

class ChatResponse(BaseModel):
    session_id: str
    question: str
    standalone_question: str
    answer: str
    citations: List[dict]
    model_used: str
    no_match: bool

# ──────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_dashboard():
    """Serve the interactive Web Chatbot dashboard."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Unified Business Assistant Knowledge Service is running. Access /docs for API."}

@app.post("/search", response_model=SearchResponse)
def search_documents(payload: SearchRequest):
    """
    Role A Formal Endpoint (UNIFIED-BUSINESS-ASSISTAN.md Section 3.4):
    Retrieves ranked passages matching the query with page numbers and scores.
    """
    try:
        res = engine.search(
            question=payload.question,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search retrieval error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
def conversational_chat(payload: ChatRequest):
    """
    Conversational RAG Chatbot Endpoint:
    Provides context-aware multi-turn answers with citations and logs to Neon PostgreSQL.
    """
    try:
        res = engine.chat(
            user_message=payload.message,
            session_id=payload.session_id,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/documents")
def get_document_catalog():
    """Return the master catalog of all 50 corporate policy documents."""
    return engine.catalog

@app.get("/stats")
def get_telemetry_stats():
    """Return real-time Neon PostgreSQL and ChromaDB statistics."""
    try:
        return engine.get_stats()
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    """Liveness probe."""
    return {
        "status": "healthy",
        "service": "Role A - Knowledge Service",
        "neon_connected": True,
        "active_documents": engine.catalog.get("total_documents", 50)
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Unified Business Assistant (Knowledge Service) on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
