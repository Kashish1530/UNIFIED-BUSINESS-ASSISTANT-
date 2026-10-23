import os
import sys
import json
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Configure Windows console encoding safety
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment variables
load_dotenv()
parent_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(parent_env):
    load_dotenv(parent_env)

# Global caches for instant sub-second retrieval queries
_CACHED_EF = None
_CACHED_MODEL_NAME = None
_CACHED_COLLECTION = None

def get_embedding_function():
    """
    Select and cache the embedding function matching the ingestion pipeline:
    - Uses local Chroma ONNX all-MiniLM-L6-v2 by default or OpenAI if quota allows.
    - Caches instance to avoid probing delays on repeated queries.
    """
    global _CACHED_EF, _CACHED_MODEL_NAME
    if _CACHED_EF is not None:
        return _CACHED_EF, _CACHED_MODEL_NAME

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and openai_key != "your_openai_key_here":
        try:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small"
            )
            ef(["test connection"])
            _CACHED_EF = ef
            _CACHED_MODEL_NAME = "text-embedding-3-small"
            return _CACHED_EF, _CACHED_MODEL_NAME
        except Exception:
            pass
            
    # Default local fast ONNX embedding function
    _CACHED_EF = embedding_functions.DefaultEmbeddingFunction()
    _CACHED_MODEL_NAME = "all-MiniLM-L6-v2"
    return _CACHED_EF, _CACHED_MODEL_NAME

def get_vector_store():
    """
    Initialize and cache persistent ChromaDB client pointing to rag_documents collection.
    """
    global _CACHED_COLLECTION, _CACHED_MODEL_NAME
    if _CACHED_COLLECTION is not None:
        return _CACHED_COLLECTION, _CACHED_MODEL_NAME

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_1 = os.path.join(curr_dir, "db", "chroma_db")
    candidate_2 = os.path.join(curr_dir, "..", "knowledge_service", "db", "chroma_db")
    
    db_path = candidate_1 if os.path.exists(candidate_1) else candidate_2
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"ChromaDB store not found at '{db_path}'. Please run 1_ingestion_pipeline.py first."
        )
        
    embedding_func, model_name = get_embedding_function()
    client = chromadb.PersistentClient(path=db_path)
    
    _CACHED_COLLECTION = client.get_collection(
        name="rag_documents",
        embedding_function=embedding_func
    )
    return _CACHED_COLLECTION, model_name

def retrieve(question: str, top_k: int = 5, score_threshold: float = 0.4):
    """
    Query ChromaDB for relevant policy chunks conforming to the Role A API Specification
    (UNIFIED-BUSINESS-ASSISTAN.md Section 3.4).

    Returns:
        dict: {
            "results": [
                {
                    "text": str,
                    "source_file": str,
                    "page": int,
                    "score": float,
                    "chunk_id": str,
                    "title": str,
                    "company": str
                }
            ],
            "no_match": bool,
            "embedding_model": str
        }
    """
    collection, model_name = get_vector_store()
    
    query_results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    documents = query_results.get("documents", [[]])[0]
    metadatas = query_results.get("metadatas", [[]])[0]
    distances = query_results.get("distances", [[]])[0]
    
    results = []
    for doc_text, meta, dist in zip(documents, metadatas, distances):
        # Convert cosine distance to cosine similarity (1.0 - distance)
        similarity = round(max(0.0, 1.0 - dist), 4)
        
        if similarity >= score_threshold:
            results.append({
                "text": doc_text,
                "source_file": meta.get("source", ""),
                "page": int(meta.get("page", 1)),
                "score": similarity,
                "chunk_id": meta.get("chunk_id", ""),
                "title": meta.get("title", ""),
                "company": meta.get("company", "")
            })
            
    no_match = len(results) == 0
    return {
        "results": results,
        "no_match": no_match,
        "embedding_model": model_name
    }

def print_retrieval_output(question: str, response: dict):
    """Format and display search response for inspection."""
    print("=" * 85)
    print(f"QUESTION: \"{question}\"")
    print(f"EMBEDDING MODEL: {response['embedding_model']} | NO_MATCH: {response['no_match']} | RESULTS RETURNED: {len(response['results'])}")
    print("=" * 85)
    
    if response["no_match"]:
        print("[!] No document chunks met the relevance threshold.\n")
        return
        
    for idx, item in enumerate(response["results"], 1):
        print(f"\n[{idx}] Score: {item['score']:.4f} | Source: {item['source_file']} (Page {item['page']})")
        print(f"    Chunk ID: {item['chunk_id']} | Company: {item['company']}")
        print(f"    Title:    {item['title']}")
        full_text = item['text'].strip()
        # Indent content nicely for console readability
        indented_lines = "\n    ".join(full_text.split("\n"))
        print(f"    Content:\n    {indented_lines}")
        print("    " + "-" * 75)
    print()

def main():
    print("=========================================================")
    print("      UNIFIED BUSINESS ASSISTANT - RETRIEVAL PIPELINE    ")
    print("=========================================================\n")
    
    # Check if a custom question was passed as a command line argument
    if len(sys.argv) > 1:
        custom_question = " ".join(sys.argv[1:])
        resp = retrieve(custom_question, top_k=5, score_threshold=0.35)
        print_retrieval_output(custom_question, resp)
        return
        
    # Standard test benchmark suite covering our corporate policy corpus
    test_suite = [
        "What is the procedure for reporting sexual harassment under POSH?",
        "What are the maternity and parental leave entitlements?",
        "What are the CSR committee guidelines and project allocations?",
        "What is the policy regarding gifts, hospitality, and anti-bribery?",
        "What are the whistleblower protection and vigil mechanism rules?"
    ]
    
    print(f"Running retrieval test suite across {len(test_suite)} policy queries...\n")
    
    for q in test_suite:
        resp = retrieve(q, top_k=3, score_threshold=0.35)
        print_retrieval_output(q, resp)
        
    print("[SUCCESS] All retrieval queries executed successfully!")
    print("You can run custom queries anytime with: python 2_retrieval_pipeline.py \"<your question>\"")

if __name__ == "__main__":
    main()
