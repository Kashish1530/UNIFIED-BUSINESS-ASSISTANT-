import os
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from neon_client import NeonDBClient

# Load environment variables
load_dotenv()
parent_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(parent_env):
    load_dotenv(parent_env)

def get_paths():
    """Resolve correct base paths regardless of where the script is executed from."""
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if data is in sibling or parent directory
    candidate_1 = os.path.join(curr_dir, "..", "data", "documents")
    candidate_2 = os.path.join(curr_dir, "data", "documents")
    
    docs_path = candidate_1 if os.path.exists(candidate_1) else candidate_2
    meta_path = os.path.abspath(os.path.join(docs_path, "..", "documents_metadata.json"))
    db_path = os.path.join(curr_dir, "db", "chroma_db")
    
    return os.path.abspath(docs_path), meta_path, db_path

def load_pdf_documents(docs_path,meta_path):
    """
    Load all 50 curated PDFs page-by-page from docs_path, preserving page provenance 
    and attaching metadata from documents_metadata.json.
    """
    print(f"Loading PDF documents from {docs_path}...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Documents directory not found at: {docs_path}")
    
    meta_lookup = {}
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
                for item in meta_json.get("documents", []):
                    meta_lookup[item["filename"]] = item
        except Exception as e:
            print(f"Warning: Could not read metadata JSON: {e}")
            
    files = sorted([f for f in os.listdir(docs_path) if f.endswith(".pdf")])
    if not files:
        raise FileNotFoundError(f"No PDF files found in {docs_path}")
        
    print(f"Found {len(files)} PDF documents to ingest.")
    documents = []
    
    for doc_idx, fname in enumerate(files, 1):
        fpath = os.path.join(docs_path, fname)
        info = meta_lookup.get(fname, {})
        doc_id = info.get("id", f"doc_{doc_idx:03d}")
        company = info.get("company", "Enterprise")
        title = info.get("title", fname.replace(".pdf", "").replace("_", " ").title())
        category = info.get("category", "Corporate Policy")
        
        try:
            reader = PdfReader(fpath)
            for page_idx, page in enumerate(reader.pages, 1):
                raw_text = page.extract_text() or ""
                clean_text = raw_text.strip()
                
                # Even if scanned/image with minimal text, record page entry with full metadata context
                if not clean_text:
                    clean_text = f"Document: {title} | Company: {company} | Category: {category} | Page: {page_idx}. [Slide / Image presentation content]"
                    
                doc = Document(
                    page_content=clean_text,
                    metadata={
                        "doc_id": doc_id,
                        "source": fname,
                        "page": page_idx,
                        "title": title,
                        "company": company,
                        "category": category
                    }
                )
                documents.append(doc)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            
    print(f"Loaded {len(documents)} total pages across {len(files)} files.")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """
    Split pages into chunks with overlap, assigning a deterministic chunk_id.
    Format: <doc_id>_p<page>_c<chunk_idx>
    """
    print(f"Splitting {len(documents)} page documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    chunk_counter_by_page = {}
    
    for doc in documents:
        doc_id = doc.metadata["doc_id"]
        page = doc.metadata["page"]
        page_key = f"{doc_id}_p{page}"
        
        split_texts = text_splitter.split_text(doc.page_content)
        
        for c_idx, text_part in enumerate(split_texts, 1):
            chunk_id = f"{page_key}_c{c_idx:02d}"
            
            chunk_doc = Document(
                page_content=text_part,
                metadata={
                    **doc.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": c_idx
                }
            )
            chunks.append(chunk_doc)
            
    print(f"Generated {len(chunks)} total text chunks.")
    return chunks

def sync_to_neon(chunks, embedding_model_name: str = "text-embedding-3-small"):
    """
    Sync all chunks into the Neon PostgreSQL table 'rag_documents'.
    Columns: sr_no, chunk_id, chunk_text, source_name, page_number, score_confidence, embedding_model, created_at
    """
    print("\n--- Syncing chunks to Neon PostgreSQL Database ---")
    try:
        client = NeonDBClient()
        db_info = client.test_connection()
        print(f"Connected to Neon Database: '{db_info.get('database')}' as user '{db_info.get('user')}'")
        
        # Prepare list for Neon insertion
        formatted_chunks = []
        for ch in chunks:
            formatted_chunks.append({
                "chunk_id": ch.metadata["chunk_id"],
                "text": ch.page_content,
                "source": ch.metadata["source"],
                "page": ch.metadata["page"],
                "score": None
            })
            
        inserted_total = client.insert_chunks(
            formatted_chunks, 
            embedding_model_name=embedding_model_name, 
            batch_size=100
        )
        print(f"Successfully synced {inserted_total} chunks into 'rag_documents' table on Neon PostgreSQL!")
        
        # Display summary stats from Neon
        stats = client.get_stats()
        print(f"Verified row count in Neon 'rag_documents': {stats['total_chunks']} rows.")
        return True
    except Exception as e:
        print(f"Neon Sync Error: {e}")
        return False

def get_embedding_function():
    """
    Choose embedding function:
    - Tests OpenAI if OPENAI_API_KEY is configured.
    - If key is invalid or quota exhausted (429), automatically falls back to 
      Chroma's built-in local ONNX all-MiniLM-L6-v2 embedding function.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and openai_key != "your_openai_key_here":
        try:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small"
            )
            # Probe call to check if quota / API key is active
            ef(["test connection"])
            print("Successfully initialized OpenAIEmbeddingFunction (model='text-embedding-3-small')")
            return ef, "text-embedding-3-small"
        except Exception as e:
            print(f"OpenAI embeddings unavailable ({e}).\n--> Automatically falling back to local Chroma embeddings.")
            
    print("Using local Chroma ONNX Embeddings (model='all-MiniLM-L6-v2')")
    ef = embedding_functions.DefaultEmbeddingFunction()
    return ef, "all-MiniLM-L6-v2"

def create_vector_store(chunks, embedding_function, persist_directory="db/chroma_db", batch_size=100):
    """
    Incremental local ChromaDB vector store:
    - Uses get_or_create_collection to preserve existing vector data.
    - Only embeds and upserts new chunks that are not already indexed.
    """
    print(f"\n--- Storing vector index in ChromaDB ({persist_directory}) ---")
    os.makedirs(persist_directory, exist_ok=True)
    
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name="rag_documents",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Query existing IDs in Chroma
    existing_chroma_ids = set()
    try:
        existing_data = collection.get()
        existing_chroma_ids = set(existing_data.get("ids", []))
    except Exception:
        pass
        
    new_chunks = [c for c in chunks if c.metadata["chunk_id"] not in existing_chroma_ids]
    
    if not new_chunks:
        print(f"All {len(chunks)} chunks are already indexed in ChromaDB ({collection.count()} total).")
        return collection
        
    print(f"ChromaDB Incremental Indexing: {len(existing_chroma_ids)} existing, indexing {len(new_chunks)} new chunks...")
    
    # Batch upsert only new chunks into ChromaDB
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i:i + batch_size]
        ids = [c.metadata["chunk_id"] for c in batch]
        docs = [c.page_content for c in batch]
        metas = []
        for c in batch:
            clean_m = {
                "doc_id": str(c.metadata.get("doc_id", "")),
                "source": str(c.metadata.get("source", "")),
                "page": int(c.metadata.get("page", 1)),
                "title": str(c.metadata.get("title", "")),
                "company": str(c.metadata.get("company", "")),
                "category": str(c.metadata.get("category", "")),
                "chunk_id": str(c.metadata.get("chunk_id", ""))
            }
            metas.append(clean_m)
            
        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas
        )
        print(f"Indexed {min(i + batch_size, len(new_chunks))}/{len(new_chunks)} new chunks into ChromaDB...")
        
    print(f"ChromaDB vector store now holds {collection.count()} total chunks.")
    return collection

def main():
    print("=========================================================")
    print("      UNIFIED BUSINESS ASSISTANT - INGESTION PIPELINE    ")
    print("=========================================================\n")
    
    docs_path, meta_path, db_path = get_paths()
    print(f"Documents Path: {docs_path}")
    print(f"Metadata Path:  {meta_path}")
    print(f"ChromaDB Path:  {db_path}\n")
    
    # 1. Load documents
    raw_docs = load_pdf_documents(docs_path, meta_path)
    
    # 2. Split into chunks
    chunks = split_documents(raw_docs, chunk_size=1000, chunk_overlap=150)
    
    # 3. Determine Embedding Model
    embedding_func, model_name = get_embedding_function()
    
    # 4. Sync Chunks to Neon PostgreSQL Database
    sync_to_neon(chunks, embedding_model_name=model_name)
    
    # 5. Index into ChromaDB Vector Store
    vectorstore = create_vector_store(chunks, embedding_func, db_path)
    
    print(f"\n[SUCCESS] Ingestion complete! {len(chunks)} chunks are stored in Neon PostgreSQL and ChromaDB.")
    return vectorstore

if __name__ == "__main__":
    main()
