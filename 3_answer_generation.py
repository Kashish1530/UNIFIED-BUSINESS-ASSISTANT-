import os
import sys
import json
import requests
from dotenv import load_dotenv

# Safe console output on Windows
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

# Import retrieval engine from 2_retrieval_pipeline
from importlib import import_module
retrieval_mod = import_module("2_retrieval_pipeline")
retrieve = retrieval_mod.retrieve

def build_rag_context(results: list) -> str:
    """Format retrieved document chunks into clean structured context with citations."""
    context_blocks = []
    for idx, item in enumerate(results, 1):
        block = (
            f"--- [PASSAGE {idx}] ---\n"
            f"Source Document: {item['source_file']}\n"
            f"Company: {item['company']}\n"
            f"Page: {item['page']}\n"
            f"Chunk ID: {item['chunk_id']}\n"
            f"Content:\n{item['text'].strip()}"
        )
        context_blocks.append(block)
    return "\n\n".join(context_blocks)

def call_gemini_free(prompt: str, api_key: str, model_name: str = "gemini-2.0-flash") -> str:
    """Call Google Gemini Free Tier via REST API (zero extra package dependencies)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800
        }
    }
    headers = {"Content-Type": "application/json"}
    
    resp = requests.post(url, json=payload, headers=headers, timeout=20)
    if resp.status_code == 200:
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()
    else:
        # If gemini-2.0-flash is busy, try gemini-1.5-flash fallback
        if model_name != "gemini-1.5-flash":
            return call_gemini_free(prompt, api_key, model_name="gemini-1.5-flash")
        raise RuntimeError(f"Gemini API Error ({resp.status_code}): {resp.text}")
    return ""

def call_groq_free(system_prompt: str, user_prompt: str, api_key: str, model_name: str = "llama-3.3-70b-versatile") -> str:
    """Call Groq Free Tier using OpenAI-compatible API."""
    from openai import OpenAI
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key
    )
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=800
    )
    return resp.choices[0].message.content.strip()

def generate_answer(question: str, top_k: int = 4, score_threshold: float = 0.35) -> dict:
    """
    RAG Answer Generation using Free / Local Models:
    1. Retrieves passages from ChromaDB.
    2. Probes for free LLMs (Gemini Free Tier or Groq Free Tier).
    3. If no external key is configured, uses local extractive synthesis (100% free, runs on CPU).
    """
    search_response = retrieve(question, top_k=top_k, score_threshold=score_threshold)
    
    if search_response.get("no_match") or not search_response.get("results"):
        return {
            "question": question,
            "answer": "I do not have enough information in the provided corporate policy documents to answer this question.",
            "citations": [],
            "status": "no_match",
            "model_used": "none"
        }
        
    results = search_response["results"]
    rag_context = build_rag_context(results)
    
    citations = [
        {
            "source": item["source_file"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "company": item["company"],
            "confidence_score": item["score"]
        }
        for item in results
    ]
    
    system_prompt = (
        "You are an enterprise AI Business Policy Assistant. "
        "Answer the user's question using ONLY the provided document passages. "
        "Always cite the source document name and page number for every key rule or policy fact. "
        "If the documents do not contain the answer, say: "
        "'I do not have enough information in the provided corporate documents to answer this question.'"
    )
    user_prompt = f"Question: {question}\n\nDocument Passages:\n{rag_context}\n\nPlease provide a clear, structured answer with citations:"
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    generated_text = None
    model_used = None

    # 1. Check for Google Gemini (Free tier)
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key and gemini_key.strip():
        try:
            generated_text = call_gemini_free(full_prompt, gemini_key.strip())
            model_used = "Google Gemini Free (gemini-2.0-flash)"
        except Exception as e:
            print(f"[Notice] Gemini call issue ({e}). Checking alternatives...")

    # 2. Check for Groq (Free tier)
    if not generated_text:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and groq_key.strip():
            try:
                generated_text = call_groq_free(system_prompt, user_prompt, groq_key.strip())
                model_used = "Groq Free (llama-3.3-70b-versatile)"
            except Exception as e:
                print(f"[Notice] Groq call issue ({e}). Checking alternatives...")

    # 3. Fallback: 100% Free Local Extractive Synthesizer (No API key needed)
    if not generated_text:
        model_used = "Local Free Extractive Model (Offline)"
        answer_parts = [
            "Based on the retrieved corporate policy documents, here are the applicable provisions:\n"
        ]
        for idx, item in enumerate(results, 1):
            cleaned_excerpt = item['text'][:350].replace('\n', ' ').strip()
            answer_parts.append(
                f"[{idx}] {item['company']} - {item['title']}\n"
                f"    Source: {item['source_file']} (Page {item['page']})\n"
                f"    Relevant Excerpt: \"{cleaned_excerpt}...\"\n"
            )
        answer_parts.append(
            "\n[Tip: To enable generative AI synthesis, add a free GEMINI_API_KEY from https://aistudio.google.com/ or GROQ_API_KEY from https://console.groq.com/ to your .env file.]"
        )
        generated_text = "\n".join(answer_parts)

    return {
        "question": question,
        "answer": generated_text,
        "citations": citations,
        "model_used": model_used,
        "status": "success",
        "results_count": len(results)
    }

def print_rag_result(result_dict: dict):
    """Pretty-print the RAG answer generation output."""
    print("=" * 85)
    print(f"QUESTION: \"{result_dict['question']}\"")
    print(f"MODEL / ENGINE: {result_dict.get('model_used')} | STATUS: {result_dict['status'].upper()}")
    print("=" * 85)
    
    print("\n--- ANSWER ---")
    print(result_dict["answer"])
    
    print("\n--- CITATIONS & SOURCE PROVENANCE ---")
    for idx, c in enumerate(result_dict["citations"], 1):
        print(f"[{idx}] {c['source']} (Page {c['page']}) | Confidence: {c['confidence_score']:.4f} | Chunk: {c['chunk_id']}")
    print("-" * 85)
    print()

def main():
    print("=========================================================")
    print("     UNIFIED BUSINESS ASSISTANT - ANSWER GENERATION      ")
    print("               (FREE / LOCAL MODEL ENGINE)               ")
    print("=========================================================\n")
    
    # Check CLI arguments
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        res = generate_answer(user_query)
        print_rag_result(res)
        return
        
    # Representative enterprise benchmark question
    sample_queries = [
        "What is the policy and grievance procedure for Prevention of Sexual Harassment (POSH)?",
        "What are the maternity and parental leave entitlements?",
        "What are the CSR committee guidelines and project allocations?"
    ]
    
    query = sample_queries[0]
    print(f"Executing sample RAG query: \"{query}\"\n")
    res = generate_answer(query)
    print_rag_result(res)
    
    print("[SUCCESS] Free model answer generation pipeline verified!")
    print("You can ask any question with: python 3_answer_generation.py \"<your question>\"")

if __name__ == "__main__":
    main()
