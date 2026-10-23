import os
import sys
import json
import re
import requests
from dotenv import load_dotenv

# Safe console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment
load_dotenv()
parent_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(parent_env):
    load_dotenv(parent_env)

import chromadb
from chromadb.utils import embedding_functions
from neon_client import NeonDBClient

class UnifiedRAGEngine:
    """
    End-to-End Enterprise RAG Engine for the Unified Business Assistant:
    - History-aware conversational memory
    - Multi-query expansion & rank fusion
    - High-precision provenance tracking (Document, Page, Chunk, Confidence Score)
    - Free model support (Gemini Free, Groq Free, or Local Offline Extractive RAG)
    - Neon PostgreSQL audit & chat telemetry
    """

    def __init__(self, persist_directory=None):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        if persist_directory is None:
            c1 = os.path.join(curr_dir, "db", "chroma_db")
            c2 = os.path.join(curr_dir, "..", "knowledge_service", "db", "chroma_db")
            self.persist_directory = c1 if os.path.exists(c1) else c2
        else:
            self.persist_directory = persist_directory

        self.meta_path = os.path.join(curr_dir, "data", "documents_metadata.json")
        self.neon_client = NeonDBClient()
        self.neon_client.create_table_if_not_exists()
        
        # Initialize embedding function & ChromaDB
        self.ef, self.embedding_model_name = self._init_embedding_function()
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_documents",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load metadata catalog
        self.catalog = self._load_catalog()

    def _init_embedding_function(self):
        """Probe for OpenAI embeddings or fall back to local Chroma ONNX embeddings."""
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key and openai_key != "your_openai_key_here":
            try:
                ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_key,
                    model_name="text-embedding-3-small"
                )
                ef(["test connection"])
                return ef, "text-embedding-3-small"
            except Exception:
                pass
        return embedding_functions.DefaultEmbeddingFunction(), "all-MiniLM-L6-v2"

    def _load_catalog(self):
        """Load document metadata catalog."""
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"total_documents": 0, "documents": []}

    def contextualize_question(self, question: str, history: list) -> str:
        """
        History-Aware Reformulation (Conversational Query Rewriting):
        If conversation history exists, reformulate dependent follow-up questions
        (e.g., 'What about paternity?', 'Who is the chairperson?', 'How do I report it?')
        into standalone, context-complete queries for vector retrieval.
        """
        if not history:
            return question.strip()

        last_turn = history[-1]
        last_user = last_turn.get("user_query", "").strip()
        last_bot = last_turn.get("bot_response", "").strip()

        # 1. LLM-Based Reformulation (if free Gemini or Groq key is present)
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        if (gemini_key and gemini_key.strip()) or (groq_key and groq_key.strip()):
            try:
                system_prompt = "You are a search query reformulation engine. Given chat history and a follow-up question, rewrite it into a single standalone search query. Do NOT answer. Output only the rewritten query."
                history_str = f"User: {last_user}\nAssistant: {last_bot[:250]}"
                user_prompt = f"Chat History:\n{history_str}\n\nFollow-up Question: {question}\n\nStandalone Search Query:"
                rewritten, _ = self._call_llm_free(user_prompt, system_prompt)
                if rewritten and len(rewritten.strip()) > 3:
                    clean_rewritten = rewritten.strip().replace('"', '').strip()
                    return clean_rewritten
            except Exception:
                pass

        # 2. Rule-Based Offline Reformulation (Fast, Zero-Dependency)
        lower_q = question.lower().strip()
        words = lower_q.split()
        
        # Pronouns or explicit dependent reference words
        explicit_triggers = [
            "it", "this", "that", "these", "those", "they", "them", "its", "their",
            "what about", "how about", "and for", "same for", "what if",
            "who is the", "who are the", "who is their", "how do i", "how can i",
            "can they", "is it", "does it", "can it", "more details", "elaborate"
        ]

        # Must have an explicit pronoun/continuation trigger OR be an ultra-short fragment (< 4 words)
        is_dependent = (
            any(t in lower_q for t in ["what about", "how about", "same for", "what if", "more details", "elaborate"])
            or any(f" {t} " in f" {lower_q} " for t in ["it", "this", "that", "these", "those", "they", "them", "its", "their"])
            or (len(words) <= 3 and lower_q.endswith("?"))
        )

        if is_dependent and last_user:
            # Extract key topic context from the prior user query
            clean_prev = re.sub(r'[^a-zA-Z0-9\s]', '', last_user)
            # Remove common conversational query filler words
            stop_words = {"what", "is", "the", "policy", "for", "at", "in", "of", "and", "a", "an", "tell", "me", "about", "can", "you", "do", "know"}
            prev_keywords = [w for w in clean_prev.split() if w.lower() not in stop_words]
            topic_hint = " ".join(prev_keywords) if prev_keywords else clean_prev
            return f"{question} regarding {topic_hint}".strip()

        return question.strip()


    def expand_queries(self, query: str) -> list:
        """
        Multi-Query expansion:
        Generates domain-relevant synonyms and variations across the 6 policy domains.
        """
        queries = [query]
        lower_q = query.lower()
        
        if "posh" in lower_q or "harassment" in lower_q:
            queries.append("Prevention of Sexual Harassment workplace policy internal complaints committee procedure redressal")
        elif "leave" in lower_q or "maternity" in lower_q or "parental" in lower_q or "paternity" in lower_q:
            queries.append("parental leave policy maternity benefit entitlement primary secondary caregiver adoption")
        elif "csr" in lower_q or "social responsibility" in lower_q:
            queries.append("corporate social responsibility policy annual action plan committee allocation")
        elif "bribery" in lower_q or "gift" in lower_q or "corruption" in lower_q:
            queries.append("anti-corruption anti-bribery code of conduct gifts hospitality entertaining")
        elif "whistleblower" in lower_q or "vigil" in lower_q:
            queries.append("vigil mechanism whistleblower policy protected disclosure investigation audit committee")
        elif "upsi" in lower_q or "insider" in lower_q:
            queries.append("unpublished price sensitive information fair disclosure code of practices insider trading")
        elif "human rights" in lower_q or "labour" in lower_q or "labor" in lower_q or "child labour" in lower_q or "forced labour" in lower_q:
            queries.append("human rights policy labour standards fair treatment non-discrimination International Labour Organisation fundamental principles")
        elif "supplier" in lower_q or "vendor" in lower_q or "supply chain" in lower_q or "representative" in lower_q:
            queries.append("supply chain policy supplier representative code of conduct vendor standards ESG business ethics")
        elif "expense" in lower_q or "travel" in lower_q or "reimbursement" in lower_q or "per diem" in lower_q:
            queries.append("expense policy travel meals accommodation reimbursement allowable expenses GitLab")
            
        return queries

    def search(self, question: str, top_k: int = 5, score_threshold: float = 0.35) -> dict:
        """
        Standard Search Endpoint complying with UNIFIED-BUSINESS-ASSISTAN.md Section 3.4.
        Features multi-query expansion, RRF ranking, and strict citation deduplication.
        """
        all_queries = self.expand_queries(question)
        
        aggregated_matches = {}
        for q_idx, q_text in enumerate(all_queries):
            query_res = self.collection.query(
                query_texts=[q_text],
                n_results=top_k * 2,
                include=["documents", "metadatas", "distances"]
            )
            docs = query_res.get("documents", [[]])[0]
            metas = query_res.get("metadatas", [[]])[0]
            dists = query_res.get("distances", [[]])[0]
            
            for rank, (doc_text, meta, dist) in enumerate(zip(docs, metas, dists)):
                cid = meta.get("chunk_id")
                similarity = round(max(0.0, 1.0 - dist), 4)
                rrf_score = 1.0 / (60 + rank)
                
                if cid not in aggregated_matches:
                    aggregated_matches[cid] = {
                        "text": doc_text,
                        "source_file": meta.get("source", ""),
                        "page": int(meta.get("page", 1)),
                        "score": similarity,
                        "chunk_id": cid,
                        "title": meta.get("title", ""),
                        "company": meta.get("company", ""),
                        "rrf": rrf_score
                    }
                else:
                    aggregated_matches[cid]["rrf"] += rrf_score
                    if similarity > aggregated_matches[cid]["score"]:
                        aggregated_matches[cid]["score"] = similarity

        # Sort by similarity and RRF fusion score
        sorted_candidates = sorted(
            aggregated_matches.values(),
            key=lambda x: (x["score"] >= score_threshold, x["score"], x["rrf"]),
            reverse=True
        )
        
        # Deduplicate results: ensure no duplicate text or same-page repetition
        seen_snippets = set()
        filtered_results = []
        for c in sorted_candidates:
            if c["score"] < score_threshold:
                continue
            # Hash first 100 normalized characters to prevent identical chunk citations
            norm_snip = re.sub(r'\s+', ' ', c["text"][:100].strip().lower())
            if norm_snip in seen_snippets:
                continue
            seen_snippets.add(norm_snip)
            filtered_results.append(c)
            if len(filtered_results) >= top_k:
                break
                
        no_match = len(filtered_results) == 0
        
        return {
            "results": filtered_results,
            "no_match": no_match,
            "embedding_model": self.embedding_model_name
        }

    def _call_llm_free(self, prompt: str, system_prompt: str) -> tuple:
        """Call free cloud LLM (Gemini or Groq) if API key exists."""
        # 1. Google Gemini Free Tier
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key and gemini_key.strip():
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key.strip()}"
                payload = {
                    "contents": [{"parts": [{"text": f"{system_prompt}\n\n{prompt}"}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800}
                }
                resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    parts = data.get("candidates", [])[0].get("content", {}).get("parts", [])
                    return parts[0].get("text", "").strip(), "Google Gemini Free (gemini-2.0-flash)"
            except Exception:
                pass

        # 2. Groq Free Tier (Using active available model qwen/qwen3.8-27b)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and groq_key.strip():
            try:
                from openai import OpenAI
                client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key.strip())
                resp = client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=800
                )
                text = resp.choices[0].message.content.strip()
                if text:
                    return text, "Groq (qwen/qwen3.8-27b)"
            except Exception:
                pass

        return None, None

    def chat(self, user_message: str, session_id: str = "default_session", top_k: int = 4, score_threshold: float = 0.50) -> dict:
        """
        Conversational Chatbot Endpoint:
        1. Recognizes conversational greetings and thank-yous.
        2. Contextualizes follow-up queries using multi-turn chat history.
        3. Retrieves relevant policy chunks with zero duplicates.
        4. Synthesizes a direct, structured answer with verifiable citations.
        5. Logs session telemetry to Neon PostgreSQL.
        """
        clean_msg = user_message.strip()
        lower_q = clean_msg.lower()
        clean_q = re.sub(r'^[^\w]+|[^\w]+$', '', lower_q)

        # 1. Graceful Greeting, Self-Introduction & Small Talk Handling
        greeting_words = {"hi", "hii", "hello", "hey", "heyy", "greetings", "good morning", "good afternoon", "good evening", "howdy", "sup", "yo"}
        is_greeting = (
            clean_q in greeting_words 
            or any(clean_q.startswith(g + " ") for g in ["hi", "hii", "hello", "hey", "good morning", "good afternoon"])
            or any(p in lower_q for p in [
                "tell me about yourself", "tell me about your self", "who are you", 
                "what are you", "what can you do", "introduce yourself", "about yourself"
            ])
        )

        if is_greeting:
            greeting_reply = (
                "Hello! 👋 I am your **Unified Business Policy Assistant**.\n\n"
                "I am an enterprise AI designed to answer questions from **50 corporate policy documents** across **Susan Electricals**, **Jayaswal Neco**, **Reckitt**, **Google**, and **GitLab**, synchronized with **Neon PostgreSQL**.\n\n"
                "You can ask me questions about:\n"
                "• **HR & Leaves**: Parental and maternity benefits, POSH redressal, workplace ethics.\n"
                "• **Governance & Ethics**: Anti-bribery standards, gift limits, whistleblower reporting.\n"
                "• **Operations & Supply Chain**: Supplier representative rules, ESG vendor guidelines.\n"
                "• **IT & Modern Workplace**: GitLab expenses, Google AI principles, data privacy.\n\n"
                "What policy would you like to inquire about today?"
            )
            self.neon_client.log_chat_interaction(session_id, user_message, greeting_reply, sources="None", score=1.0)
            return {
                "session_id": session_id,
                "question": user_message,
                "standalone_question": user_message,
                "answer": greeting_reply,
                "citations": [],
                "model_used": "Assistant Profile (Self-Introduction)",
                "no_match": False
            }

        # 2. Appreciation & Closing Handling
        thanks_words = {"thanks", "thank you", "thank you so much", "thx", "great thanks", "awesome", "perfect", "got it", "thank you!"}
        if clean_q in thanks_words:
            thanks_reply = "You are very welcome! Feel free to ask if you need further clarification on any company policy or guideline."
            self.neon_client.log_chat_interaction(session_id, user_message, thanks_reply, sources="None", score=1.0)
            return {
                "session_id": session_id,
                "question": user_message,
                "standalone_question": user_message,
                "answer": thanks_reply,
                "citations": [],
                "model_used": "Assistant Profile (Conversational)",
                "no_match": False
            }

        # 3. Conversational Query Reformulation
        history = self.neon_client.get_chat_history(session_id, limit=4)
        standalone_question = self.contextualize_question(user_message, history)

        # 4. Search relevant documents (default score_threshold=0.50 ensures high-quality passages only)
        search_resp = self.search(standalone_question, top_k=top_k, score_threshold=score_threshold)
        results = search_resp["results"]

        if search_resp["no_match"] or not results:
            bot_answer = (
                "I do not have sufficient information in the 50 corporate policy documents "
                "to answer this question. Please refer to your HR or Legal department."
            )
            self.neon_client.log_chat_interaction(session_id, user_message, bot_answer, sources="None", score=0.0)
            return {
                "session_id": session_id,
                "question": user_message,
                "standalone_question": standalone_question,
                "answer": bot_answer,
                "citations": [],
                "model_used": "Rule-based fallback (No match)",
                "no_match": True
            }

        # 5. Format context for answer generation
        context_blocks = []
        for idx, item in enumerate(results, 1):
            context_blocks.append(
                f"[Passage {idx}] Source: {item['source_file']} (Page {item['page']}) | Company: {item['company']}\n"
                f"{item['text'].strip()}"
            )
        rag_context = "\n\n".join(context_blocks)

        system_prompt = (
            "You are an enterprise AI Corporate Policy Assistant for Susan Electricals, Jayaswal Neco, Reckitt, Google, and GitLab. "
            "Answer the question directly, concisely, and accurately based ONLY on the provided policy passages. "
            "Always state the company name, policy document name, and page number for every specific rule or benefit mentioned."
        )
        user_prompt = f"Question: {standalone_question}\n\nRelevant Policy Passages:\n{rag_context}\n\nAnswer:"

        # Attempt free cloud LLMs (Gemini / Groq)
        llm_answer, model_name = self._call_llm_free(user_prompt, system_prompt)

        # Intelligent offline extractive synthesizer
        if not llm_answer:
            model_name = "Local Free Extractive Engine (Offline)"
            
            # Scan top passages for direct definition sentences (containing "means", "is defined as", "shall include")
            best_definition = None
            is_def_q = any(k in standalone_question.lower() for k in ["mean by", "what is", "defined as", "definition", "who is", "who are"])
            
            if is_def_q:
                for item in results:
                    text = item['text']
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    for sent in sentences:
                        s_lower = sent.lower()
                        if ("means" in s_lower or "defined as" in s_lower or "shall include" in s_lower) and len(sent) > 30:
                            best_definition = (sent.strip().replace('\n', ' '), item)
                            break
                    if best_definition:
                        break

            answer_parts = []
            if best_definition:
                def_sentence, def_item = best_definition
                answer_parts.append(
                    f"**Direct Policy Definition:**\n"
                    f"According to **{def_item['company']}** (*{def_item['source_file']}*, Page {def_item['page']}):\n"
                    f"> \"{def_sentence}\"\n\n"
                    f"**Applicable Policy Provisions & Context:**"
                )
            else:
                answer_parts.append("Based on the official corporate policy documents, here are the applicable provisions:\n")

            for idx, item in enumerate(results, 1):
                raw_text = item['text'].replace('\n', ' ').strip()
                # Extract first 2 complete sentences or full chunk if short
                sentences = re.split(r'(?<=[.!?])\s+', raw_text)
                if len(sentences) >= 2:
                    clean_text = " ".join(sentences[:2]).strip()
                else:
                    clean_text = raw_text
                answer_parts.append(
                    f"• **{item['company']}** ({item['source_file']}, Page {item['page']}):\n"
                    f"  \"{clean_text}\"\n"
                )
            llm_answer = "\n".join(answer_parts)

        # Build clean, deduplicated citations
        citations = [
            {
                "source": item["source_file"],
                "page": item["page"],
                "chunk_id": item["chunk_id"],
                "company": item["company"],
                "title": item["title"],
                "score": item["score"]
            }
            for item in results
        ]

        # Log interaction to Neon PostgreSQL
        sources_summary = ", ".join([f"{c['source']}:p{c['page']}" for c in citations])
        avg_score = round(sum(c['score'] for c in citations) / len(citations), 4) if citations else 0.0
        self.neon_client.log_chat_interaction(session_id, user_message, llm_answer, sources=sources_summary, score=avg_score)

        return {
            "session_id": session_id,
            "question": user_message,
            "standalone_question": standalone_question,
            "answer": llm_answer,
            "citations": citations,
            "model_used": model_name,
            "no_match": False
        }


    def get_stats(self):
        """Return combined Neon database and ChromaDB vector store telemetry."""
        neon_stats = self.neon_client.get_stats()
        chroma_count = self.collection.count()
        return {
            "neon_table_chunks": neon_stats.get("total_chunks", 0),
            "chroma_indexed_chunks": chroma_count,
            "documents_in_catalog": self.catalog.get("total_documents", 0),
            "embedding_model": self.embedding_model_name
        }
