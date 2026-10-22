"""
Comprehensive End-to-End Test and Self-Evaluation Suite
for Unified Business Assistant - Role A (Knowledge Service).
Evaluates:
1. Health & Telemetry Probes (/health, /stats, /documents)
2. Semantic Search Retrieval Accuracy & Provenance (/search)
3. Conversational Multi-Turn Contextualization (/chat)
4. Out-of-Scope / Hallucination Guardrail Check (/chat)
5. Neon PostgreSQL Database Telemetry & Persistence
"""

import sys
import json
import time
import urllib.request
import urllib.error

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:8000"

def http_get(endpoint):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "E2E-Evaluator"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def http_post(endpoint, data):
    url = f"{BASE_URL}{endpoint}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "E2E-Evaluator"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))

def run_evaluation():
    results = {}
    print("=" * 70)
    print(" UNIFIED BUSINESS ASSISTANT (ROLE A) - E2E EVALUATION SUITE")
    print("=" * 70)

    # 1. Health Probe
    print("\n[TEST 1] Testing Health Endpoint (/health)...")
    try:
        health = http_get("/health")
        assert health.get("status") == "healthy", f"Unexpected status: {health}"
        assert health.get("neon_connected") is True, "Neon not reported connected"
        print("  PASS: Health check OK. Status: healthy, Neon: connected.")
        results["health_check"] = {"passed": True, "data": health}
    except Exception as e:
        print(f"  FAIL: {e}")
        results["health_check"] = {"passed": False, "error": str(e)}

    # 2. Stats & Catalog
    print("\n[TEST 2] Testing Telemetry Stats (/stats) & Catalog (/documents)...")
    try:
        stats = http_get("/stats")
        catalog = http_get("/documents")
        chunk_count = stats.get("chroma_indexed_chunks")
        doc_count = stats.get("documents_in_catalog")
        neon_chunks = int(stats.get("neon_table_chunks", 0))
        assert chunk_count == 1509, f"Expected 1509 chunks, found {chunk_count}"
        assert doc_count == 50, f"Expected 50 documents, found {doc_count}"
        assert neon_chunks == 1509, f"Neon chunks count mismatch: {neon_chunks}"
        assert len(catalog.get("documents", [])) == 50, "Catalog documents array length != 50"
        print(f"  PASS: Telemetry consistent. Chroma chunks: {chunk_count}, Neon chunks: {neon_chunks}, Documents: {doc_count}.")
        results["telemetry_consistency"] = {"passed": True, "stats": stats}
    except Exception as e:
        print(f"  FAIL: {e}")
        results["telemetry_consistency"] = {"passed": False, "error": str(e)}

    # 3. Canonical Search Contract
    print("\n[TEST 3] Testing Semantic Search (/search) Retrieval & Provenance...")
    test_queries = [
        {"q": "maternity leave duration and policy", "expected_kw": "parental"},
        {"q": "whistleblower complaint reporting procedure", "expected_kw": "whistleblower"},
        {"q": "prevention of sexual harassment redressal committee", "expected_kw": "posh"},
    ]
    search_passed = 0
    for tq in test_queries:
        res = http_post("/search", {"question": tq["q"], "top_k": 3, "score_threshold": 0.35})
        has_results = len(res.get("results", [])) > 0
        provenance_valid = all(
            "source_file" in r and "page" in r and "score" in r and "chunk_id" in r
            for r in res.get("results", [])
        )
        scores_sorted = True
        scores = [r["score"] for r in res.get("results", [])]
        if len(scores) > 1:
            scores_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        if has_results and provenance_valid and scores_sorted:
            search_passed += 1
            print(f"  PASS: Query: '{tq['q']}' -> {len(res['results'])} chunks retrieved with strict provenance & score ranking.")
        else:
            print(f"  FAIL: Query: '{tq['q']}' -> res: {res}")
            
    results["semantic_search"] = {"passed": search_passed == len(test_queries), "passed_count": search_passed}

    # 4. Multi-turn Conversational Contextualization & History
    print("\n[TEST 4] Testing Multi-turn Chat & Conversational Reformulation (/chat)...")
    test_session = f"eval_session_{int(time.time())}"
    chat_turns = [
        {"msg": "What is the policy for supplier code of conduct at Jayaswal Neco?", "check_contains": "jayaswal"},
        {"msg": "What are their core responsibilities under ESG?", "check_followup": True},
    ]
    chat_passed = 0
    t1 = http_post("/chat", {"message": chat_turns[0]["msg"], "session_id": test_session})
    if len(t1.get("citations", [])) > 0 and len(t1.get("answer", "")) > 50:
        print(f"  Turn 1 PASS: Answer generated with {len(t1['citations'])} citations.")
        chat_passed += 1
    else:
        print(f"  Turn 1 FAIL: {t1}")

    t2 = http_post("/chat", {"message": chat_turns[1]["msg"], "session_id": test_session})
    # Standalone question reformulation check
    reformulated = t2.get("standalone_question", "")
    has_context = "jayaswal" in reformulated.lower() or "supplier" in reformulated.lower()
    if has_context and len(t2.get("citations", [])) > 0:
        print(f"  Turn 2 PASS: Reformulation resolved pronouns to context -> '{reformulated}'")
        chat_passed += 1
    else:
        print(f"  Turn 2 FAIL: Standalone query was '{reformulated}'")
    results["conversational_memory"] = {"passed": chat_passed == 2, "turn2_standalone": reformulated}

    # 5. Out-of-Scope / Safety Guardrail
    print("\n[TEST 5] Testing Out-of-Scope / Hallucination Guardrail...")
    oos_res = http_post("/chat", {"message": "What is the capital of France?", "session_id": test_session})
    if oos_res.get("no_match") is True and len(oos_res.get("citations", [])) == 0:
        print("  PASS: Out-of-scope query successfully rejected with no_match=True and zero citations.")
        results["guardrail_rejection"] = {"passed": True}
    else:
        print(f"  FAIL: Out-of-scope response: {oos_res}")
        results["guardrail_rejection"] = {"passed": False}

    # 6. Neon Telemetry Verification
    print("\n[TEST 6] Testing Neon PostgreSQL Telemetry Audit Logging...")
    from neon_client import NeonDBClient
    db = NeonDBClient()
    history = db.get_chat_history(test_session)
    if len(history) >= 2:
        print(f"  PASS: Verified {len(history)} chat interaction rows saved in Neon PostgreSQL 'rag_chat_logs'.")
        results["neon_audit_log"] = {"passed": True, "rows": len(history)}
    else:
        print(f"  FAIL: Neon history rows found: {len(history)}")
        results["neon_audit_log"] = {"passed": False}

    # Summary
    all_passed = all(r.get("passed", False) for r in results.values())
    print("\n" + "=" * 70)
    print(f" EVALUATION SUMMARY: {'ALL TESTS PASSED [100%]' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)
    return results

if __name__ == "__main__":
    run_evaluation()
