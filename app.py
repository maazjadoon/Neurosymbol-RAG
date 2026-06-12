# main.py
# ── Imports ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, Form  # FastAPI + file upload tools
from rank_bm25 import BM25Okapi                       # Keyword search algorithm
from db import load_docs, get_connection               # Our DB helpers
from datetime import datetime                          # For year comparisons
from sentence_transformers import SentenceTransformer  # Text → vector model
from fastapi.responses import FileResponse             # For serving favicon
from fastapi.staticfiles import StaticFiles            # For serving static files
import psycopg2.extras                                 # For RealDictCursor (dict rows)
import tempfile                                        # For temp PDF files on disk
import re                                              # For tokenizing text
import numpy as np                                     # For vector math

# ── Load the AI model once at startup ────────────────────────────────────────
# This model converts text into a list of 384 numbers (a vector/embedding).
# Loading it once here avoids reloading it on every search request.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ── RULE ENGINE ───────────────────────────────────────────────────────────────
# Reads the query and extracts filters like domain, verified, date range.
# Blueprint: filters = apply_rules("show me verified AI papers")
def apply_rules(query: str) -> dict:
    filters = {}
    q = query.lower()  # normalize to lowercase for matching

    # Map each domain to its trigger keywords
    DOMAIN_KEYWORDS = {
        "tech":     ["machine learning", "deep learning", "cloud", "edge ai", "ai", "cybersecurity"],
        "legal":    ["law", "legal", "act", "regulation", "compliance"],
        "business": ["business", "market", "startup", "funding", "marketing"],
        "health":   ["health", "medical", "disease", "nutrition", "exercise"]
    }

    # Check if any keyword from any domain appears in the query
    # any(iterable) → returns True if at least one item is True
    # Blueprint: any(keyword in q for keyword in keywords)
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in q for keyword in keywords):
            filters["domain"] = domain
            break  # stop after first domain match

    if "verified"      in q: filters["verified"]   = True
    if "last 6 months" in q: filters["date_range"] = "6m"

    return filters


# ── FILTER DOCS ───────────────────────────────────────────────────────────────
# Narrows down the document list based on extracted filters.
# Blueprint: filtered = filter_docs(all_docs, {"domain": "tech", "verified": True})
def filter_docs(docs: list, filters: dict) -> list:
    results = docs

    # Keep only docs matching the detected domain
    if "domain" in filters:
        results = [d for d in results if d["domain"] == filters["domain"]]

    # Keep only verified documents
    if "verified" in filters:
        results = [d for d in results if d["verified"] is True]

    # Keep docs from the last ~1 year (using the year integer column)
    if filters.get("date_range") == "6m":
        current_year = datetime.now().year
        results = [d for d in results if d["year"] and d["year"] >= current_year - 1]

    return results


# ── TOKENIZER ─────────────────────────────────────────────────────────────────
# Splits text into a list of lowercase words for BM25.
# re.findall(pattern, string) → returns all non-overlapping matches as a list
# Blueprint: tokenize("Hello World!") → ["hello", "world"]
def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())
    # \w+ means: one or more word characters (letters, digits, underscore)


# ── BM25 KEYWORD SEARCH ───────────────────────────────────────────────────────
# Ranks documents by keyword relevance using the BM25 algorithm.
# Scores are normalized to 0–1 so they're comparable with vector scores.
# Blueprint: ranked = bm25_search(docs, "machine learning trends")
def bm25_search(docs: list, query: str) -> list:
    if not docs:
        return []

    # Build BM25 index from tokenized document contents
    bm25   = BM25Okapi([tokenize(d["content"]) for d in docs])
    scores = bm25.get_scores(tokenize(query))  # score each doc vs query

    # zip(docs, scores) pairs each doc with its score → [(doc1, 0.8), (doc2, 0.3), ...]
    # sorted(..., reverse=True) → highest score first
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    # Normalize: divide all scores by the highest score → all values now 0.0 to 1.0
    # This ensures BM25 scores are on the same scale as pgvector scores (also 0–1)
    # default=1 avoids division by zero if all scores are 0
    max_score = max((score for _, score in ranked), default=1)
    if max_score > 0:
        return [(doc, score / max_score) for doc, score in ranked]
    return ranked


# ── PGVECTOR SEMANTIC SEARCH ──────────────────────────────────────────────────
# Finds semantically similar documents using vector math inside PostgreSQL.
# Returns {doc_id: similarity_score} for all filtered docs.
# Blueprint: scores = vector_search_pg(query_embedding, {1, 2, 3})
def vector_search_pg(query_embedding, filtered_ids: set, top_k: int = 50) -> dict:
    if not filtered_ids:
        return {}

    # Convert numpy array to PostgreSQL vector string format: [0.1, 0.2, ...]
    vec_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"

    conn = get_connection()
    try:
        # RealDictCursor → each row comes back as a dict, not a tuple
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM   documents
                WHERE  id = ANY(%s)
                  AND  embedding IS NOT NULL
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
            """, (vec_str, list(filtered_ids), vec_str, top_k))
            # <=> = L2 distance operator in pgvector; 1 - distance = similarity
            return {row["id"]: float(row["similarity"]) for row in cur.fetchall()}
    finally:
        conn.close()  # always close, even if an error occur


# ── EXPLANATION LAYER ─────────────────────────────────────────────────────────
# Tells the user WHY each result was returned.
# Blueprint: explain(doc, filters, bm25_score=0.9, vector_score=0.7)
def explain(doc: dict, filters: dict, score: float, vector_score: float = 0) -> list:
    reasons = []
    if filters.get("domain")   and doc["domain"]   == filters["domain"]: reasons.append("Domain matched")
    if filters.get("verified") and doc["verified"]:                       reasons.append("Verified by authority")
    if score        > 0.5: reasons.append("High keyword relevance (BM25)")
    if vector_score > 0.3: reasons.append("Semantic similarity match")
    return reasons


# ── FASTAPI APP ───────────────────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/")
def index():
    return FileResponse("static/index.html")


# ── SEARCH ENDPOINT ───────────────────────────────────────────────────────────
# Main search route. Combines BM25 + pgvector + KG PageRank into a fused ranked result.
# Usage: GET /search?q=machine+learning&session_id=s1&workflow=PHQ-9
@app.get("/search")
def search(
    q: str,
    session_id: str = None,
    workflow: str = None,
    alpha: float = 0.6,
    k: float = 0.5,
    beta: float = 0.2
):
    # Step 1: Extract filters from the query text using rules engine
    filters = apply_rules(q)
    parsed  = {
        "domain":   filters.get("domain"),
        "verified": filters.get("verified", False)
    }

    # Step 2: Load docs from PostgreSQL and apply filters
    docs     = load_docs()       # fetches from DB every request
    filtered = filter_docs(docs, filters)

    if not filtered:
        if session_id or workflow:
            return {
                "results": [], 
                "message": "No documents matched your filters.",
                "llm_answer": "No documents were matched, so no context is available to generate an answer.",
                "neurosymbolic": {
                    "session_id": session_id,
                    "accumulated_features": [],
                    "complexity": 0.0,
                    "modulation_strength": 0.0,
                    "traversed_nodes": [],
                    "pagerank_scores": {}
                }
            }
        else:
            return {"results": [], "message": "No documents matched your filters."}

    # Generate base query embedding vector
    query_embedding = model.encode([q], convert_to_numpy=True)[0].tolist()

    # ──── Standard Search Path (Fast, No session/workflow active) ────
    if not session_id and not workflow:
        filtered_ids = {d["id"] for d in filtered}
        # Run original database-level pgvector similarity search
        vector_scores = vector_search_pg(np.array(query_embedding), filtered_ids)
        ranked = bm25_search(filtered, q)
        
        results = []
        current_year = datetime.now().year
        for doc, bm25_score in ranked:
            vscore = vector_scores.get(doc["id"], 0.0)
            final_score = (bm25_score * 0.6) + (vscore * 0.4)

            # Apply boosts
            if doc["verified"] and parsed["verified"]:
                final_score += 0.05
            if parsed["domain"] and doc["domain"] == parsed["domain"]:
                final_score += 0.08
            if doc.get("year") and doc["year"] >= current_year - 1:
                final_score += 0.03

            results.append({
                "doc":          doc,
                "bm25_score":   float(bm25_score),
                "vector_score": float(vscore),
                "final_score":  float(final_score),
                "why":          explain(doc, filters, bm25_score, vscore)
            })

        return sorted(results, key=lambda x: x["final_score"], reverse=True)

    # ──── Neurosymbolic Search Path (MAR + KG-Path + Proknow-RAG) ────
    from db import SessionLocal
    import neurosymbolic_engine as ns_engine
    from models_neurosymbolic import KGNode, DocumentKGMapping

    db = SessionLocal()
    try:
        # Step 3: Map query semantics to initial KG nodes
        mapped_nodes = ns_engine.map_query_to_nodes(query_embedding, db, threshold=0.4, limit=5)
        # Also map nodes whose names are directly in the query text (keyword fallback)
        q_lower = q.lower()
        all_nodes = db.query(KGNode).all()
        for node in all_nodes:
            if node.name.lower() in q_lower and node not in mapped_nodes:
                mapped_nodes.append(node)

        # Step 4: MAR - Feature Accumulation and Query Vector Modulation
        accumulated_nodes = mapped_nodes
        complexity = 0.0
        modulation_strength = 0.0
        if session_id:
            accumulated_nodes = ns_engine.accumulate_features(session_id, mapped_nodes, db)
            complexity = ns_engine.calculate_complexity(accumulated_nodes, db)
            modulation_strength = ns_engine.calculate_modulation_strength(complexity, k=k)
            # Fetch embeddings of accumulated concepts
            concept_embeddings = [n.embedding for n in accumulated_nodes if n.embedding is not None]
            modulated_query_vector = ns_engine.modulate_vector(query_embedding, concept_embeddings, modulation_strength)
        else:
            modulated_query_vector = query_embedding

        # Step 5: KG-Path BFS Traversal
        traversed_nodes = ns_engine.traverse_kg_bfs(mapped_nodes, db, hops=2)
        traversed_node_ids = [n.id for n in traversed_nodes]
        query_node_ids = [n.id for n in mapped_nodes]

        # Step 6: Personalized PageRank scoring
        ppr_scores = ns_engine.calculate_pagerank_scores(db, query_node_ids, traversed_node_ids, iterations=5, damping=0.85)

        # Step 7: Document Modulation & Scoring
        filtered_ids = {d["id"] for d in filtered}
        mappings = db.query(DocumentKGMapping).filter(DocumentKGMapping.document_id.in_filtered_ids if hasattr(DocumentKGMapping.document_id, 'in_filtered_ids') else DocumentKGMapping.document_id.in_(filtered_ids)).all()
        
        doc_to_nodes = {}
        for m in mappings:
            if m.document_id not in doc_to_nodes:
                doc_to_nodes[m.document_id] = []
            doc_to_nodes[m.document_id].append(m.node_id)

        # Build BM25 index on filtered documents
        bm25 = BM25Okapi([tokenize(d["content"]) for d in filtered])
        bm25_scores_raw = bm25.get_scores(tokenize(q))
        
        max_bm25 = max(bm25_scores_raw) if len(bm25_scores_raw) > 0 else 1.0
        bm25_scores = [score / max_bm25 if max_bm25 > 0 else 0.0 for score in bm25_scores_raw]

        results = []
        for i, doc in enumerate(filtered):
            bm25_score = bm25_scores[i]
            mapped_node_ids = doc_to_nodes.get(doc["id"], [])
            
            # Document Modulation
            if mapped_node_ids and beta > 0.0:
                doc_kg_nodes = db.query(KGNode).filter(KGNode.id.in_(mapped_node_ids)).all()
                concept_embeddings_doc = [n.embedding for n in doc_kg_nodes if n.embedding is not None]
                modulated_doc_vector = ns_engine.modulate_vector(doc["embedding"], concept_embeddings_doc, beta)
            else:
                modulated_doc_vector = doc["embedding"]

            # Cosine similarity between modulated vectors
            vscore = 0.0
            if modulated_doc_vector and modulated_query_vector:
                vscore = float(np.dot(modulated_query_vector, modulated_doc_vector))
                vscore = max(0.0, min(1.0, vscore))

            # PageRank score (max probability mapping)
            doc_pr_sum = sum(ppr_scores.get(node_id, 0.0) for node_id in mapped_node_ids)
            doc_pr_score = min(1.0, doc_pr_sum)

            # Score fusion (0.4 * BM25 + 0.3 * Modulated Vector + 0.3 * PageRank)
            final_score = (bm25_score * 0.4) + (vscore * 0.3) + (doc_pr_score * 0.3)

            # Apply boosts
            if doc["verified"] and parsed["verified"]:
                final_score += 0.05
            if parsed["domain"] and doc["domain"] == parsed["domain"]:
                final_score += 0.08
            current_year = datetime.now().year
            if doc.get("year") and doc["year"] >= current_year - 1:
                final_score += 0.03

            results.append({
                "doc":          doc,
                "bm25_score":   float(bm25_score),
                "vector_score": float(vscore),
                "final_score":  float(final_score),
                "why":          explain(doc, filters, bm25_score, vscore)
            })

        # Sort by final score
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)

        # Step 8: Proknow Sequencing
        if workflow:
            results = ns_engine.reorder_by_workflow(results, db, workflow)

        # Build PageRank scores by name for display
        pagerank_names = {}
        for node_id, pr_val in ppr_scores.items():
            node_obj = db.query(KGNode).filter(KGNode.id == node_id).first()
            if node_obj:
                pagerank_names[node_obj.name] = float(pr_val)

        # Step 9: LLM Generation Step
        import os
        import llm_engine
        api_key = os.getenv("GEMINI_API_KEY")
        accumulated_names = [n.name for n in accumulated_nodes]
        
        # Build prompt from retrieved guidelines
        top_docs_content = ""
        for r in results[:3]:
            top_docs_content += f"\nDocument: {r['doc']['title']}\nContent: {r['doc']['content']}\n"
            
        prompt = (
            f"You are a helpful expert system. Answer the User Query based on the retrieved documents and session context.\n\n"
            f"[Session Context]\n"
            f"Active Workflow: {workflow or 'None'}\n"
            f"Accumulated Concepts: {', '.join(accumulated_names) if accumulated_names else 'None'}\n"
            f"Context Complexity: {complexity:.3f}\n\n"
            f"[Retrieved Context Documents]\n"
            f"{top_docs_content}\n"
            f"[User Query]\n"
            f"{q}\n\n"
            f"Generate a clear, helpful, and context-grounded response. If the context is clinical, write a professional assessment and next step recommendation. If the context is DevOps software release, format a release checklist plan. Use markdown syntax."
        )
        
        if api_key:
            llm_answer = llm_engine.generate_llm_response(prompt, api_key)
        else:
            llm_answer = llm_engine.generate_mock_response(q, results[:3], accumulated_names, workflow)

        return {
            "results": results,
            "llm_answer": llm_answer,
            "neurosymbolic": {
                "session_id": session_id,
                "accumulated_features": accumulated_names,
                "complexity": float(complexity),
                "modulation_strength": float(modulation_strength),
                "traversed_nodes": [n.name for n in traversed_nodes],
                "pagerank_scores": pagerank_names
            }
        }
    finally:
        db.close()


@app.post("/seed-neurosymbolic")
def seed_neurosymbolic_route():
    from seed_neurosymbolic import seed_data
    seed_data()
    return {"message": "Database seeded successfully with neurosymbolic guidelines"}


@app.post("/clear-session")
def clear_session_route(session_id: str):
    import neurosymbolic_engine as ns_engine
    ns_engine.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared"}


@app.get("/workflows")
def get_workflows_route():
    from db import SessionLocal
    from models_neurosymbolic import Workflow, WorkflowStep, KGNode
    db = SessionLocal()
    try:
        workflows = db.query(Workflow).all()
        res = {}
        for wf in workflows:
            steps = db.query(WorkflowStep).filter(WorkflowStep.workflow_id == wf.id).order_by(WorkflowStep.sequence).all()
            res[wf.name] = []
            for step in steps:
                node = db.query(KGNode).filter(KGNode.id == step.concept_id).first()
                res[wf.name].append({
                    "seq": step.sequence,
                    "title": step.title,
                    "concept": node.name if node else ""
                })
        return res
    finally:
        db.close()


# ── PDF INGEST ENDPOINT ───────────────────────────────────────────────────────
# Accepts a PDF upload, extracts text, generates embedding, saves to PostgreSQL.
# Usage: POST /ingest  (multipart form with file, domain, verified, year)
@app.post("/ingest")
async def ingest_pdf(
    file:     UploadFile = File(...),   # the PDF file
    domain:   str        = Form(...),   # e.g. "tech", "health"
    verified: bool       = Form(False), # is this an authoritative source?
    year:     int        = Form(2024)   # publication year
):
    import fitz  # pymupdf — imported here to keep startup fast

    # Save uploaded bytes to a temp file so fitz can open it
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Extract all text from every page of the PDF
    pdf_doc = fitz.open(tmp_path)
    content = "\n".join(page.get_text() for page in pdf_doc)
    title   = file.filename.replace(".pdf", "")

    # Generate 384-dimensional embedding vector for the full content
    embedding = model.encode([content])[0].tolist()
    vec_str   = "[" + ",".join(map(str, embedding)) + "]"

    # Insert into PostgreSQL with the embedding
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO documents (title, content, domain, verified, year, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
            """, (title, content, domain, verified, year, vec_str))
        conn.commit()
    finally:
        conn.close()

    return {"message": f"'{title}' ingested successfully"}