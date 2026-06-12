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
# Main search route. Combines BM25 + pgvector into a fused ranked result.
# Usage: GET /search?q=machine+learning
@app.get("/search")
def search(q: str):
    # Step 1: Extract filters from the query text
    filters = apply_rules(q)
    parsed  = {
        "domain":   filters.get("domain"),
        "verified": filters.get("verified", False)
    }

    # Step 2: Load docs from PostgreSQL and apply filters
    docs     = load_docs()       # fetches from DB every request
    filtered = filter_docs(docs, filters)

    if not filtered:
        return {"results": [], "message": "No documents matched your filters."}

    # Step 3: BM25 keyword search on filtered docs
    ranked = bm25_search(filtered, q)

    # Step 4: Semantic vector search using pgvector
    query_embedding = model.encode([q], convert_to_numpy=True)[0]
    filtered_ids    = {d["id"] for d in filtered}  # set of IDs for SQL ANY()
    vector_scores   = vector_search_pg(query_embedding, filtered_ids)

    # Step 5: Fuse scores and apply boosts
    results = []
    current_year = datetime.now().year
    for doc, bm25_score in ranked:
        vscore = vector_scores.get(doc["id"], 0)

        # Weighted fusion: BM25 handles exact keywords, vectors handle meaning
        final_score = (bm25_score * 0.6) + (vscore * 0.4)

        # Boost verified docs when user asked for verified
        if doc["verified"] and parsed["verified"]:
            final_score += 0.05

        # Boost docs that match the detected domain
        if parsed["domain"] and doc["domain"] == parsed["domain"]:
            final_score += 0.08

        # Boost recent documents (within last year)
        if doc.get("year") and doc["year"] >= current_year - 1:
            final_score += 0.03

        results.append({
            "doc":          doc,
            "bm25_score":   float(bm25_score),
            "vector_score": float(vscore),
            "final_score":  float(final_score),
            "why":          explain(doc, filters, bm25_score, vscore)
        })

    # Sort by final fused score, highest first
    return sorted(results, key=lambda x: x["final_score"], reverse=True)


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