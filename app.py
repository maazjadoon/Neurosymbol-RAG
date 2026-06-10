from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
from db import load_docs
from datetime import datetime,timedelta
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import regex as re


# Load Documents
docs=load_docs()

if not docs:
    raise ValueError("No documents found!")
# creating embeddings using sentence_transformers


model=SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


#Embedding for vector 
embeddings=model.encode(
    [doc["content"] for doc in docs]
)

index=faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))




# Simple rule engine
def apply_rules(query): # for detecting the domain
    filters={}
    query=query.lower()

    DOMAIN_KEYWORDS={
        "tech":["machine learning","deep learning","cloud","edge ai","ai","cybersecurity"],
        "legal":["law","legal","act","regulation","compliance"],
        "business":["business","market","startup","funding","marketing"],
        "health":["health","medical","disease","nutrition","exercise"]
    }
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            filters["domain"] = domain
            break

    if "verified" in query:
        filters["verified"]=True

    if "last 6 months" in query:
        filters["date_range"]="6m"


    return filters

# filter data 

def filter_docs(docs, filters):
    results = docs

    if "domain" in filters:
        results = [d for d in results if d["domain"] == filters["domain"]]

    if "verified" in filters:
        results = [d for d in results if d["verified"] == True]

    if "date_range" in filters and filters["date_range"] == "6m":
        cutoff = datetime.now() - timedelta(days=180)
        results = [
            d for d in results
            if datetime.fromisoformat(d["date"]) >= cutoff
        ]

    return results

# search 

def tokenize(text):
    return re.findall(r"\w+",text.lower())

def bm25_search(docs,query):
    if not docs:
        return []
    corpus=[tokenize(doc["content"]) for doc in docs]
    bm25=BM25Okapi(corpus)

    scores=bm25.get_scores(tokenize(query))
    ranked=sorted(zip(docs,scores),key=lambda x:x[1],reverse=True)
    return ranked

# Explanation Layer 
def explain(doc,filters,score):
    reasons=[]

    if filters.get("domain") and doc["domain"]==filters["domain"]:
        reasons.append("Domain matched")
    
    if filters.get("verified") and doc["verified"]:
        reasons.append("Verified by authority")
    
    if score>0.5:
        reasons.append("High keyword relevance (BM25)")
    
    #BM25 score - check if keywords in doc
    return  reasons




# fast api

app=FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/favicon.ico")
def favicon():
    return FileResponse("favicon.ico")

@app.get("/search")
def search(q: str):
    filters = apply_rules(q)
    filtered = filter_docs(docs, filters)

    
    # filters already has domain and verified — no need to parse twice
    parsed = {
        "domain": filters.get("domain", None),
        "verified": filters.get("verified", False),
        "keywords": q.lower()
    }

    ranked = bm25_search(filtered, q)

# Normalize BM25 scores to 0-1 range before fusion
    if ranked:
        max_bm25 = max(score for _, score in ranked)
        if max_bm25 > 0:
            ranked = [(doc, score / max_bm25) for doc, score in ranked]

    results = []

    # creating vector retrieval

    query_embedding = model.encode([q], convert_to_numpy=True)
    vector_scores={}
    if filtered:
        filtered_embeddings = model.encode([doc["content"] for doc in filtered])
        filtered_embeddings = np.array(filtered_embeddings).astype("float32")
        temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)
    
        # Get top 50
        k = min(50, len(filtered))
        distances, indices = temp_index.search(query_embedding, k)

        # Store vector scores for filtered docs
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            doc_id = filtered[idx]["id"]
            vector_scores[doc_id] = 1 / (1 + dist)
    

    
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1: continue
        doc_id = id_map.get(idx)
        if doc_id in filtered_ids:
            vector_scores[doc_id] = 1 / (1 + dist)


    for doc, score in ranked:
        # 1. start with BM25
        bm25_score=score
        vector_score = vector_scores.get(doc["id"], 0)
        final_score = (
            bm25_score*0.6 +
            vector_score*0.4
        )
        # 2. verified boost
        if doc["verified"] and parsed["verified"]:
            final_score += 0.2

        #  3. domain match boost
        if parsed["domain"] and doc["domain"] == parsed["domain"]:
            final_score += 0.3


        #  4. date boost (optional placeholder)
        try:
            doc_date = datetime.fromisoformat(doc["date"])
            if doc_date >= datetime.now() - timedelta(days=180):
                final_score += 0.1
        except (KeyError, ValueError):
            pass


        results.append({
            "doc": doc,
            "bm25_score": float(score),
            "vector_score": float(vector_score),
            "final_score": float(final_score),
            "why": explain(doc, filters, score)
        })

    # 🔥 IMPORTANT: sort by final_score
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    return results