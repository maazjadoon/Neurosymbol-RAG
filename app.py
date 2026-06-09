from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
from db import load_docs
from datetime import datetime,timedelta


# Simple rule engine
def apply_rules(query):
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

def filter_docs(docs,filters):
    results=docs

    if "domain" in filters:
        results=[d for d in results if d["domain"]==filters["domain"]]
    
    if "verified" in filters:
        results=[d for d in results if d["verified"]==True]
    
    datetime.now()-timedelta(days=180)
     

    return results

# search 



def bm25_search(docs,query):
    if not docs:
        return []
    corpus=[doc["content"].split() for doc in docs]
    bm25=BM25Okapi(corpus)

    scores=bm25.get_scores(query.split())
    ranked=sorted(zip(docs,scores),key=lambda x:x[1],reverse=True)
    return ranked

# Explanation Layer 
def explain(doc,filters,score):
    reasons=[]

    if filters.get("domain") and doc["domain"]==filters["domain"]:
        reasons.append("Domain matched")
    
    if filters.get("verified") and doc["verified"]:
        reasons.append("Verified by authority")
    
    if score>5:
        reasons.append("High keyword relevance (BM25)")
    
    #BM25 score - check if keywords in doc
    return  reasons

# query parser for detect intent of query

def parse_query(q):
    query=q.lower()
    result={
        "domain":None,
        "verified":False,
        "keywords":query
    }

    if "tech" in query or "machine learning " in query:
        result["domain"]="tech" 
    
    if "legal" in query:
        result["domain"]="legal"

    if "verified" in query:
        result["verified"]=True

    return result
    
# fast api

app=FastAPI()

@app.get("/search")
def search(q: str):
    filters = apply_rules(q)
    docs = load_docs()
    filtered = filter_docs(docs, filters)
    parsed=parse_query(q)

    ranked = bm25_search(filtered, q)

    results = []

    for doc, score in ranked:

        # 1. start with BM25
        final_score = score

        # 2. verified boost
        if doc["verified"] and parsed["verified"]:
            final_score += 2

        #  3. domain match boost
        if parsed["domain"] and doc["domain"] == parsed["domain"]:
            final_score += 3

        results.append({
            "doc":doc,
            "score":score,
            "final_score":final_score,
            
        })

        #  4. date boost (optional placeholder)
        recent_doc = True  # later you implement real date logic
        if recent_doc:
            final_score += 1

        results.append({
            "doc": doc,
            "bm25_score": float(score),
            "final_score": float(final_score),
            "why": explain(doc, filters, score)
        })

    # 🔥 IMPORTANT: sort by final_score
    results = sorted(results, key=lambda x: x["final_score"], reverse=True)

    return results