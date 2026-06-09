from fastapi import FastAPI
from rank_bm25 import BM25Okapi
from db import load_docs


# Simple rule engine
def apply_rules(query):
    filters={}

    if "legal" in query:
        filters["domain"]="legal"

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
     

    return results

# search 



def bm25_search(docs,query):
    if not docs:
        return []
    corpus=[doc["content"].split() for doc in docs]
    bm25=BM25Okapi(corpus)

    scores=bm25.get_scores(query.split())
    ranked=sorted(zip(docs,scores),key=lambda x:x[1],reverse=True)
    return [r[0] for r in ranked]

# Explanation Layer 
def explain(doc,filters):
    reasons=[]

    if doc['domain']==filters.get("domain"):
        reasons.append("Matched Domain Rule")

    if doc["verified"]:
        reasons.append("Verified by Authority")
        

    #BM25 score - check if keywords in doc
    return  reasons

# fast api

app=FastAPI()

@app.get("/search")
def search(q:str):
    filters=apply_rules(q)

    docs=load_docs()

    filtered=filter_docs(docs,filters)
    ranked=bm25_search(filtered,q)

    return ranked
    #BM25
    

    
    

