import numpy as np
import math
from sqlalchemy.orm import Session
from models_neurosymbolic import KGNode, KGEdge, DocumentKGMapping, Workflow, WorkflowStep

# In-memory session tracking for MAR feature accumulation: session_id -> set(node_ids)
SESSION_FEATURES = {}

def get_session_features(session_id: str, db: Session) -> list[KGNode]:
    """Retrieves the accumulated clinical features for a session."""
    if not session_id or session_id not in SESSION_FEATURES:
        return []
    node_ids = list(SESSION_FEATURES[session_id])
    return db.query(KGNode).filter(KGNode.id.in_(node_ids)).all()

def clear_session(session_id: str):
    """Clears the session cache."""
    if session_id in SESSION_FEATURES:
        del SESSION_FEATURES[session_id]

def accumulate_features(session_id: str, new_nodes: list[KGNode], db: Session) -> list[KGNode]:
    """Accumulates newly extracted clinical concepts into the session feature set (MAR)."""
    if not session_id:
        return new_nodes
        
    if session_id not in SESSION_FEATURES:
        SESSION_FEATURES[session_id] = set()
        
    for node in new_nodes:
        SESSION_FEATURES[session_id].add(node.id)
        
    return get_session_features(session_id, db)

# ── MAR Complexity & Modulation Calculations ────────────────────────────────

def calculate_complexity(nodes: list[KGNode], db: Session) -> float:
    """
    Computes the complexity score for a clinical presentation:
    complexity = |features| + sum(w_ij) + sum(r_i)
    """
    if not nodes:
        return 0.0
        
    node_ids = [n.id for n in nodes]
    
    # 1. Symptom count (|features|)
    symptom_count = len(nodes)
    
    # 2. Risk severity weights (sum of r_i)
    severity_sum = sum(n.severity for n in nodes)
    
    # 3. Graph connectivity weights (sum of w_ij for edges connecting features)
    edges = db.query(KGEdge).filter(
        KGEdge.source_id.in_(node_ids),
        KGEdge.target_id.in_(node_ids)
    ).all()
    
    edges_weight_sum = sum(edge.weight for edge in edges)
    
    return symptom_count + severity_sum + edges_weight_sum

def calculate_modulation_strength(complexity: float, k: float = 0.5) -> float:
    """Computes dynamic modulation strength using a sigmoid mapping."""
    if complexity == 0.0:
        return 0.0
    # Sigmoid function: 1 / (1 + exp(-k * complexity))
    return 1.0 / (1.0 + math.exp(-k * complexity))

def modulate_vector(base_vector: list[float], concept_embeddings: list[list[float]], strength: float) -> list[float]:
    """Modulates a query or document embedding by shifting it in the direction of clinical concepts."""
    if not concept_embeddings or strength == 0.0:
        return base_vector
        
    base_arr = np.array(base_vector)
    # Average the concept embeddings
    concepts_mean = np.mean([np.array(e) for e in concept_embeddings], axis=0)
    
    # Apply modulation shift: base + strength * concepts_mean
    modulated = base_arr + strength * concepts_mean
    
    # Normalize back to unit vector
    norm = np.linalg.norm(modulated)
    if norm > 0:
        modulated = modulated / norm
        
    return modulated.tolist()

# ── KG-Path Traversal & Mapping ──────────────────────────────────────────────

def map_query_to_nodes(query_vector: list[float], db: Session, threshold: float = 0.4, limit: int = 5) -> list[KGNode]:
    """Maps query semantics to initial KG nodes using cosine similarity in the database."""
    # Convert list of floats to PostgreSQL vector format string: [0.1, 0.2, ...]
    vec_str = "[" + ",".join(map(str, query_vector)) + "]"
    
    from db import get_connection
    conn = get_connection()
    node_ids = []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id
                FROM kg_nodes
                WHERE (1 - (embedding <=> %s::vector)) > %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (vec_str, threshold, vec_str, limit))
            node_ids = [row[0] for row in cur.fetchall()]
    except Exception as e:
        print("Error in raw SQL map_query_to_nodes:", e)
        node_ids = []
    finally:
        conn.close()
        
    if not node_ids:
        return []
        
    return db.query(KGNode).filter(KGNode.id.in_(node_ids)).all()

def traverse_kg_bfs(start_nodes: list[KGNode], db: Session, hops: int = 2) -> list[KGNode]:
    """Performs multi-hop BFS traversal from start nodes to retrieve related concepts."""
    if not start_nodes:
        return []
        
    visited_ids = {n.id for n in start_nodes}
    current_level = [n.id for n in start_nodes]
    
    for _ in range(hops):
        if not current_level:
            break
            
        # Get edges originating from the current level
        edges = db.query(KGEdge).filter(KGEdge.source_id.in_(current_level)).all()
        
        next_level = []
        for edge in edges:
            if edge.target_id not in visited_ids:
                visited_ids.add(edge.target_id)
                next_level.append(edge.target_id)
                
        current_level = next_level
        
    return db.query(KGNode).filter(KGNode.id.in_(list(visited_ids))).all()

# ── PageRank Graph Connectivity Ranking ──────────────────────────────────────

def calculate_pagerank_scores(db: Session, query_node_ids: list[int], traversed_node_ids: list[int], iterations: int = 5, damping: float = 0.85) -> dict[int, float]:
    """
    Computes Personalized PageRank (PPR) scores for nodes in the traversed subgraph,
    personalizing the walk on query_node_ids.
    """
    if not traversed_node_ids:
        return {}
        
    # If no starting query nodes, distribute uniformly across all traversed nodes
    personalization_set = query_node_ids if query_node_ids else traversed_node_ids
    
    # Get all edges within the traversed subgraph
    edges = db.query(KGEdge).filter(
        KGEdge.source_id.in_(traversed_node_ids),
        KGEdge.target_id.in_(traversed_node_ids)
    ).all()
    
    # Build adjacency mapping and degrees
    adj = {node_id: [] for node_id in traversed_node_ids}
    out_degree = {node_id: 0.0 for node_id in traversed_node_ids}
    
    for edge in edges:
        adj[edge.source_id].append((edge.target_id, edge.weight))
        out_degree[edge.source_id] += edge.weight
        
    # Initialize Personalized PageRank vector
    ppr = {node_id: 0.0 for node_id in traversed_node_ids}
    for q_id in personalization_set:
        if q_id in ppr:
            ppr[q_id] = 1.0 / len(personalization_set)
            
    # Power iterations
    for _ in range(iterations):
        next_ppr = {node_id: 0.0 for node_id in traversed_node_ids}
        
        for u in traversed_node_ids:
            val = ppr[u]
            if val == 0.0:
                continue
                
            deg = out_degree[u]
            if deg > 0:
                for v, weight in adj[u]:
                    next_ppr[v] += val * (weight / deg)
            else:
                # Sink node: redistribute to personalization set
                for q_id in personalization_set:
                    if q_id in next_ppr:
                        next_ppr[q_id] += val / len(personalization_set)
                        
        # Damping factor addition
        for node_id in traversed_node_ids:
            teleport = (1.0 / len(personalization_set)) if node_id in personalization_set else 0.0
            ppr[node_id] = (1.0 - damping) * teleport + damping * next_ppr[node_id]
            
    return ppr

# ── Proknow-RAG Workflow Reordering ──────────────────────────────────────────

def reorder_by_workflow(results: list[dict], db: Session, workflow_name: str) -> list[dict]:
    """
    Reorders retrieved documents based on steps in the validated workflow.
    Places documents matching earlier steps (e.g. screening) before later steps (e.g. intervention).
    """
    workflow = db.query(Workflow).filter(Workflow.name == workflow_name).first()
    if not workflow:
        return results
        
    steps = db.query(WorkflowStep).filter(WorkflowStep.workflow_id == workflow.id).order_by(WorkflowStep.sequence).all()
    if not steps:
        return results
        
    # Map node_id -> step sequence number
    node_to_sequence = {step.concept_id: step.sequence for step in steps if step.concept_id is not None}
    
    # Get mappings for our documents
    doc_ids = [r["doc"]["id"] for r in results]
    mappings = db.query(DocumentKGMapping).filter(DocumentKGMapping.document_id.in_(doc_ids)).all()
    
    # Map document_id -> set of concept ids it contains
    doc_to_concepts = {}
    for m in mappings:
        if m.document_id not in doc_to_concepts:
            doc_to_concepts[m.document_id] = set()
        doc_to_concepts[m.document_id].add(m.node_id)
        
    # Score items by their earliest matching workflow step (lower sequence is earlier, which comes first)
    # If a document doesn't match any workflow steps, we place it after the steps (max sequence + 1)
    max_sequence = max((step.sequence for step in steps), default=0)
    default_sequence = max_sequence + 1
    
    for r in results:
        doc_id = r["doc"]["id"]
        doc_nodes = doc_to_concepts.get(doc_id, set())
        
        # Find matching steps
        matched_sequences = [node_to_sequence[n_id] for n_id in doc_nodes if n_id in node_to_sequence]
        
        # Target sequence is the earliest step the document satisfies
        r["workflow_step"] = min(matched_sequences) if matched_sequences else default_sequence
        
    # Sort results first by workflow step sequence ascending (earliest first), then by final score descending
    return sorted(results, key=lambda x: (x["workflow_step"], -x["final_score"]))
