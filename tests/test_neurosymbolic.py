"""
tests/test_neurosymbolic.py
===========================
Unit and integration tests for the Neurosymbolic RAG engine.
This verifies the mathematical operations (MAR complexity, vector shifts, sigmoid),
Personalized PageRank connectivity rankings, KG BFS hops, and Proknow-RAG workflow re-ranking.
Uses an in-memory SQLite database for testing database-dependent methods.
"""
import pytest
import numpy as np
import math
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.compiler import compiles

# ── Intercept/Stub db module dynamically before any model imports ───────────
import sys
import types
from sqlalchemy.orm import declarative_base

if "db" not in sys.modules:
    fake_db = types.ModuleType("db")
    fake_db.load_docs = lambda: []
    fake_db.get_connection = lambda: None
    sys.modules["db"] = fake_db
else:
    fake_db = sys.modules["db"]

if not hasattr(fake_db, "Base"):
    fake_db.Base = declarative_base()
if not hasattr(fake_db, "SessionLocal"):
    fake_db.SessionLocal = sessionmaker()

# Now it is safe to import Base and models
from db import Base
from models_neurosymbolic import KGNode, KGEdge, DocumentKGMapping, Workflow, WorkflowStep
from models import Document  # We need the real Document model class too

# Stub vector type compilation for SQLite dialect
from pgvector.sqlalchemy import Vector
@compiles(Vector, 'sqlite')
def compile_vector_sqlite(element, compiler, **kw):
    return "TEXT"

import neurosymbolic_engine as ns_engine

# ── Database Session Fixture ───────────────────────────────────────────────
@pytest.fixture
def db_session():
    """Provides an isolated in-memory SQLite database with schema created."""
    from sqlalchemy.pool import StaticPool
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    
    # Crucial: Override models that might use other bases or construct tables
    # For SQLite, Base contains our models since they inherit from it.
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

# ── Unit Tests ──────────────────────────────────────────────────────────────

def test_calculate_complexity(db_session):
    """Verifies that calculate_complexity correctly sums node count, severity, and edge weights."""
    # Create test nodes
    n1 = KGNode(name="symptom_a", category="symptom", severity=0.5)
    n2 = KGNode(name="symptom_b", category="symptom", severity=0.3)
    n3 = KGNode(name="risk_a", category="risk_factor", severity=0.8)
    db_session.add_all([n1, n2, n3])
    db_session.commit()
    
    # Add an edge connecting n1 and n2
    edge = KGEdge(source_id=n1.id, target_id=n2.id, relationship="maintains", weight=0.6)
    db_session.add(edge)
    db_session.commit()
    
    # Complexity: symptom count (3) + severity (0.5 + 0.3 + 0.8 = 1.6) + edge weight (0.6) = 5.2
    comp = ns_engine.calculate_complexity([n1, n2, n3], db_session)
    assert abs(comp - 5.2) < 1e-6
    
    # Check empty list returns 0.0
    assert ns_engine.calculate_complexity([], db_session) == 0.0

def test_calculate_modulation_strength():
    """Verifies calculation of sigmoid-based modulation strength."""
    # Zero complexity -> zero strength
    assert ns_engine.calculate_modulation_strength(0.0) == 0.0
    
    # Normal sigmoid check
    comp = 5.2
    k = 0.5
    expected = 1.0 / (1.0 + math.exp(-k * comp))
    assert abs(ns_engine.calculate_modulation_strength(comp, k) - expected) < 1e-6

def test_modulate_vector():
    """Verifies that modulate_vector shifts the base vector towards concepts and normalizes it."""
    base = [1.0, 0.0]
    concepts = [[0.0, 1.0]]
    strength = 0.5
    
    # Expected shift: base + strength * concept_mean = [1.0, 0.0] + 0.5 * [0.0, 1.0] = [1.0, 0.5]
    # Expected normalization: norm = sqrt(1.0 + 0.25) = sqrt(1.25) = 1.1180339887
    # Modulated: [1.0 / 1.118, 0.5 / 1.118] = [0.894427, 0.447213]
    mod = ns_engine.modulate_vector(base, concepts, strength)
    
    expected_x = 1.0 / math.sqrt(1.25)
    expected_y = 0.5 / math.sqrt(1.25)
    
    assert abs(mod[0] - expected_x) < 1e-5
    assert abs(mod[1] - expected_y) < 1e-5
    
    # Zero strength -> original vector
    assert ns_engine.modulate_vector(base, concepts, 0.0) == base
    
    # Empty concept embeddings -> original vector
    assert ns_engine.modulate_vector(base, [], strength) == base

def test_traverse_kg_bfs(db_session):
    """Verifies multi-hop BFS traversal paths from query nodes."""
    # Create a chain: n1 -> n2 -> n3 -> n4
    n1 = KGNode(name="n1", category="symptom")
    n2 = KGNode(name="n2", category="symptom")
    n3 = KGNode(name="n3", category="symptom")
    n4 = KGNode(name="n4", category="symptom")
    db_session.add_all([n1, n2, n3, n4])
    db_session.commit()
    
    e1 = KGEdge(source_id=n1.id, target_id=n2.id, relationship="rel")
    e2 = KGEdge(source_id=n2.id, target_id=n3.id, relationship="rel")
    e3 = KGEdge(source_id=n3.id, target_id=n4.id, relationship="rel")
    db_session.add_all([e1, e2, e3])
    db_session.commit()
    
    # 0 hops: just start node
    res_0 = ns_engine.traverse_kg_bfs([n1], db_session, hops=0)
    assert {n.name for n in res_0} == {"n1"}
    
    # 1 hop: n1 -> n2
    res_1 = ns_engine.traverse_kg_bfs([n1], db_session, hops=1)
    assert {n.name for n in res_1} == {"n1", "n2"}
    
    # 2 hops: n1 -> n2 -> n3
    res_2 = ns_engine.traverse_kg_bfs([n1], db_session, hops=2)
    assert {n.name for n in res_2} == {"n1", "n2", "n3"}
    
    # Empty list input -> empty output
    assert ns_engine.traverse_kg_bfs([], db_session) == []

def test_calculate_pagerank_scores(db_session):
    """Verifies Personalized PageRank scores over a small subgraph."""
    n1 = KGNode(name="n1", category="symptom")
    n2 = KGNode(name="n2", category="symptom")
    n3 = KGNode(name="n3", category="symptom")
    db_session.add_all([n1, n2, n3])
    db_session.commit()
    
    e1 = KGEdge(source_id=n1.id, target_id=n2.id, relationship="rel", weight=1.0)
    e2 = KGEdge(source_id=n2.id, target_id=n3.id, relationship="rel", weight=1.0)
    db_session.add_all([e1, e2])
    db_session.commit()
    
    # PPR personalizing on n1 (query node) within subgraph {n1, n2, n3}
    ppr = ns_engine.calculate_pagerank_scores(
        db_session, 
        query_node_ids=[n1.id], 
        traversed_node_ids=[n1.id, n2.id, n3.id],
        iterations=5,
        damping=0.85
    )
    
    # n1 should have PageRank score, and all node scores should sum up
    assert n1.id in ppr
    assert n2.id in ppr
    assert n3.id in ppr
    
    # In PPR, damping teleports back to n1, so n1 should have a significant score
    assert ppr[n1.id] > 0.0
    
    # PageRank scores should be non-negative
    assert all(val >= 0.0 for val in ppr.values())

def test_reorder_by_workflow(db_session):
    """Verifies Proknow-RAG document re-ordering strictly sequences documents by workflow steps."""
    # Setup workflow & steps
    wf = Workflow(name="PHQ-9", description="Screening workflow")
    db_session.add(wf)
    db_session.flush()
    
    n_screening = KGNode(name="screening_node", category="protocol")
    n_diagnostic = KGNode(name="diag_node", category="protocol")
    n_treatment = KGNode(name="treat_node", category="intervention")
    db_session.add_all([n_screening, n_diagnostic, n_treatment])
    db_session.flush()
    
    step1 = WorkflowStep(workflow_id=wf.id, sequence=1, title="Screening Step", concept_id=n_screening.id)
    step2 = WorkflowStep(workflow_id=wf.id, sequence=2, title="Diagnostic Step", concept_id=n_diagnostic.id)
    step3 = WorkflowStep(workflow_id=wf.id, sequence=3, title="Treatment Step", concept_id=n_treatment.id)
    db_session.add_all([step1, step2, step3])
    db_session.flush()
    
    # Create documents
    doc_screen = Document(title="Screening Guideline", content="Details about initial depression screening.")
    doc_diag = Document(title="Diagnostic Guide", content="Details about formal diagnosis criteria.")
    doc_treat = Document(title="CBT Intervention", content="Details about Cognitive Behavioral Therapy.")
    doc_unmapped = Document(title="General Protocol", content="General clinical practices.")
    db_session.add_all([doc_screen, doc_diag, doc_treat, doc_unmapped])
    db_session.flush()
    
    # Map documents to nodes
    m1 = DocumentKGMapping(document_id=doc_screen.id, node_id=n_screening.id)
    m2 = DocumentKGMapping(document_id=doc_diag.id, node_id=n_diagnostic.id)
    m3 = DocumentKGMapping(document_id=doc_treat.id, node_id=n_treatment.id)
    db_session.add_all([m1, m2, m3])
    db_session.commit()
    
    # Prepare dummy candidate search results (shuffled scores)
    results = [
        {"doc": {"id": doc_treat.id, "title": doc_treat.title}, "final_score": 0.95},
        {"doc": {"id": doc_unmapped.id, "title": doc_unmapped.title}, "final_score": 0.99},
        {"doc": {"id": doc_screen.id, "title": doc_screen.title}, "final_score": 0.75},
        {"doc": {"id": doc_diag.id, "title": doc_diag.title}, "final_score": 0.85}
    ]
    
    # Sort
    reordered = ns_engine.reorder_by_workflow(results, db_session, "PHQ-9")
    
    # Expected ordering:
    # 1. doc_screen (workflow step 1)
    # 2. doc_diag (workflow step 2)
    # 3. doc_treat (workflow step 3)
    # 4. doc_unmapped (unmapped, falls to sequence 4)
    ordered_titles = [r["doc"]["title"] for r in reordered]
    assert ordered_titles == [
        "Screening Guideline",
        "Diagnostic Guide",
        "CBT Intervention",
        "General Protocol"
    ]
    
    # Confirm workflow step values were assigned
    assert reordered[0]["workflow_step"] == 1
    assert reordered[1]["workflow_step"] == 2
    assert reordered[2]["workflow_step"] == 3
    assert reordered[3]["workflow_step"] == 4


def test_generate_mock_response():
    """Verifies that generate_mock_response returns expected strings based on workflow type."""
    import llm_engine
    
    # 1. Test PHQ-9 mock response
    resp_phq = llm_engine.generate_mock_response(
        query="feel depressed",
        documents=[{"doc": {"title": "Clinical Guideline PHQ-9", "content": "xyz"}}],
        features=["low mood", "insomnia"],
        workflow="PHQ-9"
    )
    assert "MOCK LLM GENERATION" in resp_phq
    assert "Clinical Context & Analysis" in resp_phq
    assert "low mood" in resp_phq
    assert "Clinical Guideline PHQ-9" in resp_phq
    
    # 2. Test Software_Release mock response
    resp_release = llm_engine.generate_mock_response(
        query="release code to staging",
        documents=[{"doc": {"title": "Release Guidelines", "content": "xyz"}}],
        features=["ci_build", "vulnerability_scan"],
        workflow="Software_Release"
    )
    assert "MOCK LLM GENERATION" in resp_release
    assert "Software Release Assessment" in resp_release
    assert "ci_build" in resp_release
    assert "vulnerability_scan" in resp_release
    
    # 3. Test generic fallback response
    resp_generic = llm_engine.generate_mock_response(
        query="some general query",
        documents=[{"doc": {"title": "General Info", "content": "xyz"}}],
        features=["general_concept"],
        workflow="Unknown_Workflow"
    )
    assert "MOCK LLM GENERATION" in resp_generic
    assert "RAG Context Summary" in resp_generic
    assert "General Info" in resp_generic


def test_get_workflows_endpoint(db_session):
    """Verifies the GET /workflows endpoint dynamically retrieves workflow configuration."""
    import sys
    import types
    from unittest.mock import patch
    
    # Stub sentence_transformers to avoid loading real model during app import
    if "sentence_transformers" not in sys.modules:
        class FakeModel:
            def encode(self, texts, convert_to_numpy=False):
                import numpy as np
                return np.ones((len(texts), 384), dtype="float32")
        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = lambda *a, **kw: FakeModel()
        sys.modules["sentence_transformers"] = fake_st
        
    from fastapi.testclient import TestClient
    import db as db_module
    
    # Seed a test workflow and step in the test database session
    wf = Workflow(name="Test_Workflow", description="Test Desc")
    db_session.add(wf)
    db_session.flush()
    
    n_concept = KGNode(name="test_concept", category="test_category")
    db_session.add(n_concept)
    db_session.flush()
    
    step = WorkflowStep(workflow_id=wf.id, sequence=1, title="Test Step 1", concept_id=n_concept.id)
    db_session.add(step)
    db_session.commit()
    
    # We patch the db.SessionLocal so the route uses our db_session
    with patch.object(db_module, "SessionLocal", return_value=db_session):
        from app import app
        client = TestClient(app)
        response = client.get("/workflows")
        
        assert response.status_code == 200
        data = response.json()
        assert "Test_Workflow" in data
        assert len(data["Test_Workflow"]) == 1
        assert data["Test_Workflow"][0]["seq"] == 1
        assert data["Test_Workflow"][0]["title"] == "Test Step 1"
        assert data["Test_Workflow"][0]["concept"] == "test_concept"

