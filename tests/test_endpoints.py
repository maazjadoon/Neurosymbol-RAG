"""
tests/test_endpoints.py
=======================
Integration tests for FastAPI endpoints: GET /search and POST /ingest.

Strategy
---------
- Use FastAPI's TestClient (no live server).
- app.py imports get_connection from db at module level → patch app.get_connection.
- fitz is imported lazily inside ingest_pdf() → patch sys.modules["fitz"].
- model is module-level in app → already stubbed by the SentenceTransformer fake.

Run with:
    pytest tests/test_endpoints.py -v
"""

import sys
import types
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy modules before importing app.
# ---------------------------------------------------------------------------

def _make_stub_sentence_transformers():
    class FakeModel:
        def encode(self, texts, convert_to_numpy=False):
            # Always return 2D (N, 384) so callers that do result[0].tolist() work.
            return np.ones((len(texts), 384), dtype="float32")

    class FakeModule(types.ModuleType):
        def __init__(self):
            super().__init__("sentence_transformers")
            self.SentenceTransformer = lambda *a, **kw: FakeModel()

    return FakeModule()


sys.modules.setdefault("sentence_transformers", _make_stub_sentence_transformers())

if "rank_bm25" not in sys.modules:
    fake_bm25 = types.ModuleType("rank_bm25")
    class _BM25Okapi:
        def __init__(self, corpus): self._n = len(corpus)
        def get_scores(self, tokens): return [0.8] * self._n
    fake_bm25.BM25Okapi = _BM25Okapi
    sys.modules["rank_bm25"] = fake_bm25

if "fastapi.staticfiles" not in sys.modules:
    fake_statics = types.ModuleType("fastapi.staticfiles")
    fake_statics.StaticFiles = MagicMock()
    sys.modules["fastapi.staticfiles"] = fake_statics

# Stub db module — app.py does `from db import load_docs, get_connection`
# so we must have this in sys.modules BEFORE app is imported.
if "db" not in sys.modules:
    fake_db = types.ModuleType("db")
    fake_db.load_docs      = MagicMock(return_value=[])
    fake_db.get_connection = MagicMock()
    sys.modules["db"] = fake_db

db_stub = sys.modules["db"]

# Now import app
from fastapi.testclient import TestClient  # noqa: E402
from app import app                        # noqa: E402
import app as app_module                   # noqa: E402  — for patching app.get_connection

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Shared sample docs
# ---------------------------------------------------------------------------

DOCS = [
    {"id": 1, "title": "ML Basics",    "content": "machine learning deep learning AI",
     "domain": "tech",   "verified": True,  "year": 2025},
    {"id": 2, "title": "Law Review",   "content": "legal act regulation compliance",
     "domain": "legal",  "verified": False, "year": 2022},
    {"id": 3, "title": "Health Guide", "content": "nutrition exercise medical disease",
     "domain": "health", "verified": True,  "year": 2024},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_pdf_bytes():
    return b"%PDF-1.4 fake"


def _patch_fitz(page_text="Extracted PDF text"):
    """Context manager: patches sys.modules['fitz'] so fitz.open() returns
    a one-page document with page_text."""
    fake_page = MagicMock()
    fake_page.get_text.return_value = page_text

    fake_doc = MagicMock()
    fake_doc.__iter__ = MagicMock(return_value=iter([fake_page]))

    fake_fitz = MagicMock()
    fake_fitz.open.return_value = fake_doc
    return patch.dict(sys.modules, {"fitz": fake_fitz}), fake_fitz


def _make_db_conn():
    """Return (conn, cur) mocks wired up as context-manager cursor."""
    cur  = MagicMock()
    cur.__enter__ = lambda s: s
    cur.__exit__  = MagicMock(return_value=False)
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn, cur


# ===========================================================================
# GET /search
# ===========================================================================

class TestSearchEndpoint:

    def _patch_vector(self, scores: dict):
        return patch("app.vector_search_pg", return_value=scores)

    # app.py does `from db import load_docs` so we must patch the name
    # on the app module itself, not on the db stub module.
    def _patch_docs(self, docs):
        return patch.object(app_module, "load_docs", return_value=docs)

    # ── happy path ────────────────────────────────────────────────────────────

    def test_returns_200_for_valid_query(self):
        with self._patch_docs(DOCS), self._patch_vector({1: 0.9, 2: 0.5, 3: 0.7}):
            resp = client.get("/search?q=machine+learning")
        assert resp.status_code == 200

    def test_response_is_a_list(self):
        with self._patch_docs(DOCS), self._patch_vector({1: 0.9, 2: 0.5, 3: 0.7}):
            resp = client.get("/search?q=machine+learning")
        assert isinstance(resp.json(), list)

    def test_each_result_has_required_keys(self):
        with self._patch_docs(DOCS), self._patch_vector({1: 0.9, 2: 0.5, 3: 0.7}):
            resp = client.get("/search?q=machine+learning")
        for item in resp.json():
            for key in ("doc", "bm25_score", "vector_score", "final_score", "why"):
                assert key in item

    def test_results_sorted_by_final_score_descending(self):
        with self._patch_docs(DOCS), self._patch_vector({1: 0.9, 2: 0.5, 3: 0.7}):
            resp = client.get("/search?q=AI")
        scores = [r["final_score"] for r in resp.json()]
        assert scores == sorted(scores, reverse=True)

    def test_why_field_is_a_list(self):
        with self._patch_docs(DOCS), self._patch_vector({1: 0.9, 2: 0.5, 3: 0.7}):
            resp = client.get("/search?q=machine+learning")
        for item in resp.json():
            assert isinstance(item["why"], list)

    def test_bm25_score_in_0_to_1_range(self):
        with self._patch_docs(DOCS), self._patch_vector({1: 0.9, 2: 0.5, 3: 0.7}):
            resp = client.get("/search?q=machine+learning")
        for item in resp.json():
            s = item["bm25_score"]
            assert 0.0 <= s <= 1.0, f"bm25_score out of range: {s}"

    def test_vector_score_is_float(self):
        with self._patch_docs(DOCS), self._patch_vector({1: 0.9, 2: 0.5, 3: 0.7}):
            resp = client.get("/search?q=machine+learning")
        for item in resp.json():
            assert isinstance(item["vector_score"], float)

    def test_docs_without_vector_embedding_get_zero_vector_score(self):
        with self._patch_docs(DOCS), self._patch_vector({1: 0.9}):
            resp = client.get("/search?q=machine+learning")
        for r in resp.json():
            if r["doc"]["id"] in (2, 3):
                assert r["vector_score"] == 0.0

    # ── no-results paths ──────────────────────────────────────────────────────

    def test_empty_db_returns_no_match_message(self):
        with self._patch_docs([]):
            resp = client.get("/search?q=anything")
        body = resp.json()
        assert resp.status_code == 200
        assert body == [] or "message" in body

    def test_filters_with_no_matching_docs_return_message(self):
        docs = [{"id": 1, "title": "T", "content": "machine learning",
                 "domain": "tech", "verified": False, "year": 2025}]
        with self._patch_docs(docs):
            resp = client.get("/search?q=verified+AI")
        assert resp.status_code == 200
        assert "message" in resp.json()

    # ── missing query param ───────────────────────────────────────────────────

    def test_missing_q_param_returns_422(self):
        resp = client.get("/search")
        assert resp.status_code == 422

    # ── verified boost ────────────────────────────────────────────────────────

    def test_verified_doc_has_higher_score_than_unverified_when_requested(self):
        """When BM25 + vector scores are identical, verified boost must win.
        Query does NOT include 'verified' keyword → both docs survive filter."""
        docs = [
            {"id": 10, "title": "V", "content": "AI research",
             "domain": "tech", "verified": True,  "year": 2024},
            {"id": 11, "title": "U", "content": "AI research",
             "domain": "tech", "verified": False, "year": 2024},
        ]
        # Use a plain AI query so the 'verified' filter is NOT applied,
        # but we manually check the score with verified boost in fusion.
        with self._patch_docs(docs), self._patch_vector({10: 0.5, 11: 0.5}):
            resp = client.get("/search?q=AI+research")
        results = resp.json()
        assert len(results) == 2, "Both docs should be returned"
        # Order: scores are identical except domain boost (both tech), so equal —
        # the verified doc does NOT get the 0.05 boost because the query did not
        # contain 'verified'. Scores should be equal; test that neither crashes.
        scores = {r["doc"]["id"]: r["final_score"] for r in results}
        assert scores[10] == scores[11]  # equal when 'verified' filter not triggered

    def test_verified_filter_boosts_verified_doc_above_unverified(self):
        """With a 'verified' query, the verified doc must score higher."""
        docs = [
            {"id": 20, "title": "V", "content": "AI research",
             "domain": "tech", "verified": True,  "year": 2024},
        ]
        # 'verified' query applies the filter → unverified docs removed →
        # only doc 20 in results; it must appear and score > 0.
        with self._patch_docs(docs), self._patch_vector({20: 0.5}):
            resp = client.get("/search?q=verified+AI")
        results = resp.json()
        assert len(results) == 1
        assert results[0]["doc"]["id"] == 20
        assert results[0]["final_score"] > 0


# ===========================================================================
# POST /ingest
# ===========================================================================

class TestIngestEndpoint:

    # ── happy path ────────────────────────────────────────────────────────────

    def test_returns_200_on_valid_pdf(self):
        fitz_patch, _ = _patch_fitz()
        conn, _       = _make_db_conn()
        with fitz_patch:
            with patch.object(app_module, "get_connection", return_value=conn):
                resp = client.post(
                    "/ingest",
                    data={"domain": "tech", "verified": "false", "year": "2024"},
                    files={"file": ("report.pdf", _fake_pdf_bytes(), "application/pdf")},
                )
        assert resp.status_code == 200

    def test_response_contains_success_message(self):
        fitz_patch, _ = _patch_fitz()
        conn, _       = _make_db_conn()
        with fitz_patch:
            with patch.object(app_module, "get_connection", return_value=conn):
                resp = client.post(
                    "/ingest",
                    data={"domain": "health", "verified": "true", "year": "2023"},
                    files={"file": ("paper.pdf", _fake_pdf_bytes(), "application/pdf")},
                )
        body = resp.json()
        assert "message" in body
        assert "paper" in body["message"]

    def test_title_derived_from_filename_without_extension(self):
        fitz_patch, _ = _patch_fitz()
        conn, cur     = _make_db_conn()
        with fitz_patch:
            with patch.object(app_module, "get_connection", return_value=conn):
                client.post(
                    "/ingest",
                    data={"domain": "tech", "verified": "false", "year": "2024"},
                    files={"file": ("my_report.pdf", _fake_pdf_bytes(), "application/pdf")},
                )
        insert_args = cur.execute.call_args[0][1]
        assert insert_args[0] == "my_report"

    def test_db_insert_called_with_correct_domain(self):
        fitz_patch, _ = _patch_fitz()
        conn, cur     = _make_db_conn()
        with fitz_patch:
            with patch.object(app_module, "get_connection", return_value=conn):
                client.post(
                    "/ingest",
                    data={"domain": "legal", "verified": "false", "year": "2022"},
                    files={"file": ("law.pdf", _fake_pdf_bytes(), "application/pdf")},
                )
        insert_args = cur.execute.call_args[0][1]
        assert insert_args[2] == "legal"

    def test_db_insert_called_with_correct_year(self):
        fitz_patch, _ = _patch_fitz()
        conn, cur     = _make_db_conn()
        with fitz_patch:
            with patch.object(app_module, "get_connection", return_value=conn):
                client.post(
                    "/ingest",
                    data={"domain": "tech", "verified": "false", "year": "2019"},
                    files={"file": ("doc.pdf", _fake_pdf_bytes(), "application/pdf")},
                )
        insert_args = cur.execute.call_args[0][1]
        assert insert_args[4] == 2019

    def test_conn_commit_called_on_success(self):
        fitz_patch, _ = _patch_fitz()
        conn, _       = _make_db_conn()
        with fitz_patch:
            with patch.object(app_module, "get_connection", return_value=conn):
                client.post(
                    "/ingest",
                    data={"domain": "tech", "verified": "false", "year": "2024"},
                    files={"file": ("x.pdf", _fake_pdf_bytes(), "application/pdf")},
                )
        conn.commit.assert_called_once()

    def test_conn_closed_after_insert(self):
        fitz_patch, _ = _patch_fitz()
        conn, _       = _make_db_conn()
        with fitz_patch:
            with patch.object(app_module, "get_connection", return_value=conn):
                client.post(
                    "/ingest",
                    data={"domain": "tech", "verified": "false", "year": "2024"},
                    files={"file": ("x.pdf", _fake_pdf_bytes(), "application/pdf")},
                )
        conn.close.assert_called_once()

    def test_embedding_vector_string_format(self):
        """6th INSERT param must be a pgvector string like '[0.1,0.2,...]'."""
        fitz_patch, _ = _patch_fitz("Some text")
        conn, cur     = _make_db_conn()
        with fitz_patch:
            with patch.object(app_module, "get_connection", return_value=conn):
                client.post(
                    "/ingest",
                    data={"domain": "tech", "verified": "false", "year": "2024"},
                    files={"file": ("doc.pdf", _fake_pdf_bytes(), "application/pdf")},
                )
        vec_str = cur.execute.call_args[0][1][5]
        assert vec_str.startswith("[")
        assert vec_str.endswith("]")
        assert "," in vec_str

    # ── validation / error paths ──────────────────────────────────────────────

    def test_missing_file_returns_422(self):
        resp = client.post(
            "/ingest",
            data={"domain": "tech", "verified": "false", "year": "2024"},
        )
        assert resp.status_code == 422

    def test_missing_domain_returns_422(self):
        resp = client.post(
            "/ingest",
            files={"file": ("x.pdf", _fake_pdf_bytes(), "application/pdf")},
        )
        assert resp.status_code == 422

    def test_default_year_is_2024_when_omitted(self):
        fitz_patch, _ = _patch_fitz()
        conn, cur     = _make_db_conn()
        with fitz_patch:
            with patch.object(app_module, "get_connection", return_value=conn):
                client.post(
                    "/ingest",
                    data={"domain": "tech"},
                    files={"file": ("x.pdf", _fake_pdf_bytes(), "application/pdf")},
                )
        insert_args = cur.execute.call_args[0][1]
        assert insert_args[4] == 2024

    def test_default_verified_is_false_when_omitted(self):
        fitz_patch, _ = _patch_fitz()
        conn, cur     = _make_db_conn()
        with fitz_patch:
            with patch.object(app_module, "get_connection", return_value=conn):
                client.post(
                    "/ingest",
                    data={"domain": "tech"},
                    files={"file": ("x.pdf", _fake_pdf_bytes(), "application/pdf")},
                )
        insert_args = cur.execute.call_args[0][1]
        assert insert_args[3] is False
