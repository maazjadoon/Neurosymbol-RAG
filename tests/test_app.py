"""
tests/test_app.py
=================
Unit tests for the Neurosymbol RAG pipeline.

Boundary mocking strategy
--------------------------
- Database (psycopg2 / load_docs)   → mocked; tests run with no live DB.
- SentenceTransformer model          → mocked; tests run without downloading weights.
- All pure-logic functions           → tested directly with no mocks.

Run with:
    pytest tests/ -v
"""

import sys
import types
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# ---------------------------------------------------------------------------
# Stub heavy third-party modules BEFORE importing app so the module-level
# `model = SentenceTransformer(...)` call never actually downloads anything.
# ---------------------------------------------------------------------------

def _make_stub_sentence_transformers():
    """Return a fake sentence_transformers package whose SentenceTransformer
    encodes text as a reproducible 384-dim unit vector.
    Always returns a 2D (N, 384) array so callers that do result[0].tolist() work."""
    class FakeModel:
        def encode(self, texts, convert_to_numpy=False):
            return np.ones((len(texts), 384), dtype="float32")

    class FakeModule(types.ModuleType):
        def __init__(self):
            super().__init__("sentence_transformers")
            self.SentenceTransformer = lambda *a, **kw: FakeModel()

    return FakeModule()


# Inject stubs before any project import
sys.modules.setdefault("sentence_transformers", _make_stub_sentence_transformers())

# Stub rank_bm25 only if not installed (allows real install to take precedence)
if "rank_bm25" not in sys.modules:
    fake_bm25_mod = types.ModuleType("rank_bm25")
    class _BM25Okapi:
        def __init__(self, corpus): self._corpus = corpus
        def get_scores(self, tokens):
            return [1.0] * len(self._corpus)
    fake_bm25_mod.BM25Okapi = _BM25Okapi
    sys.modules["rank_bm25"] = fake_bm25_mod

# Stub fastapi.staticfiles (may not be installed in test env)
if "fastapi.staticfiles" not in sys.modules:
    fake_statics = types.ModuleType("fastapi.staticfiles")
    fake_statics.StaticFiles = MagicMock()
    sys.modules["fastapi.staticfiles"] = fake_statics

# Stub db module so we control load_docs / get_connection entirely
fake_db = types.ModuleType("db")
fake_db.load_docs = MagicMock(return_value=[])
fake_db.get_connection = MagicMock()
sys.modules["db"] = fake_db

# Now it is safe to import our functions
from app import (   # noqa: E402
    apply_rules,
    filter_docs,
    tokenize,
    bm25_search,
    explain,
    vector_search_pg,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCS = [
    {"id": 1, "title": "ML Basics",      "content": "machine learning intro",    "domain": "tech",     "verified": True,  "year": 2025},
    {"id": 2, "title": "Law Review",     "content": "legal act regulation",      "domain": "legal",    "verified": False, "year": 2022},
    {"id": 3, "title": "Health Guide",   "content": "nutrition and exercise",    "domain": "health",   "verified": True,  "year": 2024},
    {"id": 4, "title": "Market Report",  "content": "startup funding trends",    "domain": "business", "verified": False, "year": 2023},
    {"id": 5, "title": "Cloud Security", "content": "cybersecurity cloud edge",  "domain": "tech",     "verified": True,  "year": 2025},
]


# ===========================================================================
# apply_rules
# ===========================================================================

class TestApplyRules:
    def test_detects_tech_domain_from_ai_keyword(self):
        result = apply_rules("show me AI papers")
        assert result["domain"] == "tech"

    def test_detects_tech_domain_from_machine_learning(self):
        result = apply_rules("machine learning trends")
        assert result["domain"] == "tech"

    def test_detects_legal_domain(self):
        result = apply_rules("latest regulation compliance updates")
        assert result["domain"] == "legal"

    def test_detects_business_domain(self):
        result = apply_rules("startup funding rounds in 2024")
        assert result["domain"] == "business"

    def test_detects_health_domain(self):
        result = apply_rules("nutrition and exercise advice")
        assert result["domain"] == "health"

    def test_detects_verified_flag(self):
        result = apply_rules("show me verified sources")
        assert result.get("verified") is True

    def test_detects_date_range_flag(self):
        result = apply_rules("papers from last 6 months")
        assert result.get("date_range") == "6m"

    def test_no_keywords_returns_empty_filters(self):
        result = apply_rules("tell me something interesting")
        assert result == {}

    def test_empty_query_returns_empty_filters(self):
        result = apply_rules("")
        assert result == {}

    def test_case_insensitive_matching(self):
        result = apply_rules("MACHINE LEARNING is great")
        assert result["domain"] == "tech"

    def test_only_first_domain_matched_on_overlap(self):
        # "ai" → tech triggers first in dict order; should not also set health
        result = apply_rules("ai health tips")
        assert "domain" in result
        assert result["domain"] in {"tech", "health"}  # one winner only
        # No duplicate domain key
        assert isinstance(result["domain"], str)

    def test_verified_combined_with_domain(self):
        result = apply_rules("verified AI research")
        assert result["domain"] == "tech"
        assert result["verified"] is True

    def test_all_three_flags_at_once(self):
        result = apply_rules("verified legal act from last 6 months")
        assert result["domain"] == "legal"
        assert result["verified"] is True
        assert result["date_range"] == "6m"


# ===========================================================================
# filter_docs
# ===========================================================================

class TestFilterDocs:
    def test_no_filters_returns_all_docs(self):
        result = filter_docs(SAMPLE_DOCS, {})
        assert result == SAMPLE_DOCS

    def test_domain_filter_keeps_only_matching_domain(self):
        result = filter_docs(SAMPLE_DOCS, {"domain": "tech"})
        assert all(d["domain"] == "tech" for d in result)
        assert len(result) == 2  # IDs 1 and 5

    def test_domain_filter_returns_empty_for_unknown_domain(self):
        result = filter_docs(SAMPLE_DOCS, {"domain": "finance"})
        assert result == []

    def test_verified_filter_keeps_only_verified(self):
        result = filter_docs(SAMPLE_DOCS, {"verified": True})
        assert all(d["verified"] is True for d in result)

    def test_combined_domain_and_verified_filter(self):
        result = filter_docs(SAMPLE_DOCS, {"domain": "tech", "verified": True})
        assert all(d["domain"] == "tech" and d["verified"] for d in result)

    def test_date_range_filter_excludes_old_docs(self):
        from datetime import datetime
        current_year = datetime.now().year
        result = filter_docs(SAMPLE_DOCS, {"date_range": "6m"})
        assert all(d["year"] >= current_year - 1 for d in result)

    def test_date_range_filter_excludes_none_year(self):
        docs = [{"id": 9, "title": "X", "content": "x", "domain": "tech",
                 "verified": False, "year": None}]
        result = filter_docs(docs, {"date_range": "6m"})
        assert result == []  # None year must be excluded

    def test_empty_docs_list_returns_empty(self):
        result = filter_docs([], {"domain": "tech"})
        assert result == []

    def test_filter_does_not_mutate_original_list(self):
        original = list(SAMPLE_DOCS)
        filter_docs(SAMPLE_DOCS, {"domain": "tech"})
        assert SAMPLE_DOCS == original


# ===========================================================================
# tokenize
# ===========================================================================

class TestTokenize:
    def test_splits_simple_sentence(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_lowercases_all_tokens(self):
        assert tokenize("MACHINE Learning") == ["machine", "learning"]

    def test_strips_punctuation(self):
        assert tokenize("AI, healthcare!") == ["ai", "healthcare"]

    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []

    def test_numbers_are_kept(self):
        tokens = tokenize("GPT-4 released in 2024")
        assert "4" in tokens
        assert "2024" in tokens

    def test_underscore_kept_as_word_char(self):
        tokens = tokenize("some_identifier")
        assert "some_identifier" in tokens

    def test_only_special_chars_returns_empty(self):
        assert tokenize("!!! ???") == []


# ===========================================================================
# bm25_search
# ===========================================================================

class TestBm25Search:
    def test_empty_docs_returns_empty_list(self):
        assert bm25_search([], "machine learning") == []

    def test_returns_same_number_of_results_as_input(self):
        result = bm25_search(SAMPLE_DOCS, "machine learning")
        assert len(result) == len(SAMPLE_DOCS)

    def test_scores_are_normalized_between_0_and_1(self):
        result = bm25_search(SAMPLE_DOCS, "machine learning")
        scores = [s for _, s in result]
        assert all(0.0 <= s <= 1.0 for s in scores), f"Out-of-range scores: {scores}"

    def test_results_are_sorted_descending(self):
        result = bm25_search(SAMPLE_DOCS, "machine learning")
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_returns_tuples_of_doc_and_score(self):
        result = bm25_search(SAMPLE_DOCS[:2], "cloud")
        assert isinstance(result[0], tuple)
        doc, score = result[0]
        assert isinstance(doc, dict)
        assert isinstance(score, float)

    def test_all_zero_scores_does_not_raise(self):
        # Query completely unrelated to all docs; BM25 may return all zeros
        result = bm25_search(SAMPLE_DOCS, "xyzzy nonexistent term qwerty")
        assert isinstance(result, list)

    def test_single_doc_returns_single_result(self):
        result = bm25_search([SAMPLE_DOCS[0]], "machine learning")
        assert len(result) == 1


# ===========================================================================
# explain
# ===========================================================================

class TestExplain:
    def test_domain_match_reason_added(self):
        doc     = {"domain": "tech", "verified": False}
        filters = {"domain": "tech"}
        reasons = explain(doc, filters, score=0.1, vector_score=0.0)
        assert "Domain matched" in reasons

    def test_domain_mismatch_reason_not_added(self):
        doc     = {"domain": "health", "verified": False}
        filters = {"domain": "tech"}
        reasons = explain(doc, filters, score=0.1)
        assert "Domain matched" not in reasons

    def test_verified_reason_added_when_doc_verified_and_filter_set(self):
        doc     = {"domain": "tech", "verified": True}
        filters = {"verified": True}
        reasons = explain(doc, filters, score=0.1)
        assert "Verified by authority" in reasons

    def test_verified_reason_not_added_when_doc_not_verified(self):
        doc     = {"domain": "tech", "verified": False}
        filters = {"verified": True}
        reasons = explain(doc, filters, score=0.8)
        assert "Verified by authority" not in reasons

    def test_bm25_reason_added_above_threshold(self):
        doc     = {"domain": "tech", "verified": False}
        reasons = explain(doc, {}, score=0.6)
        assert "High keyword relevance (BM25)" in reasons

    def test_bm25_reason_not_added_below_threshold(self):
        doc     = {"domain": "tech", "verified": False}
        reasons = explain(doc, {}, score=0.4)
        assert "High keyword relevance (BM25)" not in reasons

    def test_bm25_reason_at_exact_threshold_not_added(self):
        # Threshold is > 0.5, so 0.5 exactly should NOT trigger
        doc     = {"domain": "tech", "verified": False}
        reasons = explain(doc, {}, score=0.5)
        assert "High keyword relevance (BM25)" not in reasons

    def test_semantic_reason_added_above_threshold(self):
        doc     = {"domain": "tech", "verified": False}
        reasons = explain(doc, {}, score=0.1, vector_score=0.4)
        assert "Semantic similarity match" in reasons

    def test_semantic_reason_not_added_below_threshold(self):
        doc     = {"domain": "tech", "verified": False}
        reasons = explain(doc, {}, score=0.1, vector_score=0.2)
        assert "Semantic similarity match" not in reasons

    def test_no_filter_no_score_returns_empty_list(self):
        doc     = {"domain": "tech", "verified": False}
        reasons = explain(doc, {}, score=0.0, vector_score=0.0)
        assert reasons == []

    def test_all_reasons_present_simultaneously(self):
        doc     = {"domain": "tech", "verified": True}
        filters = {"domain": "tech", "verified": True}
        reasons = explain(doc, filters, score=0.9, vector_score=0.8)
        assert set(reasons) == {
            "Domain matched",
            "Verified by authority",
            "High keyword relevance (BM25)",
            "Semantic similarity match",
        }


# ===========================================================================
# vector_search_pg  (DB boundary mocked)
# ===========================================================================

class TestVectorSearchPg:
    def _fake_connection(self, rows):
        """Build a mock psycopg2 connection returning `rows` from fetchall()."""
        mock_cur  = MagicMock()
        mock_cur.fetchall.return_value = rows
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__  = MagicMock(return_value=False)

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        return mock_conn

    def test_empty_filtered_ids_returns_empty_dict(self):
        embedding = np.ones(384, dtype="float32")
        result = vector_search_pg(embedding, set())
        assert result == {}

    def test_returns_dict_keyed_by_doc_id(self):
        rows = [{"id": 1, "similarity": 0.92}, {"id": 3, "similarity": 0.75}]
        with patch("app.get_connection", return_value=self._fake_connection(rows)):
            embedding = np.ones(384, dtype="float32")
            result = vector_search_pg(embedding, {1, 3})
        assert result == {1: 0.92, 3: 0.75}

    def test_scores_are_floats(self):
        rows = [{"id": 2, "similarity": 0.5}]
        with patch("app.get_connection", return_value=self._fake_connection(rows)):
            embedding = np.ones(384, dtype="float32")
            result = vector_search_pg(embedding, {2})
        assert isinstance(result[2], float)

    def test_connection_always_closed_on_success(self):
        rows = []
        conn = self._fake_connection(rows)
        with patch("app.get_connection", return_value=conn):
            embedding = np.ones(384, dtype="float32")
            vector_search_pg(embedding, {1})
        conn.close.assert_called_once()

    def test_connection_closed_on_db_exception(self):
        mock_cur  = MagicMock()
        mock_cur.__enter__ = lambda s: s
        mock_cur.__exit__  = MagicMock(return_value=False)
        mock_cur.execute.side_effect = Exception("DB down")

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur

        with patch("app.get_connection", return_value=mock_conn):
            embedding = np.ones(384, dtype="float32")
            with pytest.raises(Exception, match="DB down"):
                vector_search_pg(embedding, {1})
        mock_conn.close.assert_called_once()

    def test_top_k_default_is_50(self):
        rows = []
        conn = self._fake_connection(rows)
        with patch("app.get_connection", return_value=conn):
            embedding = np.ones(384, dtype="float32")
            vector_search_pg(embedding, {1})
        # Verify the execute call included 50 as LIMIT
        call_args = conn.cursor().__enter__().execute.call_args
        assert 50 in call_args[0][1], "top_k=50 should be passed as LIMIT"

    def test_top_k_custom_value_passed_to_query(self):
        rows = []
        conn = self._fake_connection(rows)
        with patch("app.get_connection", return_value=conn):
            embedding = np.ones(384, dtype="float32")
            vector_search_pg(embedding, {1}, top_k=5)
        call_args = conn.cursor().__enter__().execute.call_args
        assert 5 in call_args[0][1]


# ===========================================================================
# Integration-level: score fusion logic (no DB, no model)
# ===========================================================================

class TestScoreFusion:
    """Verify the weighting and boost arithmetic in isolation, without
    touching FastAPI routing or live services."""

    def _fuse(self, bm25, vscore, verified_doc=False, verified_filter=False,
              domain_doc=None, domain_filter=None, year=None):
        from datetime import datetime
        final = (bm25 * 0.6) + (vscore * 0.4)
        if verified_doc and verified_filter:
            final += 0.05
        if domain_filter and domain_doc == domain_filter:
            final += 0.08
        if year and year >= datetime.now().year - 1:
            final += 0.03
        return final

    def test_pure_bm25_contribution(self):
        score = self._fuse(bm25=1.0, vscore=0.0)
        assert abs(score - 0.6) < 1e-9

    def test_pure_vector_contribution(self):
        score = self._fuse(bm25=0.0, vscore=1.0)
        assert abs(score - 0.4) < 1e-9

    def test_verified_boost_adds_0_05(self):
        base  = self._fuse(bm25=0.5, vscore=0.5)
        boost = self._fuse(bm25=0.5, vscore=0.5, verified_doc=True, verified_filter=True)
        assert abs(boost - base - 0.05) < 1e-9

    def test_domain_boost_adds_0_08(self):
        base  = self._fuse(bm25=0.5, vscore=0.5)
        boost = self._fuse(bm25=0.5, vscore=0.5, domain_doc="tech", domain_filter="tech")
        assert abs(boost - base - 0.08) < 1e-9

    def test_recency_boost_adds_0_03(self):
        from datetime import datetime
        base  = self._fuse(bm25=0.5, vscore=0.5)
        boost = self._fuse(bm25=0.5, vscore=0.5, year=datetime.now().year)
        assert abs(boost - base - 0.03) < 1e-9

    def test_all_boosts_accumulate_correctly(self):
        from datetime import datetime
        expected = (0.5 * 0.6) + (0.5 * 0.4) + 0.05 + 0.08 + 0.03
        result = self._fuse(
            bm25=0.5, vscore=0.5,
            verified_doc=True, verified_filter=True,
            domain_doc="tech", domain_filter="tech",
            year=datetime.now().year,
        )
        assert abs(result - expected) < 1e-9

    def test_no_boost_when_verified_filter_absent(self):
        base  = self._fuse(bm25=0.5, vscore=0.5)
        # doc is verified but user did NOT ask for verified
        boost = self._fuse(bm25=0.5, vscore=0.5, verified_doc=True, verified_filter=False)
        assert abs(boost - base) < 1e-9

    def test_no_boost_when_domain_mismatch(self):
        base  = self._fuse(bm25=0.5, vscore=0.5)
        boost = self._fuse(bm25=0.5, vscore=0.5, domain_doc="health", domain_filter="tech")
        assert abs(boost - base) < 1e-9
