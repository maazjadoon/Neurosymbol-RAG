"""
tests/test_db.py
================
Unit tests for db.py  — standalone file, imports the REAL db module.

Isolation strategy
------------------
- We load db.py under a private module alias ('_real_db') using importlib so
  we never evict the stub at sys.modules['db'] that test_app / test_endpoints
  depend on.
- psycopg2.connect  → patched via patch.object(real_db, ...)
- SQLAlchemy engine → mocked via the declarative_base stub.
- dotenv            → already loaded; env vars are set before import.
- No live database required.

Run with:
    pytest tests/test_db.py -v
"""

import sys
import types
import os
import importlib
import importlib.util
import pytest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Provide stubs for all packages db.py imports at module level.
# These must be in sys.modules BEFORE we load db.py.
# ---------------------------------------------------------------------------

# 1. dotenv
if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = MagicMock()
    sys.modules["dotenv"] = fake_dotenv

# 2. Env vars
os.environ.update({
    "DB_HOST":     "localhost",
    "DB_PORT":     "5433",
    "DB_NAME":     "ragdb",
    "DB_USER":     "testuser",
    "DB_PASSWORD": "testpass",
})

# 3. psycopg2
if "psycopg2" not in sys.modules:
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_psycopg2.connect = MagicMock()
    fake_extras = types.ModuleType("psycopg2.extras")
    class _RealDictCursor: pass
    fake_extras.RealDictCursor = _RealDictCursor
    fake_psycopg2.extras = fake_extras
    sys.modules["psycopg2"]        = fake_psycopg2
    sys.modules["psycopg2.extras"] = fake_extras

# 4. sqlalchemy
if "sqlalchemy" not in sys.modules:
    fake_sa = types.ModuleType("sqlalchemy")
    fake_sa.create_engine = MagicMock(return_value=MagicMock())
    sys.modules["sqlalchemy"] = fake_sa

if "sqlalchemy.orm" not in sys.modules:
    fake_orm = types.ModuleType("sqlalchemy.orm")
    fake_orm.sessionmaker     = MagicMock(return_value=MagicMock())
    _mock_base_instance       = MagicMock()
    fake_orm.declarative_base = MagicMock(return_value=_mock_base_instance)
    sys.modules["sqlalchemy.orm"] = fake_orm

# ---------------------------------------------------------------------------
# Load db.py as '_real_db' without touching sys.modules['db'] (the stub).
# ---------------------------------------------------------------------------

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db.py")
_spec     = importlib.util.spec_from_file_location("_real_db", _DB_PATH)
real_db   = importlib.util.module_from_spec(_spec)
# Temporarily register under its private name so relative deps resolve
sys.modules["_real_db"] = real_db
_spec.loader.exec_module(real_db)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cursor(rows):
    cur = MagicMock()
    cur.fetchall.return_value = rows
    cur.__enter__ = lambda s: s
    cur.__exit__  = MagicMock(return_value=False)
    return cur


def _make_conn(cursor):
    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


# ===========================================================================
# real_db.load_docs
# ===========================================================================

class TestLoadDocs:

    def test_returns_list_of_dicts(self):
        row = {"id": 1, "title": "A", "content": "hello",
               "domain": "tech", "verified": True, "year": 2024}
        cur  = _make_cursor([row])
        conn = _make_conn(cur)
        with patch.object(real_db, "get_connection", return_value=conn):
            result = real_db.load_docs()
        assert isinstance(result, list)
        assert result[0] == dict(row)

    def test_returns_all_rows(self):
        rows = [
            {"id": i, "title": f"D{i}", "content": "x",
             "domain": "tech", "verified": False, "year": 2023}
            for i in range(5)
        ]
        cur  = _make_cursor(rows)
        conn = _make_conn(cur)
        with patch.object(real_db, "get_connection", return_value=conn):
            result = real_db.load_docs()
        assert len(result) == 5

    def test_empty_table_returns_empty_list(self):
        cur  = _make_cursor([])
        conn = _make_conn(cur)
        with patch.object(real_db, "get_connection", return_value=conn):
            result = real_db.load_docs()
        assert result == []

    def test_connection_closed_on_success(self):
        cur  = _make_cursor([])
        conn = _make_conn(cur)
        with patch.object(real_db, "get_connection", return_value=conn):
            real_db.load_docs()
        conn.close.assert_called_once()

    def test_connection_closed_on_db_exception(self):
        cur = MagicMock()
        cur.__enter__ = lambda s: s
        cur.__exit__  = MagicMock(return_value=False)
        cur.execute.side_effect = Exception("connection refused")

        conn = MagicMock()
        conn.cursor.return_value = cur

        with patch.object(real_db, "get_connection", return_value=conn):
            with pytest.raises(Exception, match="connection refused"):
                real_db.load_docs()
        conn.close.assert_called_once()

    def test_selects_correct_columns(self):
        cur  = _make_cursor([])
        conn = _make_conn(cur)
        with patch.object(real_db, "get_connection", return_value=conn):
            real_db.load_docs()
        sql = cur.execute.call_args[0][0]
        for col in ("id", "title", "content", "domain", "verified", "year"):
            assert col in sql, f"Column '{col}' missing from SELECT"

    def test_rows_converted_to_plain_dicts(self):
        """RealDictRow objects must be converted via dict()."""
        class RealDictRow(dict):
            pass

        raw = [RealDictRow({"id": 1, "title": "T", "content": "c",
                             "domain": "tech", "verified": True, "year": 2025})]
        cur  = _make_cursor(raw)
        conn = _make_conn(cur)
        with patch.object(real_db, "get_connection", return_value=conn):
            result = real_db.load_docs()
        assert type(result[0]) is dict


# ===========================================================================
# real_db.create_table
# ===========================================================================

class TestCreateTable:

    def test_create_all_called_on_engine(self):
        fake_models = types.ModuleType("models")
        # real_db.Base is the mock returned by declarative_base()
        with patch.dict(sys.modules, {"models": fake_models}):
            real_db.create_table()
        real_db.Base.metadata.create_all.assert_called_once_with(bind=real_db.engine)

    def test_create_table_returns_none(self):
        fake_models = types.ModuleType("models")
        with patch.dict(sys.modules, {"models": fake_models}):
            result = real_db.create_table()
        assert result is None


# ===========================================================================
# real_db.get_sqlalchemy_url
# ===========================================================================

class TestGetSqlalchemyUrl:

    def test_url_contains_all_env_components(self):
        with patch.dict(os.environ, {
            "DB_USER": "alice", "DB_PASSWORD": "secret",
            "DB_HOST": "dbhost", "DB_PORT": "5432", "DB_NAME": "mydb"
        }):
            url = real_db.get_sqlalchemy_url()
        assert "alice"  in url
        assert "secret" in url
        assert "dbhost" in url
        assert "5432"   in url
        assert "mydb"   in url

    def test_url_starts_with_postgresql_scheme(self):
        url = real_db.get_sqlalchemy_url()
        assert url.startswith("postgresql://")

    def test_host_defaults_to_localhost_when_env_unset(self):
        env = dict(os.environ)
        env.pop("DB_HOST", None)
        with patch.dict(os.environ, env, clear=True):
            url = real_db.get_sqlalchemy_url()
        assert "localhost" in url

