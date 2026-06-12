"""
tests/conftest.py
=================
Session-scoped setup run once before any test collection.

Purpose
-------
Both test_app.py and test_endpoints.py stub sys.modules["db"] with a fake
module so app.py can be imported without a live database.  test_db.py then
needs to import the REAL db module under a *different* sys.modules key so it
doesn't evict the stub.

This conftest restores the stub after test_db.py is done, ensuring the stub
is always in place when test_app.py / test_endpoints.py run regardless of
pytest collection order.
"""
import sys
import types
from unittest.mock import MagicMock
import pytest


@pytest.fixture(autouse=True, scope="session")
def _ensure_db_stub_in_sys_modules():
    """Guarantee sys.modules['db'] is always our fake before and after every
    test module, regardless of collection order."""
    # Build (or retrieve) the stub
    if "db" not in sys.modules or not isinstance(sys.modules["db"].get_connection
                                                  if hasattr(sys.modules["db"], "get_connection")
                                                  else None, MagicMock):
        from sqlalchemy.orm import declarative_base, sessionmaker
        fake_db = types.ModuleType("db")
        fake_db.load_docs      = MagicMock(return_value=[])
        fake_db.get_connection = MagicMock()
        fake_db.Base           = declarative_base()
        fake_db.SessionLocal   = sessionmaker()
        sys.modules["db"] = fake_db

    stub = sys.modules["db"]
    yield stub
    # After session: restore stub so any teardown code is safe
    sys.modules["db"] = stub
