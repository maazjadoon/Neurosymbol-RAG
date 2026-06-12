# db.py
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load .env file — must happen before any os.getenv() calls
load_dotenv()

# ── Raw psycopg2 connection ───────────────────────────────────────────────────
def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5433)),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

# ── Load all docs from PostgreSQL ─────────────────────────────────────────────
def load_docs():
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, content, domain, verified, year, embedding::text
                FROM documents
            """)
            results = []
            for row in cur.fetchall():
                d = dict(row)
                if "embedding" in d:
                    if d.get("embedding"):
                        # Parse the string format '[0.1,0.2,...]' into a list of floats
                        emb_str = d["embedding"].strip("[]")
                        d["embedding"] = [float(x) for x in emb_str.split(",") if x]
                    else:
                        d["embedding"] = []
                results.append(d)
            return results
    finally:
        conn.close()

# ── SQLAlchemy setup ──────────────────────────────────────────────────────────
# Build the URL inside a function so it reads env vars AFTER load_dotenv() runs.
# If built at module level, env vars may still be empty at that point.
def get_sqlalchemy_url():
    return (
        f"postgresql://"
        f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', 5433)}"
        f"/{os.getenv('DB_NAME')}"
    )

engine       = create_engine(get_sqlalchemy_url())
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_table():
    import models  # noqa: F401
    import models_neurosymbolic  # noqa: F401
    Base.metadata.create_all(bind=engine)