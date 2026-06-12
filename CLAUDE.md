# Neurosymbol RAG — Project Guidelines

## Tech Stack
* **Backend**: Python 3.10+, FastAPI, `psycopg2` (raw SQL vector operations), SQLAlchemy (database ORM / models), `rank-bm25` (BM25Okapi), `sentence-transformers` (all-MiniLM-L6-v2)
* **Frontend**: Vanilla HTML5, Vanilla CSS, Vanilla JavaScript (No frameworks, Tailwind, or bundling tools)
* **Database**: PostgreSQL 18+ with `pgvector` extension

---

## Commands

* **Start Dev Server**:
  ```powershell
  rag_venv\Scripts\uvicorn app:app --reload
  ```
* **Run Test Suite**:
  ```powershell
  rag_venv\Scripts\python -m pytest
  ```
* **Run Single Test File**:
  ```powershell
  rag_venv\Scripts\python -m pytest tests/test_app.py
  ```
* **Install Dependencies**:
  ```powershell
  rag_venv\Scripts\pip install -r requirements.txt
  ```

---

## Code Conventions

### Backend (Python)
* **API Endpoints**: Kept in [app.py](file:///e:/Projects/Neurosymbol%20RAG/app.py). Ensure routes are clearly documented.
* **Database Access**: 
  - Raw SQL queries via psycopg2 connection [db.py:get_connection()](file:///e:/Projects/Neurosymbol%20RAG/db.py#L13) are preferred for vector math (`<=>` similarity operator).
  - Always use `psycopg2.extras.RealDictCursor` for returning query outputs as dictionaries.
  - Wrap database connection blocks in `try/finally` to guarantee `conn.close()` is called on every query.
* **Database Models**: Defined in [models.py](file:///e:/Projects/Neurosymbol%20RAG/models.py). Always import `Vector` from `pgvector.sqlalchemy` when modeling embedding fields.
* **Performance**: Optimize iteration loops. Avoid making redundant calls like `datetime.now()` inside doc processing or scoring loops; calculate values beforehand.

### Frontend (HTML/CSS/JS)
* **Visibility Control**: Toggle visibility by updating both the HTML `hidden` property AND the inline `element.style.display` (`'none'` or `''`) to ensure robust behavior regardless of class specificity.
* **Styles & Asset Updates**: Always cache-bust imported custom CSS/JS references in [index.html](file:///e:/Projects/Neurosymbol%20RAG/static/index.html) (e.g., `style.css?v=X` and `app.js?v=X`) when making updates.
* **Design System**: Use the custom variables declared in `:root` inside [style.css](file:///e:/Projects/Neurosymbol%20RAG/static/style.css) (such as `--color-bg`, `--color-surface-1`, `--space-*` tokens). Avoid hardcoded hex values or arbitrary spacings.
* **Accessibility (WCAG 2.1 AA)**: 
  - Ensure all interactive controls have proper labels, `aria-*` tags, and `role` indicators.
  - Animate progress elements (e.g., score bars) dynamically by adjusting width inside a deferred `requestAnimationFrame` callback.
  - Implement full keyboard accessibility (`tabindex`, focus management, keyboard shortcuts).

---

## Database Architecture
* Database name: `ragdb` (configured via `.env` variables)
* Primary table: `documents` containing columns:
  - `id` (serial primary key)
  - `title` (varchar)
  - `content` (text)
  - `domain` (varchar)
  - `verified` (boolean)
  - `year` (integer)
  - `embedding` (vector(384) matching `all-MiniLM-L6-v2` dimensions)
