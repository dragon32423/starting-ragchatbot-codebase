# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A full-stack RAG (Retrieval-Augmented Generation) system for querying course materials. ChromaDB stores embeddings, Anthropic's Claude generates answers, FastAPI serves a JSON API plus the static frontend.

## Commands

**Run the app** (serves frontend + API on http://localhost:8000, API docs at `/docs`):
```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync
```

**Always use `uv`** for running Python and managing dependencies in this repo — never bare `python`/`pip`:
```bash
uv run python <script.py>   # run any Python file
uv add <package>            # add a new dependency (updates pyproject.toml + uv.lock)
```

**Run tests** (from repo root; `testpaths` is set to `backend/tests` in `pyproject.toml`):
```bash
uv run pytest
uv run pytest backend/tests/test_rag_system.py     # single file
uv run pytest backend/tests/test_rag_system.py::test_name  # single test
uv run pytest -m "not integration"   # skip tests that hit the real Anthropic API / ChromaDB
```

There is no lint/format command configured in this repo.

Environment: requires `ANTHROPIC_API_KEY` in a `.env` file at the repo root (loaded via `python-dotenv` in `backend/config.py`).

## Architecture

### Request flow (tool-calling agentic RAG, not naive retrieve-then-generate)

`frontend/script.js` → `POST /api/query` → `backend/app.py` → `RAGSystem.query()` (`backend/rag_system.py`) → `AIGenerator.generate_response()` (`backend/ai_generator.py`).

The AI generator gives Claude tool definitions and lets it *decide* whether to search:
1. First call to Claude includes `tools` + `tool_choice: auto`. Claude either answers directly (general knowledge) or returns `stop_reason == "tool_use"`.
2. If a tool is requested, `_handle_tool_execution()` runs it via `ToolManager.execute_tool()`, appends the `tool_result`, and makes a **second** Claude call (no tools this time) to synthesize the final answer.
3. System prompt enforces **one tool call per query maximum**.

Sources are NOT part of Claude's text response — they're a side channel. Tools (`backend/search_tools.py`) stash `last_sources` as an instance attribute during `execute()`; `RAGSystem.query()` pulls them via `ToolManager.get_last_sources()` and explicitly calls `reset_sources()` after each query. Any change to tool execution must preserve this reset, or sources will leak across queries/sessions.

Two tools are registered with `ToolManager`:
- `CourseSearchTool` — semantic content search, optionally filtered by `course_name` (fuzzy/semantic match, not exact) and `lesson_number`.
- `CourseOutlineTool` — returns a course's title, link, and full lesson list.

### Vector store (`backend/vector_store.py`)

ChromaDB with two separate collections:
- `course_catalog` — one entry per course (title as ID), used only to resolve fuzzy course names to exact titles via semantic search (`_resolve_course_name`). Lessons are stored serialized as a `lessons_json` string in metadata (ChromaDB metadata values must be primitives).
- `course_content` — the actual chunked text, filterable by `course_title`/`lesson_number` via ChromaDB `where` clauses.

`CourseSearchTool.execute()` always resolves `course_name` against `course_catalog` first, then queries `course_content` with the resolved exact title — so course name matching is semantic/fuzzy even though content filtering is exact.

### Document ingestion (`backend/document_processor.py`)

Course documents are plain text files with a required header format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [lesson title]
Lesson Link: [url]
[lesson content...]

Lesson 1: ...
```
Chunking is sentence-aware (regex-based sentence splitting, not naive char slicing) with configurable size/overlap, and each chunk is prefixed with course/lesson context (e.g. `"Course X Lesson N content: ..."`) before embedding, so retrieved chunks carry context independent of metadata.

On FastAPI startup (`app.py`), `../docs` is scanned and any course not already in the vector store (by title) is added — existing courses are skipped, so re-adding a doc with the same title is a no-op rather than a duplicate.

### Session/history

`SessionManager` (`backend/session_manager.py`) keeps an in-memory dict of session_id → message list, trimmed to `MAX_HISTORY` exchanges. There is no persistence — restarting the server drops all sessions. History is formatted as a flat `"Role: content"` string and injected into Claude's system prompt, not as structured message turns.

### Config

All tunables live in `backend/config.py` as a single `Config` dataclass instance (`config`), constructed once and imported everywhere (`from config import config`). Notable: `MAX_RESULTS = 5` (search result count), `CHUNK_SIZE = 800` / `CHUNK_OVERLAP = 100`, `MAX_HISTORY = 2` (conversation exchanges remembered), `CHROMA_PATH` anchored to the backend directory (independent of CWD, since the app runs from `backend/` but tests run from repo root).

### Import style

Backend modules use flat imports (`from config import config`, `from vector_store import VectorStore`), not package-relative imports. `backend/tests/conftest.py` inserts the `backend` directory onto `sys.path` to make this work when running tests from the repo root.
