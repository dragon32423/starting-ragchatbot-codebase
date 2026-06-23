"""Shared fixtures for the backend test suite.

Backend modules use flat imports (e.g. ``from config import config``), so we
insert the ``backend`` directory onto ``sys.path`` before any test imports.
"""
import os
import sys
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

# --- Make backend/* importable (flat imports) ---------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from vector_store import SearchResults  # noqa: E402


# --- SearchResults fixtures ---------------------------------------------------
@pytest.fixture
def success_results():
    """A populated SearchResults, as the vector store returns on a good hit."""
    return SearchResults(
        documents=[
            "Lesson 1 content: MCP lets clients connect to servers.",
            "Servers expose tools, resources and prompts.",
        ],
        metadata=[
            {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 2, "chunk_index": 5},
        ],
        distances=[0.12, 0.34],
    )


@pytest.fixture
def empty_results():
    """An empty (but non-error) SearchResults."""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_results():
    """A SearchResults carrying an error message."""
    return SearchResults.empty("No course found matching 'Nonexistent'")


# --- Mocked VectorStore -------------------------------------------------------
@pytest.fixture
def mock_vector_store(success_results):
    """A VectorStore mock whose .search() returns success_results by default."""
    store = MagicMock()
    store.search.return_value = success_results
    store.get_lesson_link.return_value = None
    store.get_course_link.return_value = None
    return store


# --- Anthropic response builders ----------------------------------------------
def make_text_response(text):
    """Build a fake Messages API response that is a plain text answer."""
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(stop_reason="end_turn", content=[block])


def make_tool_use_response(tool_name, tool_input, tool_id="toolu_123"):
    """Build a fake Messages API response that requests a tool call."""
    block = SimpleNamespace(
        type="tool_use",
        name=tool_name,
        input=tool_input,
        id=tool_id,
    )
    return SimpleNamespace(stop_reason="tool_use", content=[block])


@pytest.fixture
def text_response_factory():
    return make_text_response


@pytest.fixture
def tool_use_response_factory():
    return make_tool_use_response


# --- API endpoint testing -----------------------------------------------------
# backend/app.py mounts the static frontend ("../frontend") at import time and
# constructs a real RAGSystem (real ChromaDB + Anthropic client). Importing it in
# the test environment therefore fails (no frontend dir) and would hit the network.
#
# To test the HTTP layer in isolation we build an equivalent FastAPI app inline,
# backed by a mock RAGSystem. The route signatures/models mirror app.py exactly so
# these tests still exercise real request validation and response serialization.


@pytest.fixture
def mock_rag_system():
    """A RAGSystem mock with sensible defaults for the API layer.

    - ``query()`` returns an (answer, sources) tuple like the real system.
    - ``session_manager.create_session()`` hands back a stable id.
    - ``get_course_analytics()`` returns a small catalog.
    """
    rag = MagicMock()
    rag.query.return_value = (
        "MCP is a protocol for connecting AI clients to servers.",
        [{"text": "MCP: Build Rich-Context AI Apps - Lesson 1", "link": "http://example.com/lesson1"}],
    )
    rag.session_manager.create_session.return_value = "test-session-1"
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["MCP: Build Rich-Context AI Apps", "Advanced RAG"],
    }
    return rag


@pytest.fixture
def test_app(mock_rag_system):
    """A FastAPI app mirroring backend/app.py's routes, wired to mock_rag_system.

    Defined inline (no static-file mount, no real RAGSystem) so it imports cleanly
    and stays offline. Returns the app; pair with ``client`` for requests.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class ClearSessionRequest(BaseModel):
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    app = FastAPI(title="Course Materials RAG System (test)")
    rag_system = mock_rag_system

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    @app.post("/api/session/clear")
    async def clear_session(request: ClearSessionRequest):
        try:
            rag_system.session_manager.end_session(request.session_id)
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Stand-in for the real static frontend mounted at "/" in app.py.
    @app.get("/", response_class=HTMLResponse)
    async def root():
        return "<html><body>RAG frontend</body></html>"

    return app


@pytest.fixture
def client(test_app):
    """A TestClient bound to the inline test app."""
    from fastapi.testclient import TestClient

    return TestClient(test_app)
