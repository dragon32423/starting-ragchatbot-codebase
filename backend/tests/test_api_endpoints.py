"""Tests for the FastAPI HTTP layer (request validation + response shaping).

These run against an inline test app (see ``test_app`` in conftest.py) backed by a
mock RAGSystem, so they exercise the real routing/serialization without touching
ChromaDB, the Anthropic API, or the static frontend mount in backend/app.py.
"""
import pytest

# Every test in this module hits the HTTP layer; tag them so they can be run (or
# skipped) as a group, e.g. `uv run pytest -m api`.
pytestmark = pytest.mark.api


# --- POST /api/query ----------------------------------------------------------
class TestQueryEndpoint:
    def test_query_with_session_returns_answer_and_sources(self, client, mock_rag_system):
        resp = client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "existing-session"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "MCP is a protocol for connecting AI clients to servers."
        assert body["session_id"] == "existing-session"
        assert body["sources"] == [
            {"text": "MCP: Build Rich-Context AI Apps - Lesson 1", "link": "http://example.com/lesson1"}
        ]
        # The provided session id is forwarded to the RAG system as-is.
        mock_rag_system.query.assert_called_once_with("What is MCP?", "existing-session")

    def test_query_without_session_creates_one(self, client, mock_rag_system):
        resp = client.post("/api/query", json={"query": "What is MCP?"})

        assert resp.status_code == 200
        body = resp.json()
        # Falls back to a freshly created session id.
        assert body["session_id"] == "test-session-1"
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is MCP?", "test-session-1")

    def test_query_missing_query_field_is_422(self, client):
        resp = client.post("/api/query", json={"session_id": "s1"})
        assert resp.status_code == 422  # pydantic validation error

    def test_query_empty_sources_serializes_to_empty_list(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("General knowledge answer.", [])

        resp = client.post("/api/query", json={"query": "Hello"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "General knowledge answer."
        assert body["sources"] == []

    def test_query_propagates_internal_error_as_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("boom")

        resp = client.post("/api/query", json={"query": "What is MCP?"})

        assert resp.status_code == 500
        assert "RuntimeError: boom" in resp.json()["detail"]


# --- GET /api/courses ---------------------------------------------------------
class TestCoursesEndpoint:
    def test_courses_returns_stats(self, client, mock_rag_system):
        resp = client.get("/api/courses")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_courses"] == 2
        assert body["course_titles"] == [
            "MCP: Build Rich-Context AI Apps",
            "Advanced RAG",
        ]
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_courses_error_is_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = ValueError("db down")

        resp = client.get("/api/courses")

        assert resp.status_code == 500
        assert "db down" in resp.json()["detail"]


# --- POST /api/session/clear --------------------------------------------------
class TestClearSessionEndpoint:
    def test_clear_session_ok(self, client, mock_rag_system):
        resp = client.post("/api/session/clear", json={"session_id": "s1"})

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        mock_rag_system.session_manager.end_session.assert_called_once_with("s1")

    def test_clear_session_missing_id_is_422(self, client):
        resp = client.post("/api/session/clear", json={})
        assert resp.status_code == 422


# --- GET / (root / frontend) --------------------------------------------------
class TestRootEndpoint:
    def test_root_serves_html(self, client):
        resp = client.get("/")

        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
