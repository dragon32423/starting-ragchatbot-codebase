"""Tests for RAGSystem query handling.

Two layers:
  * Unit test with mocked components — proves the orchestration in
    RAGSystem.query() is correct.
  * Integration test (marked, opt-in) — runs the REAL system against the real
    ChromaDB and the real Anthropic API. This is the diagnostic that reproduces
    the "query failed" bug and surfaces the true exception.
"""

from unittest.mock import MagicMock, patch

import pytest

from rag_system import RAGSystem


def make_config():
    cfg = MagicMock()
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.CHROMA_PATH = "./chroma_db"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "test-model"
    return cfg


@patch("rag_system.SessionManager")
@patch("rag_system.AIGenerator")
@patch("rag_system.VectorStore")
@patch("rag_system.DocumentProcessor")
def test_query_returns_answer_and_sources(mock_dp, mock_vs, mock_ai, mock_sm):
    """RAGSystem.query() wires the AI answer + tool sources together."""
    mock_ai.return_value.generate_response.return_value = "MCP is a protocol."
    mock_sm.return_value.get_conversation_history.return_value = None

    rag = RAGSystem(make_config())
    # Stand in for the search tool's recorded sources.
    rag.tool_manager.get_last_sources = MagicMock(
        return_value=[
            {"text": "MCP: Build Rich-Context AI Apps - Lesson 1", "link": None}
        ]
    )
    rag.tool_manager.reset_sources = MagicMock()

    answer, sources = rag.query("What is MCP?", session_id="session_1")

    assert answer == "MCP is a protocol."
    assert sources == [
        {"text": "MCP: Build Rich-Context AI Apps - Lesson 1", "link": None}
    ]


@patch("rag_system.SessionManager")
@patch("rag_system.AIGenerator")
@patch("rag_system.VectorStore")
@patch("rag_system.DocumentProcessor")
def test_query_passes_tools_to_generator(mock_dp, mock_vs, mock_ai, mock_sm):
    """The generator must be invoked with the tool definitions + tool manager."""
    mock_ai.return_value.generate_response.return_value = "answer"
    mock_sm.return_value.get_conversation_history.return_value = None

    rag = RAGSystem(make_config())
    rag.query("What is MCP?", session_id="s1")

    kwargs = mock_ai.return_value.generate_response.call_args.kwargs
    assert kwargs["tools"] is not None
    assert kwargs["tool_manager"] is rag.tool_manager


@patch("rag_system.SessionManager")
@patch("rag_system.AIGenerator")
@patch("rag_system.VectorStore")
@patch("rag_system.DocumentProcessor")
def test_query_surfaces_generator_exception(mock_dp, mock_vs, mock_ai, mock_sm):
    """If the AI layer raises (e.g. an API auth error), query() currently lets it
    propagate — which is exactly what becomes the HTTP 500 / 'query failed'."""
    mock_ai.return_value.generate_response.side_effect = RuntimeError("boom")
    mock_sm.return_value.get_conversation_history.return_value = None

    rag = RAGSystem(make_config())
    with pytest.raises(RuntimeError, match="boom"):
        rag.query("What is MCP?", session_id="s1")


# --- Integration / diagnostic -------------------------------------------------
@pytest.mark.integration
def test_configured_model_is_accessible():
    """Fast diagnostic: assert config.ANTHROPIC_MODEL is one the account can actually
    use. Catches the '404 model not found' failure instantly, without a full query,
    and guards against config drifting to a retired/inaccessible model id."""
    import anthropic

    from config import config

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    available = {m.id for m in client.models.list(limit=100).data}

    assert config.ANTHROPIC_MODEL in available, (
        f"Configured ANTHROPIC_MODEL '{config.ANTHROPIC_MODEL}' is not accessible to this "
        f"account. Available models: {sorted(available)}"
    )


@pytest.mark.integration
def test_content_query_end_to_end():
    """Reproduce the real user flow: real ChromaDB + real Anthropic API.

    If "query failed" is caused by the API call, this test fails here with the
    real exception (auth, model, etc.), pinpointing the broken layer.
    """
    from config import config

    rag = RAGSystem(config)
    answer, sources = rag.query("What is covered in the MCP course?")

    assert isinstance(answer, str) and answer.strip(), "Expected a non-empty answer"
    # Must be a REAL answer, not the graceful API-error fallback. This fails (with a
    # clear reason) whenever the Anthropic call is broken -- e.g. no credit balance.
    assert "AI service is currently unavailable" not in answer, answer


@pytest.mark.integration
def test_outline_query_end_to_end():
    """Reproduce the reported 'Outline of a course' failure end-to-end.

    Pre-fix this raised IndexError (empty final content) -> 'query failed'. Post-fix
    the get_course_outline tool returns a real lesson list and no crash occurs.
    """
    from config import config

    rag = RAGSystem(config)
    answer, sources = rag.query("Give me the outline of the MCP course")

    assert isinstance(answer, str) and answer.strip(), "Expected a non-empty answer"
    assert "AI service is currently unavailable" not in answer, answer
    assert "couldn't generate a response" not in answer, answer
    # A real outline should reference lessons.
    assert "lesson" in answer.lower() or "Lesson" in answer
