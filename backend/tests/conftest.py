"""Shared fixtures for the backend test suite.

Backend modules use flat imports (e.g. ``from config import config``), so we
insert the ``backend`` directory onto ``sys.path`` before any test imports.
"""
import os
import sys
from types import SimpleNamespace
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
