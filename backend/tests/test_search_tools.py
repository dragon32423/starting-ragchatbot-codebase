"""Unit tests for CourseSearchTool.execute and ToolManager.

These use a mocked VectorStore, so they exercise the tool's own logic without
touching ChromaDB or the Anthropic API. If these pass, the search/tool layer is
sound and the "query failed" bug lives elsewhere.
"""
from unittest.mock import MagicMock

from vector_store import SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager


# --- CourseSearchTool.execute -------------------------------------------------
def test_execute_returns_formatted_results(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)

    output = tool.execute(query="what is MCP")

    # Each result gets a [course - lesson] header followed by the document.
    assert "[MCP: Build Rich-Context AI Apps - Lesson 1]" in output
    assert "MCP lets clients connect to servers." in output
    assert "[MCP: Build Rich-Context AI Apps - Lesson 2]" in output


def test_execute_passes_filters_to_store(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="topic", course_name="MCP", lesson_number=2)

    mock_vector_store.search.assert_called_once_with(
        query="topic", course_name="MCP", lesson_number=2
    )


def test_execute_returns_error_string_verbatim(mock_vector_store, error_results):
    mock_vector_store.search.return_value = error_results
    tool = CourseSearchTool(mock_vector_store)

    output = tool.execute(query="anything", course_name="Nonexistent")

    assert output == "No course found matching 'Nonexistent'"


def test_execute_empty_results_message(mock_vector_store, empty_results):
    mock_vector_store.search.return_value = empty_results
    tool = CourseSearchTool(mock_vector_store)

    output = tool.execute(query="anything")

    assert output == "No relevant content found."


def test_execute_empty_results_includes_filter_info(mock_vector_store, empty_results):
    mock_vector_store.search.return_value = empty_results
    tool = CourseSearchTool(mock_vector_store)

    output = tool.execute(query="x", course_name="MCP", lesson_number=3)

    assert "in course 'MCP'" in output
    assert "in lesson 3" in output


def test_execute_populates_last_sources(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)

    tool.execute(query="what is MCP")

    assert tool.last_sources == [
        {"text": "MCP: Build Rich-Context AI Apps - Lesson 1", "link": None},
        {"text": "MCP: Build Rich-Context AI Apps - Lesson 2", "link": None},
    ]


def test_execute_handles_missing_lesson_number(mock_vector_store):
    mock_vector_store.search.return_value = SearchResults(
        documents=["General course intro"],
        metadata=[{"course_title": "Some Course", "chunk_index": 0}],
        distances=[0.1],
    )
    tool = CourseSearchTool(mock_vector_store)

    output = tool.execute(query="intro")

    assert "[Some Course]" in output  # no " - Lesson N" suffix
    assert tool.last_sources == [{"text": "Some Course", "link": None}]


# --- get_tool_definition ------------------------------------------------------
def test_tool_definition_shape(mock_vector_store):
    tool = CourseSearchTool(mock_vector_store)

    definition = tool.get_tool_definition()

    assert definition["name"] == "search_course_content"
    assert "query" in definition["input_schema"]["properties"]
    assert definition["input_schema"]["required"] == ["query"]


# --- ToolManager --------------------------------------------------------------
def test_tool_manager_registers_and_executes(mock_vector_store):
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))

    defs = manager.get_tool_definitions()
    assert any(d["name"] == "search_course_content" for d in defs)

    result = manager.execute_tool("search_course_content", query="what is MCP")
    assert "MCP lets clients connect to servers." in result


def test_tool_manager_unknown_tool(mock_vector_store):
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))

    assert manager.execute_tool("does_not_exist", query="x") == "Tool 'does_not_exist' not found"


def test_tool_manager_sources_lifecycle(mock_vector_store):
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))

    manager.execute_tool("search_course_content", query="what is MCP")
    assert manager.get_last_sources()  # populated

    manager.reset_sources()
    assert manager.get_last_sources() == []


# --- CourseOutlineTool --------------------------------------------------------
COURSE_TITLE = "MCP: Build Rich-Context AI Apps"


def make_outline_store():
    """A VectorStore mock that resolves a course name and returns its metadata."""
    store = MagicMock()
    store._resolve_course_name.return_value = COURSE_TITLE
    store.get_all_courses_metadata.return_value = [
        {
            "title": COURSE_TITLE,
            "instructor": "Someone",
            "course_link": "https://example.com/mcp",
            "lessons": [
                {"lesson_number": 2, "lesson_title": "Servers", "lesson_link": None},
                {"lesson_number": 1, "lesson_title": "Introduction", "lesson_link": None},
            ],
            "lesson_count": 2,
        }
    ]
    return store


def test_outline_tool_definition_shape():
    tool = CourseOutlineTool(MagicMock())

    definition = tool.get_tool_definition()

    assert definition["name"] == "get_course_outline"
    assert definition["input_schema"]["required"] == ["course_name"]


def test_outline_formats_title_link_and_lessons():
    tool = CourseOutlineTool(make_outline_store())

    output = tool.execute(course_name="MCP")

    assert f"Course: {COURSE_TITLE}" in output
    assert "Course link: https://example.com/mcp" in output
    assert "Lessons (2):" in output
    # Lessons are ordered by lesson_number regardless of stored order.
    assert output.index("1. Introduction") < output.index("2. Servers")


def test_outline_resolves_course_name():
    store = make_outline_store()
    tool = CourseOutlineTool(store)

    tool.execute(course_name="MCP")

    store._resolve_course_name.assert_called_once_with("MCP")


def test_outline_populates_last_sources():
    tool = CourseOutlineTool(make_outline_store())

    tool.execute(course_name="MCP")

    assert tool.last_sources == [
        {"text": COURSE_TITLE, "link": "https://example.com/mcp"}
    ]


def test_outline_unknown_course_returns_message():
    store = MagicMock()
    store._resolve_course_name.return_value = None
    tool = CourseOutlineTool(store)

    output = tool.execute(course_name="Nonexistent")

    assert output == "No course found matching 'Nonexistent'."


def test_outline_via_tool_manager():
    manager = ToolManager()
    manager.register_tool(CourseOutlineTool(make_outline_store()))

    output = manager.execute_tool("get_course_outline", course_name="MCP")

    assert "1. Introduction" in output
