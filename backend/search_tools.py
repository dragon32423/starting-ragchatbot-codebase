from abc import ABC, abstractmethod
from typing import Any

from vector_store import SearchResults, VectorStore


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: str | None = None,
        lesson_number: int | None = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI

        for doc, meta in zip(results.documents, results.metadata, strict=False):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI
            source = course_title
            link = None
            if lesson_num is not None:
                source += f" - Lesson {lesson_num}"
                link = self.store.get_lesson_link(course_title, lesson_num)
            sources.append({"text": source, "link": link})

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for returning a course's outline: title, link, and full lesson list."""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last lookup

    def get_tool_definition(self) -> dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_course_outline",
            "description": (
                "Get the outline of a course: its title, course link, and the complete "
                "numbered list of lessons. Use this for questions about a course's outline, "
                "syllabus, or list of lessons."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    }
                },
                "required": ["course_name"],
            },
        }

    def execute(self, course_name: str) -> str:
        """
        Resolve the course and return its formatted outline.

        Args:
            course_name: Course title (partial matches work)

        Returns:
            Formatted course outline or an error message
        """
        # Reuse the vector store's semantic course-name resolution.
        course_title = self.store._resolve_course_name(course_name)
        if not course_title:
            return f"No course found matching '{course_name}'."

        # Pull the matching course metadata (title, link, parsed lessons).
        course_meta = None
        for meta in self.store.get_all_courses_metadata():
            if meta.get("title") == course_title:
                course_meta = meta
                break

        if not course_meta:
            return f"No outline available for course '{course_title}'."

        return self._format_outline(course_meta)

    def _format_outline(self, course_meta: dict[str, Any]) -> str:
        """Format course metadata into a readable outline."""
        title = course_meta.get("title", "unknown")
        course_link = course_meta.get("course_link")
        lessons = course_meta.get("lessons", [])

        lines = [f"Course: {title}"]
        if course_link:
            lines.append(f"Course link: {course_link}")
        lines.append(f"Lessons ({len(lessons)}):")

        # Lessons are ordered by lesson_number for a stable outline.
        for lesson in sorted(lessons, key=lambda x: x.get("lesson_number", 0)):
            number = lesson.get("lesson_number")
            lesson_title = lesson.get("lesson_title", "")
            lines.append(f"  {number}. {lesson_title}")

        # Track source for the UI.
        self.last_sources = [{"text": title, "link": course_link}]

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
