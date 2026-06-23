"""Tests for AIGenerator's tool-calling logic.

The anthropic client is mocked, so these verify the orchestration (does it ask
for tools, does it execute them, does it round-trip the results) without any
network calls.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx

from ai_generator import AIGenerator


def make_text_response(text):
    """Fake Messages API response that is a plain text answer."""
    return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)])


def make_tool_use_response(tool_name, tool_input, tool_id="toolu_123"):
    """Fake Messages API response that requests a tool call."""
    block = SimpleNamespace(type="tool_use", name=tool_name, input=tool_input, id=tool_id)
    return SimpleNamespace(stop_reason="tool_use", content=[block])


def make_empty_response():
    """Fake response with an empty content list (the 'outline' edge case)."""
    return SimpleNamespace(stop_reason="end_turn", content=[])


def build_generator(mock_responses):
    """Create an AIGenerator whose mocked client yields mock_responses in order."""
    with patch("ai_generator.anthropic.Anthropic") as mock_anthropic:
        client = MagicMock()
        client.messages.create.side_effect = mock_responses
        mock_anthropic.return_value = client
        generator = AIGenerator(api_key="test-key", model="test-model")
    return generator, client


def test_direct_answer_without_tools():
    generator, client = build_generator([make_text_response("Paris is the capital.")])

    answer = generator.generate_response(query="What is the capital of France?")

    assert answer == "Paris is the capital."
    assert client.messages.create.call_count == 1


def test_tools_and_tool_choice_added_to_params():
    generator, client = build_generator([make_text_response("hi")])
    tools = [{"name": "search_course_content"}]

    generator.generate_response(query="hello", tools=tools, tool_manager=MagicMock())

    params = client.messages.create.call_args.kwargs
    assert params["tools"] == tools
    assert params["tool_choice"] == {"type": "auto"}


def test_tool_use_triggers_execution_and_returns_final_text():
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "[MCP - Lesson 1]\nsome content"

    generator, client = build_generator([
        make_tool_use_response("search_course_content", {"query": "what is MCP"}),
        make_text_response("MCP is a protocol for connecting AI to tools."),
    ])

    answer = generator.generate_response(
        query="what is MCP",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    # The tool was executed with the exact name + input from the tool_use block.
    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="what is MCP"
    )
    # A second API call (post-tool) produced the final answer.
    assert client.messages.create.call_count == 2
    assert answer == "MCP is a protocol for connecting AI to tools."


def test_tool_result_message_structure():
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search output"

    generator, client = build_generator([
        make_tool_use_response("search_course_content", {"query": "x"}, tool_id="toolu_abc"),
        make_text_response("final"),
    ])

    generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    # Inspect the second (final) call's message history.
    second_call_messages = client.messages.create.call_args_list[1].kwargs["messages"]
    roles = [m["role"] for m in second_call_messages]
    assert roles == ["user", "assistant", "user"]

    tool_result_msg = second_call_messages[2]["content"][0]
    assert tool_result_msg["type"] == "tool_result"
    assert tool_result_msg["tool_use_id"] == "toolu_abc"
    assert tool_result_msg["content"] == "search output"


def test_api_error_returns_friendly_message_instead_of_raising():
    """An Anthropic API failure (e.g. low credit balance) must not bubble up as an
    unhandled exception -> HTTP 500 -> 'query failed'. It should return a clear message."""
    api_error = anthropic.APIError(
        "Your credit balance is too low to access the Anthropic API.",
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        body=None,
    )
    generator, client = build_generator([])
    client.messages.create.side_effect = api_error

    answer = generator.generate_response(query="What is MCP?")

    assert "AI service is currently unavailable" in answer
    assert "credit balance is too low" in answer


def test_empty_final_content_returns_fallback():
    """After a tool call, the model may return an empty content list (seen on
    'outline' queries). This must NOT raise IndexError -> HTTP 500 -> 'query failed';
    it should return a clear fallback string."""
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "some tool output"

    generator, client = build_generator([
        make_tool_use_response("get_course_outline", {"course_name": "MCP"}),
        make_empty_response(),  # final response: content == []
    ])

    answer = generator.generate_response(
        query="outline of the MCP course",
        tools=[{"name": "get_course_outline"}],
        tool_manager=tool_manager,
    )

    assert isinstance(answer, str) and answer.strip()
    assert "couldn't generate a response" in answer


def test_empty_direct_content_returns_fallback():
    """The direct (no-tool) path must also tolerate an empty content list."""
    generator, client = build_generator([make_empty_response()])

    answer = generator.generate_response(query="hello")

    assert isinstance(answer, str) and answer.strip()
    assert "couldn't generate a response" in answer


def test_extract_text_joins_multiple_text_blocks():
    """When several text blocks are present, all are returned (not just the first)."""
    multi = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="text", text="part one"),
            SimpleNamespace(type="text", text="part two"),
        ],
    )
    generator, client = build_generator([multi])

    answer = generator.generate_response(query="hi")

    assert answer == "part one\npart two"


def test_conversation_history_in_system_prompt():
    generator, client = build_generator([make_text_response("ok")])

    generator.generate_response(
        query="follow up question",
        conversation_history="User: hi\nAssistant: hello",
    )

    system = client.messages.create.call_args.kwargs["system"]
    assert "Previous conversation:" in system
    assert "User: hi" in system
