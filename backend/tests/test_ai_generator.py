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


# --- Sequential (multi-round) tool calling --------------------------------


def test_two_sequential_tool_rounds():
    """Round 1 outline -> round 2 search -> final synthesis. Claude chains two
    tools across separate API calls, reasoning about the first result."""
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = ["Lesson 4: Prompt caching", "search hits"]

    generator, client = build_generator([
        make_tool_use_response("get_course_outline", {"course_name": "MCP"}, tool_id="t1"),
        make_tool_use_response("search_course_content", {"query": "Prompt caching"}, tool_id="t2"),
        make_text_response("final"),
    ])

    answer = generator.generate_response(
        query="find a course like lesson 4 of MCP",
        tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert answer == "final"
    # Three API calls: round 1, round 2, synthesis.
    assert client.messages.create.call_count == 3
    # Both tools executed, in order, with their respective inputs.
    assert tool_manager.execute_tool.call_count == 2
    tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP")
    tool_manager.execute_tool.assert_any_call("search_course_content", query="Prompt caching")
    # History accumulates across rounds: by the synthesis call the messages hold
    # both rounds' tool_use turns and their results.
    synthesis_messages = client.messages.create.call_args_list[2].kwargs["messages"]
    assert [m["role"] for m in synthesis_messages] == [
        "user", "assistant", "user", "assistant", "user"
    ]


def test_stops_after_two_rounds_with_final_synthesis():
    """When Claude wants a tool in both rounds, the loop stops at the cap and the
    3rd (synthesis) call drops tools to force a text answer."""
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "tool output"

    generator, client = build_generator([
        make_tool_use_response("search_course_content", {"query": "a"}, tool_id="t1"),
        make_tool_use_response("search_course_content", {"query": "b"}, tool_id="t2"),
        make_text_response("synth"),
    ])

    answer = generator.generate_response(
        query="compare a and b",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert answer == "synth"
    assert tool_manager.execute_tool.call_count == 2
    third_call = client.messages.create.call_args_list[2].kwargs
    assert "tools" not in third_call


def test_second_round_includes_tools():
    """The core fix: tools are offered on BOTH the first and second round (old
    behavior removed tools on the second call)."""
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "tool output"
    tools = [{"name": "search_course_content"}]

    generator, client = build_generator([
        make_tool_use_response("search_course_content", {"query": "a"}),
        make_text_response("done"),
    ])

    generator.generate_response(query="q", tools=tools, tool_manager=tool_manager)

    first_call = client.messages.create.call_args_list[0].kwargs
    second_call = client.messages.create.call_args_list[1].kwargs
    assert first_call["tools"] == tools
    assert first_call["tool_choice"] == {"type": "auto"}
    assert second_call["tools"] == tools
    assert second_call["tool_choice"] == {"type": "auto"}


def test_first_round_direct_answer():
    """If round 1 returns no tool_use, answer immediately with a single API call."""
    tool_manager = MagicMock()

    generator, client = build_generator([make_text_response("direct")])

    answer = generator.generate_response(
        query="general knowledge",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert answer == "direct"
    assert client.messages.create.call_count == 1
    tool_manager.execute_tool.assert_not_called()


def test_tool_execution_error_handled_gracefully():
    """A tool raising must not propagate. The error is fed back as a tool_result and
    a tools-off synthesis call produces the final answer."""
    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = RuntimeError("db down")

    generator, client = build_generator([
        make_tool_use_response("search_course_content", {"query": "x"}, tool_id="t1"),
        make_text_response("recovered"),
    ])

    answer = generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert answer == "recovered"
    assert client.messages.create.call_count == 2
    # The synthesis call drops tools and carries the error tool_result.
    synthesis_call = client.messages.create.call_args_list[1].kwargs
    assert "tools" not in synthesis_call
    error_result = synthesis_call["messages"][-1]["content"][0]
    assert error_result["is_error"] is True
    assert "db down" in error_result["content"]


def test_empty_synthesis_content_returns_fallback():
    """An empty final response after two tool rounds yields the fallback, no IndexError."""
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "tool output"

    generator, client = build_generator([
        make_tool_use_response("search_course_content", {"query": "a"}, tool_id="t1"),
        make_tool_use_response("search_course_content", {"query": "b"}, tool_id="t2"),
        make_empty_response(),
    ])

    answer = generator.generate_response(
        query="q",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert isinstance(answer, str) and answer.strip()
    assert "couldn't generate a response" in answer


def test_api_error_during_second_round():
    """An APIError on a later round is still caught by the loop-wide handler."""
    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "tool output"

    api_error = anthropic.APIError(
        "Your credit balance is too low to access the Anthropic API.",
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        body=None,
    )
    generator, client = build_generator([
        make_tool_use_response("search_course_content", {"query": "x"}),
        api_error,
    ])

    answer = generator.generate_response(
        query="x",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert "AI service is currently unavailable" in answer
