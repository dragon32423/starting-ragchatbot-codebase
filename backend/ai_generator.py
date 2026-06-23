import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool-executing rounds per user query.
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Tool Usage:
- **search_course_content**: use for questions about specific course content or detailed educational materials.
- **get_course_outline**: use for questions about a course's outline, syllabus, or list of lessons. Report the course title, course link, and the full numbered lesson list.
- You may use tools across up to two sequential rounds for a single query. Use the result of an earlier tool call to decide whether a second is needed (e.g. look up a lesson's title with get_course_outline, then search_course_content for that title).
- Use a second tool call only when the first result is insufficient to answer. If the first result already answers the question, respond directly.
- Synthesize all tool results into one accurate, fact-based response.
- If a tool yields no results, state this clearly without offering alternatives.

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: str | None = None,
        tools: list | None = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Conversation messages accumulate across tool rounds.
        messages = [{"role": "user", "content": query}]

        try:
            # Fast path: with no tools to call, a single direct call answers the query.
            if not tools or not tool_manager:
                response = self.client.messages.create(
                    **self.base_params, messages=messages, system=system_content
                )
                return self._extract_text(response)

            # Up to MAX_TOOL_ROUNDS tool-executing rounds, each a separate API call
            # with tools attached so Claude can reason about prior results and chain
            # a second, result-informed tool call.
            for _ in range(self.MAX_TOOL_ROUNDS):
                response = self.client.messages.create(
                    **self.base_params,
                    messages=messages,
                    system=system_content,
                    tools=tools,
                    tool_choice={"type": "auto"},
                )

                # Terminate (b): no tool requested -> this is the final answer.
                if response.stop_reason != "tool_use":
                    return self._extract_text(response)

                # Record the assistant's tool_use turn, then execute the tools.
                messages.append({"role": "assistant", "content": response.content})
                tool_results, had_error = self._execute_tools(response, tool_manager)
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

                # Terminate (c): a tool failed -> synthesize an answer from the error.
                if had_error:
                    return self._final_synthesis(messages, system_content)

            # Terminate (a): both rounds used a tool -> one tools-off call to answer.
            return self._final_synthesis(messages, system_content)
        except anthropic.APIError as e:
            # Surface the real reason (auth, billing, model, rate limit) instead of
            # letting it bubble up as an opaque HTTP 500 / "query failed".
            return f"The AI service is currently unavailable: {e}"

    def _execute_tools(self, response, tool_manager):
        """
        Execute every tool_use block in a response and collect tool_result blocks.

        Args:
            response: A response whose content contains tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            (tool_results, had_error): the list of tool_result blocks (all for one
            round, returned in a single user message), and whether any tool raised.
        """
        tool_results = []
        had_error = False
        for content_block in response.content:
            if content_block.type != "tool_use":
                continue
            try:
                result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result,
                    }
                )
            except Exception as e:
                # A hard tool failure: feed the error back so Claude can phrase a
                # graceful answer, and flag it so the loop stops issuing more tools.
                had_error = True
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool '{content_block.name}' failed: {e}",
                        "is_error": True,
                    }
                )
        return tool_results, had_error

    def _final_synthesis(self, messages, system_content):
        """Make a final tools-off call so Claude produces a text answer."""
        response = self.client.messages.create(
            **self.base_params, messages=messages, system=system_content
        )
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """Safely pull the assistant's text out of a Messages response.

        The API can return a response whose ``content`` is empty or contains only
        non-text blocks (e.g. an ``end_turn`` with no text after a tool result).
        Indexing ``content[0].text`` blindly raises IndexError in that case, which
        previously surfaced as an opaque HTTP 500 / "query failed".
        """
        texts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        if texts:
            return "\n".join(texts)
        return (
            "I couldn't generate a response for that. "
            "Please try rephrasing your question."
        )
