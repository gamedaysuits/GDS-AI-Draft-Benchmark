"""
GDS AI Playoff Draft — Tool Implementations

Tools that the orchestrator calls on behalf of models:
- web_search: Search the internet via Perplexity Sonar
- self_research: Search about the model itself
- update_scratchpad: Update the model's private notes
"""


def web_search(query: str, api_client, config: dict) -> str:
    """
    Search the internet using Perplexity Sonar (cheap, clean results).

    This is the search backend for ALL models — separate from
    Perplexity the competitor (who uses sonar-pro for drafting).
    """
    try:
        result = api_client.call_model(
            model=config["tools"]["search_model"],
            messages=[{"role": "user", "content": query}],
            max_tokens=config["tools"]["search_max_tokens"],
            temperature=0.3,
        )
        return result["content"]
    except Exception as e:
        return f"(Search unavailable: {str(e)[:100]})"


def self_research(query: str, model_name: str, provider: str,
                  api_client, config: dict) -> str:
    """
    Search for info about the model itself.
    Prepends the model's identity to the search query.
    """
    enriched_query = f"{model_name} AI model by {provider}: {query}"
    return web_search(enriched_query, api_client, config)


def update_scratchpad(content: str, team_state, config: dict) -> str:
    """
    Update a team's private scratchpad/notes.
    Truncates to the configured max token limit.
    """
    max_chars = config["tools"]["scratchpad_max_tokens"] * 4  # ~4 chars per token
    if len(content) > max_chars:
        content = content[:max_chars] + "... [truncated]"
    team_state.scratchpad = content
    return "Scratchpad updated successfully."


def execute_tool(tool_name: str, tool_args: dict, team_name: str,
                 draft_state, api_client, config: dict) -> str:
    """
    Execute a tool call and return the result string.

    Args:
        tool_name: "web_search", "self_research", or "update_scratchpad"
        tool_args: dict of tool arguments
        team_name: which team is calling
        draft_state: the DraftState instance
        api_client: the APIClient instance
        config: full config dict

    Returns:
        Tool result as a string
    """
    team = draft_state.teams[team_name]

    if tool_name == "web_search":
        query = tool_args.get("query", "")
        if not query:
            return "(No query provided)"
        return web_search(query, api_client, config)

    elif tool_name == "self_research":
        query = tool_args.get("query", "")
        if not query:
            return "(No query provided)"
        return self_research(query, team.name, team.provider, api_client, config)

    elif tool_name == "update_scratchpad":
        content = tool_args.get("content", "")
        if not content:
            return "(No content provided)"
        return update_scratchpad(content, team, config)

    else:
        return f"(Unknown tool: {tool_name})"
