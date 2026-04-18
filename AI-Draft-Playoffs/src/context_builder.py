"""
GDS AI Playoff Draft — Context Builder

Assembles prompts within token budgets using modular components.
Each component has a hard cap — truncate if exceeded.
"""

from . import prompts


def truncate(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximate token limit (1 token ≈ 4 chars).
    Slightly over-counts for safety.
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def build_identity_prompt(team_name: str, draft_state, config: dict) -> list[dict]:
    """
    Build the Phase 1 identity + scouting prompt for a team.
    Returns messages list for the API call.
    """
    team = draft_state.teams[team_name]

    system_content = prompts.IDENTITY_SYSTEM.format(
        model_name=team.name,
        provider=team.provider,
        backstory=team.backstory,
    )

    user_content = prompts.IDENTITY_USER.format(
        player_pool_summary=truncate(
            draft_state.get_player_pool_summary(),
            800  # More generous for identity phase
        ),
        competitor_list=truncate(
            draft_state.get_competitor_list(exclude_team=team_name),
            400
        ),
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_identity_followup(tool_results: str) -> list[dict]:
    """
    Build the follow-up message after tool calls in Phase 1.
    Returns a single user message to append.
    """
    content = prompts.IDENTITY_FOLLOWUP.format(
        tool_results=truncate(tool_results, 5000),
    )
    return [{"role": "user", "content": content}]


def build_pick_prompt(team_name: str, draft_state, round_num: int,
                      pick_overall: int, config: dict) -> list[dict]:
    """
    Build the Phase 2 draft pick prompt for a team.
    Each component is truncated to its token budget.
    Returns messages list for the API call.
    """
    team = draft_state.teams[team_name]
    budgets = config["token_budgets"]
    roster_format = config["draft"]["roster_format"]

    # System prompt with persona
    system_content = prompts.PICK_SYSTEM.format(
        nickname=team.display_name,
        persona=team.persona,
    )

    # Get available players
    avail_f, avail_d, avail_g = draft_state.get_available_display()

    # Get snake order for current round
    round_order = draft_state.get_snake_order(round_num)
    round_order_display = " → ".join(
        f"**{draft_state.teams[t].display_name}**" if t == team_name
        else draft_state.teams[t].display_name
        for t in round_order
    )

    # Positions needed
    needed = team.positions_needed(roster_format)
    needs_display = ", ".join(f"{v}{k}" for k, v in needed.items()) if needed else "Roster complete!"

    # Build explicit position constraint text
    # This makes it crystal clear which positions the model can/can't pick
    filled = team.roster_by_position()
    constraint_lines = []
    for pos in ("F", "D", "G"):
        max_count = roster_format.get(pos, 0)
        have = filled.get(pos, 0)
        if have >= max_count:
            constraint_lines.append(f"  ✘ {pos} — FULL ({have}/{max_count}) — DO NOT pick a {pos}")
        else:
            remaining = max_count - have
            constraint_lines.append(f"  ✔ {pos} — OPEN ({have}/{max_count}) — need {remaining} more")
    position_constraint = "\n".join(constraint_lines)

    # Build user message
    user_content = prompts.PICK_USER.format(
        strategy_doc=truncate(team.strategy_doc, budgets["strategy_doc"]),
        scouting_notes=truncate(team.scouting_notes, budgets["scouting_notes"]),
        scratchpad=truncate(team.scratchpad or "(empty — use update_scratchpad to add notes)", budgets["scratchpad"]),
        round_num=round_num,
        total_rounds=draft_state.total_rounds,
        pick_overall=pick_overall,
        team_pick_num=len(team.roster) + 1,
        round_order=round_order_display,
        picks_made=len(team.roster),
        roster_display=team.roster_display(),
        positions_needed=needs_display,
        position_constraint=position_constraint,
        all_rosters_compressed=truncate(
            draft_state.get_all_rosters_compressed(),
            budgets["draft_state"]
        ),
        available_forwards=truncate(avail_f, budgets["available_players"] // 2),
        available_defense=truncate(avail_d, budgets["available_players"] // 4),
        available_goalies=truncate(avail_g, budgets["available_players"] // 4),
        recent_picks=truncate(
            draft_state.get_recent_picks(count=12),
            budgets["recent_activity"]
        ),
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_pick_tool_followup(tool_name: str, tool_result: str,
                              config: dict) -> list[dict]:
    """Build follow-up after a tool call during a pick turn."""
    content = prompts.PICK_TOOL_RESULT.format(
        tool_name=tool_name,
        tool_result=truncate(tool_result, config["token_budgets"]["research_results"]),
    )
    return [{"role": "user", "content": content}]


def build_chirp_prompt(reactor_name: str, drafter_name: str,
                       player_name: str, player_team: str,
                       position: str, round_num: int,
                       pick_overall: int, drafter_chirp: str,
                       draft_state, config: dict) -> list[dict]:
    """
    Build the chirp/reaction prompt for a non-drafting GM.
    Intentionally lightweight — minimal context for fast responses.
    """
    reactor = draft_state.teams[reactor_name]
    drafter = draft_state.teams[drafter_name]

    system_content = prompts.CHIRP_SYSTEM.format(
        nickname=reactor.display_name,
        persona=reactor.persona,
    )

    user_content = prompts.CHIRP_USER.format(
        drafter_nickname=drafter.display_name,
        player_name=player_name,
        player_team=player_team,
        position=position,
        round_num=round_num,
        pick_overall=pick_overall,
        drafter_chirp=drafter_chirp,
        max_chars=config["reactions"]["max_chars"],
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_closing_prompt(team_name: str, draft_state, config: dict) -> list[dict]:
    """Build the Phase 3 closing statement prompt."""
    team = draft_state.teams[team_name]

    system_content = prompts.CLOSING_SYSTEM.format(
        nickname=team.display_name,
        persona=team.persona,
    )

    user_content = prompts.CLOSING_USER.format(
        all_final_rosters=truncate(
            draft_state.get_all_final_rosters(), 1500
        ),
        own_roster_detailed=team.roster_display(),
        tiebreaker=team.tiebreaker,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
