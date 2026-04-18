from __future__ import annotations

"""
GDS AI Playoff Draft — Pick Validators

Handles fuzzy matching of player names and position validation.
Ensures every pick is valid or has a deterministic fallback.
"""

import difflib


def fuzzy_match_player(player_name: str, available_players: list[dict],
                       threshold: float = 0.6) -> dict | None:
    """
    Find the best matching player in the available pool.

    1. Try exact match (case-insensitive)
    2. Try fuzzy match with SequenceMatcher
    3. Return None if no match above threshold

    Returns the matched player dict or None.
    """
    name_lower = player_name.strip().lower()

    # Step 1: exact match (case-insensitive)
    for p in available_players:
        if p["name"].lower() == name_lower:
            return p

    # Step 2: fuzzy match
    best_match = None
    best_score = 0.0

    for p in available_players:
        score = difflib.SequenceMatcher(None, name_lower, p["name"].lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = p

    if best_score >= threshold and best_match is not None:
        return best_match

    return None


def auto_pick_fallback(team_state, available_players: list[dict],
                       roster_format: dict) -> dict:
    """
    When a model's pick is invalid after all parsing attempts,
    auto-select the best available player at the most-needed position.

    Priority logic:
    1. Determine which positions the team still needs
    2. Among needed positions, pick the highest-value available player
    3. If no position-specific need, pick the overall best available

    Returns the player dict.
    """
    needed = team_state.positions_needed(roster_format)

    if not needed:
        # Shouldn't happen, but just pick the best available overall
        return available_players[0] if available_players else None

    # Priority positions — pick the one with the most slots remaining
    # (e.g., if we need 4F and 1D, prioritize forward)
    priority_positions = sorted(needed.keys(), key=lambda p: needed[p], reverse=True)

    for pos in priority_positions:
        candidates = [p for p in available_players if p["position"] == pos]
        if candidates:
            # Already sorted by value (pts descending) from state loader
            return candidates[0]

    # Last resort: any available player
    return available_players[0] if available_players else None


def validate_position_for_team(position: str, team_state,
                                roster_format: dict) -> tuple[bool, str]:
    """
    Check if a team can draft at this position.
    Returns (is_valid, reason).
    """
    if position not in roster_format:
        return False, f"Unknown position '{position}'"

    if not team_state.can_draft_position(position, roster_format):
        have = team_state.roster_by_position()
        return False, (
            f"Already have {have[position]}/{roster_format[position]} "
            f"{position} — position full"
        )

    return True, "OK"
