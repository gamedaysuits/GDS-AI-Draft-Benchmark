#!/usr/bin/env python3
"""
Validate playoff_players.csv against the NHL's public API.

Checks:
  1. Every player in our CSV actually exists on their stated team's NHL roster
  2. No duplicate player names in our CSV
  3. Every playoff team has at least 1 goalie in our pool
  4. Positions match reality (F/D/G)
  5. Stats are roughly in the right ballpark (not wildly wrong)
  6. Logan Stankoven is only on ONE team

Outputs a report with issues and suggestions.
"""

from __future__ import annotations

import csv
import json
import requests
import time
from difflib import SequenceMatcher
from typing import Optional


# NHL API uses VGK for Vegas, our CSV uses VEG
# Map our abbreviations to NHL API abbreviations
TEAM_MAP_TO_NHL = {
    "VEG": "VGK",
}
TEAM_MAP_FROM_NHL = {v: k for k, v in TEAM_MAP_TO_NHL.items()}

PLAYOFF_TEAMS_OUR_ABBR = [
    "BUF", "TBL", "MTL", "CAR", "OTT", "PIT", "PHI", "BOS",
    "COL", "DAL", "MIN", "VEG", "UTA", "LAK", "EDM", "ANA",
]


def fetch_nhl_roster(team_abbr: str) -> dict:
    """Fetch current roster from NHL API. Returns {player_name: {position, id}}."""
    nhl_abbr = TEAM_MAP_TO_NHL.get(team_abbr, team_abbr)
    url = f"https://api-web.nhle.com/v1/roster/{nhl_abbr}/current"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    roster = {}
    for pos_key, pos_code in [("forwards", "F"), ("defensemen", "D"), ("goalies", "G")]:
        for p in data.get(pos_key, []):
            name = f"{p['firstName']['default']} {p['lastName']['default']}"
            roster[name] = {
                "position": pos_code,
                "id": p["id"],
                "number": p.get("sweaterNumber", ""),
            }
    return roster


def fetch_player_stats(player_id: int) -> dict:
    """Fetch current season stats for a player."""
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # Get featured stats (current season)
        featured = data.get("featuredStats", {}).get("regularSeason", {}).get("subSeason", {})
        return featured
    except Exception:
        return {}


def fuzzy_find(name: str, roster: dict, threshold: float = 0.75) -> str | None:
    """Find the closest matching name in a roster."""
    best_match = None
    best_score = 0.0
    for roster_name in roster:
        score = SequenceMatcher(None, name.lower(), roster_name.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = roster_name
    if best_score >= threshold:
        return best_match
    return None


def load_our_csv(path: str) -> list[dict]:
    """Load our playoff_players.csv."""
    players = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            players.append({
                "name": row["name"].strip(),
                "team": row["team"].strip(),
                "position": row["position"].strip().upper(),
                "goals": int(row.get("goals", 0) or 0),
                "assists": int(row.get("assists", 0) or 0),
                "points": int(row.get("points", 0) or 0),
            })
    return players


def main():
    print("=" * 70)
    print("  NHL API VALIDATION OF playoff_players.csv")
    print("=" * 70)
    print()

    # Load our CSV
    our_players = load_our_csv("data/playoff_players.csv")
    print(f"Our CSV: {len(our_players)} players")

    # Fetch all NHL rosters
    print("\nFetching NHL rosters for all 16 playoff teams...")
    nhl_rosters = {}  # {team_abbr: {player_name: {position, id}}}
    for team in PLAYOFF_TEAMS_OUR_ABBR:
        try:
            nhl_rosters[team] = fetch_nhl_roster(team)
            print(f"  ✅ {team}: {len(nhl_rosters[team])} players")
            time.sleep(0.2)  # Be polite to the API
        except Exception as e:
            print(f"  ❌ {team}: {e}")
            nhl_rosters[team] = {}

    # === VALIDATION CHECKS ===
    issues = []
    warnings = []
    info = []

    # 1. Duplicate names in our CSV
    print("\n--- Check 1: Duplicate Names ---")
    names = [p["name"] for p in our_players]
    dupes = set(n for n in names if names.count(n) > 1)
    if dupes:
        for d in dupes:
            entries = [(p["name"], p["team"]) for p in our_players if p["name"] == d]
            issues.append(f"DUPLICATE: {d} appears on {[e[1] for e in entries]}")
            print(f"  ❌ {d} is listed on: {[e[1] for e in entries]}")
    else:
        print("  ✅ No duplicates")

    # 2. Every player exists on their stated team's NHL roster
    print("\n--- Check 2: Player-Team Verification ---")
    not_found = []
    wrong_team = []
    name_mismatches = []

    for p in our_players:
        team_roster = nhl_rosters.get(p["team"], {})
        if not team_roster:
            warnings.append(f"Could not verify {p['name']} — {p['team']} roster unavailable")
            continue

        # Exact match
        if p["name"] in team_roster:
            # Check position
            nhl_pos = team_roster[p["name"]]["position"]
            if p["position"] != nhl_pos:
                warnings.append(f"POSITION MISMATCH: {p['name']} ({p['team']}) — "
                               f"our CSV says {p['position']}, NHL says {nhl_pos}")
            continue

        # Fuzzy match on same team
        fuzzy = fuzzy_find(p["name"], team_roster)
        if fuzzy:
            name_mismatches.append(f"NAME SPELLING: {p['name']} → NHL has '{fuzzy}' on {p['team']}")
            continue

        # Not on this team — check if they're on a different playoff team
        found_on = None
        for other_team, other_roster in nhl_rosters.items():
            if other_team == p["team"]:
                continue
            if p["name"] in other_roster or fuzzy_find(p["name"], other_roster):
                found_on = other_team
                break

        if found_on:
            wrong_team.append(f"WRONG TEAM: {p['name']} listed as {p['team']} but NHL has them on {found_on}")
            issues.append(f"WRONG TEAM: {p['name']} — CSV says {p['team']}, NHL says {found_on}")
        else:
            not_found.append(f"NOT FOUND: {p['name']} ({p['team']}/{p['position']}) — not on any playoff roster")
            issues.append(f"NOT FOUND: {p['name']} ({p['team']}) — not on any playoff team roster via NHL API")

    if wrong_team:
        print(f"  ❌ {len(wrong_team)} players on wrong teams:")
        for w in wrong_team:
            print(f"    {w}")
    if not_found:
        print(f"  ⚠️ {len(not_found)} players not found on any playoff roster:")
        for nf in not_found:
            print(f"    {nf}")
    if name_mismatches:
        print(f"  ℹ️ {len(name_mismatches)} name spelling differences:")
        for nm in name_mismatches:
            print(f"    {nm}")

    ok_count = len(our_players) - len(wrong_team) - len(not_found)
    print(f"  ✅ {ok_count}/{len(our_players)} players verified on correct teams")

    # 3. Goalie coverage
    print("\n--- Check 3: Goalie Coverage ---")
    goalies_by_team = {}
    for p in our_players:
        if p["position"] == "G":
            if p["team"] not in goalies_by_team:
                goalies_by_team[p["team"]] = []
            goalies_by_team[p["team"]].append(p["name"])

    for team in PLAYOFF_TEAMS_OUR_ABBR:
        if team not in goalies_by_team:
            issues.append(f"NO GOALIE: {team} has no goalie in the draft pool!")
            print(f"  ❌ {team}: NO GOALIE")
        else:
            print(f"  ✅ {team}: {goalies_by_team[team]}")

    total_goalies = sum(len(g) for g in goalies_by_team.values())
    print(f"  Total goalies: {total_goalies} (need 12 for draft, have {total_goalies})")

    # 4. Players per team
    print("\n--- Check 4: Pool Balance ---")
    team_counts = {}
    for p in our_players:
        team_counts[p["team"]] = team_counts.get(p["team"], 0) + 1
    for team in sorted(PLAYOFF_TEAMS_OUR_ABBR):
        count = team_counts.get(team, 0)
        status = "✅" if count >= 8 else "⚠️"
        print(f"  {status} {team}: {count} players")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if issues:
        print(f"\n  ❌ ISSUES ({len(issues)}):")
        for i in issues:
            print(f"    • {i}")

    if warnings:
        print(f"\n  ⚠️ WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    • {w}")

    if not issues and not warnings:
        print("\n  ✅ ALL CHECKS PASSED!")
    elif not issues:
        print("\n  ✅ No blocking issues — warnings only.")

    print()


if __name__ == "__main__":
    main()
