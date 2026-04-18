"""
GDS AI Playoff Draft — Draft State Tracker

Manages all mutable state: teams, rosters, available players,
pick history, and draft order. Single source of truth.
"""

import csv
import random
from dataclasses import dataclass, field


@dataclass
class TeamState:
    """All state for one draft team."""
    name: str                      # Display name ("Grok", "Claude", etc.)
    slug: str                      # OpenRouter model ID
    provider: str                  # Provider name for display
    backstory: str                 # One-liner for identity prompt

    # Set during Phase 1 (Identity)
    nickname: str = ""
    persona: str = ""
    voice_description: str = ""
    strategy_doc: str = ""
    scouting_notes: str = ""
    scratchpad: str = ""           # Editable during Phase 2
    tiebreaker: int = 0

    # Built during Phase 2 (Draft)
    roster: list = field(default_factory=list)  # [{name, team, position, stats}]

    # Closing statement from Phase 3
    closing_statement: str = ""

    @property
    def display_name(self) -> str:
        """Return nickname if set, else team name."""
        return self.nickname if self.nickname else self.name

    def roster_by_position(self) -> dict:
        """Count picks by position."""
        counts = {"F": 0, "D": 0, "G": 0}
        for player in self.roster:
            pos = player.get("position", "F")
            if pos in counts:
                counts[pos] += 1
        return counts

    def positions_needed(self, roster_format: dict) -> dict:
        """How many of each position still needed."""
        have = self.roster_by_position()
        return {
            pos: roster_format[pos] - have.get(pos, 0)
            for pos in roster_format
            if roster_format[pos] - have.get(pos, 0) > 0
        }

    def can_draft_position(self, position: str, roster_format: dict) -> bool:
        """Check if team can still draft at this position."""
        have = self.roster_by_position()
        return have.get(position, 0) < roster_format.get(position, 0)

    def roster_display(self) -> str:
        """Formatted roster string for prompt injection."""
        if not self.roster:
            return "(empty)"
        lines = []
        for p in self.roster:
            lines.append(f"  {p['name']} ({p['team']}/{p['position']}) — {p.get('pts_display', '')}")
        return "\n".join(lines)


class DraftState:
    """
    Central state manager for the entire draft.
    Holds teams, available players, pick history, and draft order.
    """

    def __init__(self, config: dict):
        self.config = config
        self.roster_format = config["draft"]["roster_format"]
        self.total_rounds = config["draft"]["rounds"]

        # Initialize teams from config
        self.teams: dict[str, TeamState] = {}
        for name, model_cfg in config["models"].items():
            self.teams[name] = TeamState(
                name=name,
                slug=model_cfg["slug"],
                provider=model_cfg["provider"],
                backstory=model_cfg["backstory"],
            )

        # Player pool — loaded from CSV
        self.all_players: list[dict] = []
        self.available_players: list[dict] = []

        # Pick history — ordered list of all picks
        self.pick_history: list[dict] = []

        # Draft order — set during initialization
        self.base_order: list[str] = []

        # Current position tracking
        self.current_round: int = 0
        self.current_pick_in_round: int = 0

    def load_players(self, csv_path: str):
        """Load player pool from CSV file."""
        self.all_players = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize CSV keys to lowercase for case-insensitive access
                r = {k.lower().strip(): v for k, v in row.items()}

                # Normalize position to F/D/G
                pos = r.get("position", "F").strip().upper()
                if pos in ("C", "LW", "RW", "F"):
                    pos = "F"
                elif pos in ("D", "LD", "RD"):
                    pos = "D"
                elif pos in ("G", "GK"):
                    pos = "G"

                player = {
                    "name": r.get("name", "").strip(),
                    "team": r.get("team", "").strip(),
                    "position": pos,
                }

                # Skater stats
                if pos in ("F", "D"):
                    player["g"] = int(r.get("goals", r.get("g", 0)) or 0)
                    player["a"] = int(r.get("assists", r.get("a", 0)) or 0)
                    player["pts"] = int(r.get("points", r.get("pts", 0)) or 0)
                    # Use pre-built display string if available, else generate
                    if r.get("pts_display"):
                        player["pts_display"] = r["pts_display"].strip()
                    else:
                        player["pts_display"] = f"{player['pts']} PTS ({player['g']}G {player['a']}A)"
                # Goalie stats
                elif pos == "G":
                    player["w"] = int(r.get("w", r.get("goals", 0)) or 0)
                    player["so"] = int(r.get("so", 0) or 0)
                    player["sv_pct"] = r.get("sv%", r.get("sv_pct", "")).strip()
                    player["gaa"] = r.get("gaa", "").strip()
                    # For draft value display — wins as proxy
                    player["pts"] = player["w"]
                    # Use pre-built display string if available
                    if r.get("pts_display"):
                        player["pts_display"] = r["pts_display"].strip()
                    else:
                        player["pts_display"] = f"{player['w']}W (SV% {player['sv_pct']})"

                self.all_players.append(player)

        # Sort by points descending — best players first
        self.all_players.sort(key=lambda p: p.get("pts", 0), reverse=True)
        # Copy to available pool
        self.available_players = list(self.all_players)
        print(f"  Loaded {len(self.all_players)} players from {csv_path}")

    def set_draft_order(self, defending_champion: str):
        """
        Set draft order: defending champion picks 1st, rest randomized.
        """
        other_teams = [name for name in self.teams if name != defending_champion]
        random.shuffle(other_teams)
        self.base_order = [defending_champion] + other_teams
        print(f"  Draft order: {' → '.join(self.base_order)}")

    def get_snake_order(self, round_num: int) -> list[str]:
        """
        Snake draft: odd rounds = normal order, even rounds = reversed.
        Round numbers are 1-indexed.
        """
        if round_num % 2 == 0:
            return list(reversed(self.base_order))
        return list(self.base_order)

    def make_pick(self, team_name: str, player_name: str, round_num: int,
                  pick_overall: int) -> dict:
        """
        Record a draft pick. Removes player from available pool.
        Returns the pick record dict.
        """
        # Find the player in available pool
        player = None
        for p in self.available_players:
            if p["name"].lower() == player_name.lower():
                player = p
                break

        if player is None:
            raise ValueError(f"Player '{player_name}' not found in available pool")

        # Remove from available
        self.available_players.remove(player)

        # Add to team roster
        self.teams[team_name].roster.append(player)

        # Record in history
        pick_record = {
            "round": round_num,
            "pick_overall": pick_overall,
            "team": team_name,
            "nickname": self.teams[team_name].display_name,
            "player": player["name"],
            "player_team": player["team"],
            "position": player["position"],
            "stats": player.get("pts_display", ""),
            "chirp": "",        # Filled by orchestrator
            "reactions": [],    # Filled by orchestrator
        }
        self.pick_history.append(pick_record)
        return pick_record

    def get_available_by_position(self, position: str, limit: int = 15) -> list[dict]:
        """Get top available players at a specific position."""
        filtered = [p for p in self.available_players if p["position"] == position]
        return filtered[:limit]

    def get_available_display(self, f_limit: int = 40, d_limit: int = 20,
                               g_limit: int = 10) -> tuple[str, str, str]:
        """
        Format available players for prompt injection.
        Returns (forwards_str, defense_str, goalies_str).
        """
        def format_list(players: list[dict]) -> str:
            if not players:
                return "(none remaining)"
            lines = []
            for p in players:
                lines.append(f"  {p['name']} ({p['team']}) — {p.get('pts_display', '?')}")
            return "\n".join(lines)

        forwards = self.get_available_by_position("F", f_limit)
        defense = self.get_available_by_position("D", d_limit)
        goalies = self.get_available_by_position("G", g_limit)

        return format_list(forwards), format_list(defense), format_list(goalies)

    def get_all_rosters_compressed(self) -> str:
        """Compressed view of all teams' rosters for prompt injection."""
        lines = []
        for name, team in self.teams.items():
            if team.roster:
                players = ", ".join(p["name"] for p in team.roster)
                lines.append(f"  {team.display_name}: {players}")
            else:
                lines.append(f"  {team.display_name}: (no picks yet)")
        return "\n".join(lines)

    def get_recent_picks(self, count: int = 5) -> str:
        """Last N picks formatted for prompt injection."""
        recent = self.pick_history[-count:] if self.pick_history else []
        if not recent:
            return "(draft just started — no picks yet)"
        lines = []
        for pick in recent:
            line = f"  Pick #{pick['pick_overall']}: {pick['nickname']} took " \
                   f"{pick['player']} ({pick['player_team']}/{pick['position']})"
            if pick.get("chirp"):
                line += f' — "{pick["chirp"]}"'
            # Include reactions if any
            for rxn in pick.get("reactions", []):
                line += f'\n    → {rxn["nickname"]}: "{rxn["chirp"]}"'
            lines.append(line)
        return "\n".join(lines)

    def get_all_final_rosters(self) -> str:
        """Detailed final rosters for closing statements."""
        lines = []
        for name, team in self.teams.items():
            lines.append(f"\n  {team.display_name} ({name}):")
            if team.roster:
                for p in team.roster:
                    lines.append(f"    {p['position']}: {p['name']} ({p['team']}) — {p.get('pts_display', '')}")
            else:
                lines.append("    (no roster)")
        return "\n".join(lines)

    def get_competitor_list(self, exclude_team: str = "") -> str:
        """List of competing models for identity prompt."""
        lines = []
        for name, team in self.teams.items():
            if name == exclude_team:
                continue
            lines.append(f"  • {name} ({team.provider}) — {team.backstory}")
        return "\n".join(lines)

    def get_player_pool_summary(self) -> str:
        """Summary of player pool by team for identity prompt."""
        # Group by team
        by_team = {}
        for p in self.all_players:
            t = p["team"]
            if t not in by_team:
                by_team[t] = []
            by_team[t].append(p)

        lines = []
        for team_name, players in sorted(by_team.items()):
            top3 = players[:3]
            names = ", ".join(f"{p['name']} ({p['pts_display']})" for p in top3)
            lines.append(f"  {team_name}: {names}, +{len(players)-3} more")
        return "\n".join(lines)
