"""
GDS AI Playoff Draft — Logger

Dual output: JSON log (structured data) + Markdown log (blog-ready).
Records every event with full metadata.
"""

import csv
import json
import os
from datetime import datetime, timezone


class DraftLogger:
    """
    Logs all draft events to both JSON and Markdown formats.

    JSON log: full structured data for analysis and voice rendering.
    Markdown log: human-readable, blog-formatted draft narrative.
    """

    def __init__(self, config: dict, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Paths
        self.json_path = os.path.join(output_dir, config["output"]["json_log"])
        self.md_path = os.path.join(output_dir, config["output"]["markdown_log"])
        self.personas_path = os.path.join(output_dir, config["output"]["personas_file"])
        self.csv_path = os.path.join(output_dir, config["output"]["results_csv"])

        # In-memory log structure
        self.log = {
            "draft_id": f"playoff_draft_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_summary": {
                "rounds": config["draft"]["rounds"],
                "roster_format": config["draft"]["roster_format"],
                "scoring": config["draft"]["scoring"],
                "models": {
                    name: cfg["slug"]
                    for name, cfg in config["models"].items()
                },
            },
            "draft_order": [],
            "phases": {
                "identity": [],
                "draft": [],
                "closing": [],
            },
            "final_rosters": {},
            "usage": {},
        }

        # Start the markdown log
        self._md_lines = []
        self._md_lines.append("# 🏒 GDS AI Playoff Draft 2026\n")
        self._md_lines.append(
            f"*Draft started: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*\n"
        )
        self._md_lines.append("---\n")

    # ── Phase 0: Draft Order ────────────────────────────────────

    def log_draft_order(self, order: list[str], teams: dict):
        """Record the randomized draft order."""
        self.log["draft_order"] = order

        self._md_lines.append("## Draft Order\n")
        for i, name in enumerate(order, 1):
            team = teams[name]
            self._md_lines.append(
                f"{i}. **{team.display_name}** ({name} — {team.provider})"
            )
        self._md_lines.append("")

    # ── Phase 1: Identity ───────────────────────────────────────

    def log_persona(self, team_name: str, persona_data: dict, usage: dict,
                    latency: float, tool_calls: list = None):
        """Record Phase 1 persona creation."""
        entry = {
            "team": team_name,
            "persona": persona_data,
            "tool_calls": tool_calls or [],
            "usage": usage,
            "latency": latency,
        }
        self.log["phases"]["identity"].append(entry)

    def log_personas_intro(self, teams: dict):
        """Write the personas section to markdown."""
        self._md_lines.append("\n## Meet the GMs\n")
        for name, team in teams.items():
            if team.nickname:
                self._md_lines.append(f"### {team.display_name} ({name})\n")
                self._md_lines.append(f"*\"{team.persona}\"*\n")
                if team.voice_description:
                    self._md_lines.append(f"🎙️ Voice: {team.voice_description}\n")
                self._md_lines.append("")
        self._md_lines.append("---\n")

    # ── Phase 2: Draft ──────────────────────────────────────────

    def log_round_start(self, round_num: int, order: list[str], teams: dict):
        """Write round header to markdown."""
        order_display = " → ".join(teams[t].display_name for t in order)
        self._md_lines.append(f"\n## Round {round_num}\n")
        self._md_lines.append(f"*Order: {order_display}*\n")

    def log_pick(self, pick_data: dict, team_display: str, usage: dict,
                 latency: float, tool_calls: list = None, was_auto: bool = False):
        """Record a draft pick to both logs."""
        entry = {
            **pick_data,
            "usage": usage,
            "latency": latency,
            "tool_calls": tool_calls or [],
            "was_auto_pick": was_auto,
        }
        self.log["phases"]["draft"].append(entry)

        # Markdown
        auto_tag = " *(auto-pick)*" if was_auto else ""
        self._md_lines.append(
            f"### Pick {pick_data['pick_overall']} — "
            f"{team_display}{auto_tag}\n"
        )
        self._md_lines.append(
            f"> **{pick_data['player']}** ({pick_data['player_team']} / "
            f"{pick_data['position']}) — {pick_data.get('stats', '')}"
        )
        if pick_data.get("chirp"):
            self._md_lines.append(f">\n> *\"{pick_data['chirp']}\"*")
        self._md_lines.append("")

        # Reactions
        if pick_data.get("reactions"):
            self._md_lines.append("**Reactions:**")
            for rxn in pick_data["reactions"]:
                self._md_lines.append(
                    f"> **{rxn['nickname']}:** \"{rxn['chirp']}\""
                )
            self._md_lines.append("")

    # ── Phase 3: Closing ────────────────────────────────────────

    def log_closing(self, team_name: str, team_display: str,
                    statement: str, usage: dict, latency: float):
        """Record a closing statement."""
        entry = {
            "team": team_name,
            "nickname": team_display,
            "closing_statement": statement,
            "usage": usage,
            "latency": latency,
        }
        self.log["phases"]["closing"].append(entry)

    def log_closings_section(self, teams: dict):
        """Write all closing statements to markdown."""
        self._md_lines.append("\n---\n")
        self._md_lines.append("## Closing Statements\n")
        for name, team in teams.items():
            if team.closing_statement:
                self._md_lines.append(f"### {team.display_name} ({name})\n")
                self._md_lines.append(f"> {team.closing_statement}\n")
                self._md_lines.append("")

    # ── Final Output ────────────────────────────────────────────

    def log_final_rosters(self, teams: dict):
        """Record final rosters to both logs."""
        self._md_lines.append("\n---\n")
        self._md_lines.append("## Final Rosters\n")

        for name, team in teams.items():
            roster_data = []
            self._md_lines.append(f"### {team.display_name} ({name})\n")
            self._md_lines.append("| Pos | Player | Team | Reg Season |")
            self._md_lines.append("|:----|:-------|:-----|:-----------|")

            for p in team.roster:
                self._md_lines.append(
                    f"| {p['position']} | {p['name']} | {p['team']} | "
                    f"{p.get('pts_display', '')} |"
                )
                roster_data.append({
                    "name": p["name"],
                    "team": p["team"],
                    "position": p["position"],
                })

            self._md_lines.append(
                f"\n*Tiebreaker prediction: {team.tiebreaker} total playoff goals*\n"
            )
            self.log["final_rosters"][name] = roster_data

    def save(self, api_client=None):
        """Write all logs to disk."""
        # Add usage summary if available
        if api_client:
            self.log["usage"] = api_client.get_usage_summary()

            self._md_lines.append("\n---\n")
            usage = api_client.get_usage_summary()
            self._md_lines.append("## Draft Stats\n")
            self._md_lines.append(f"- **Total API calls**: {usage['total_calls']}")
            self._md_lines.append(f"- **Input tokens**: {usage['total_input_tokens']:,}")
            self._md_lines.append(f"- **Output tokens**: {usage['total_output_tokens']:,}")
            if usage["total_reasoning_tokens"]:
                self._md_lines.append(
                    f"- **Reasoning tokens**: {usage['total_reasoning_tokens']:,}"
                )
            self._md_lines.append("")

            # Per-model breakdown
            if usage.get("per_model"):
                self._md_lines.append("### Per-Model Token Usage\n")
                self._md_lines.append("| Model | Calls | Input | Output | Reasoning |")
                self._md_lines.append("|:------|------:|------:|-------:|----------:|")
                for model_slug, m in sorted(usage["per_model"].items()):
                    reasoning = f"{m['reasoning_tokens']:,}" if m['reasoning_tokens'] else "—"
                    self._md_lines.append(
                        f"| {model_slug} | {m['calls']} | "
                        f"{m['input_tokens']:,} | {m['output_tokens']:,} | {reasoning} |"
                    )
                self._md_lines.append("")

        # Write JSON log
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2, ensure_ascii=False)
        print(f"  📄 JSON log saved: {self.json_path}")

        # Write Markdown log
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._md_lines))
        print(f"  📝 Markdown log saved: {self.md_path}")

        # Write personas JSON (for voice rendering later)
        personas = {}
        for entry in self.log["phases"]["identity"]:
            personas[entry["team"]] = entry.get("persona", {})
        with open(self.personas_path, "w", encoding="utf-8") as f:
            json.dump(personas, f, indent=2, ensure_ascii=False)
        print(f"  🎭 Personas saved: {self.personas_path}")

    def save_results_csv(self, teams: dict):
        """Write draft results to CSV for scoring later."""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Team", "Nickname", "Pick_Order", "Player", "NHL_Team",
                "Position", "Reg_Season_Stats"
            ])
            for name, team in teams.items():
                for i, p in enumerate(team.roster, 1):
                    writer.writerow([
                        name,
                        team.display_name,
                        i,
                        p["name"],
                        p["team"],
                        p["position"],
                        p.get("pts_display", ""),
                    ])
        print(f"  📊 Results CSV saved: {self.csv_path}")
