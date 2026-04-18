from __future__ import annotations

"""
GDS AI Playoff Draft — Orchestrator

Controls the 3-phase draft flow:
  Phase 1: Identity & Scouting (parallel across 12 models)
  Phase 2: Snake Draft (sequential, 120 picks)
  Phase 3: Closing Statements (sequential, 12 statements)
"""

import json
import random
import re
import time
import concurrent.futures

from .api_client import APIClient, extract_json, extract_pick_from_text
from .context_builder import (
    build_identity_prompt, build_identity_followup,
    build_pick_prompt, build_pick_tool_followup,
    build_chirp_prompt, build_closing_prompt,
)
from .tools import execute_tool
from .validators import fuzzy_match_player, validate_position_for_team
from .logger import DraftLogger


class DraftError(Exception):
    """Raised when the draft encounters an unrecoverable error (no auto-picks)."""
    pass


class Orchestrator:
    """
    Main draft controller. Runs all three phases sequentially.
    """

    def __init__(self, config: dict, draft_state, api_client: APIClient,
                 logger: DraftLogger):
        self.config = config
        self.state = draft_state
        self.api = api_client
        self.logger = logger
        # Track used nicknames across all models to prevent duplicates
        self.used_nicknames: set[str] = set()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Identity & Scouting
    # ═══════════════════════════════════════════════════════════════

    def run_phase_1(self):
        """
        Each model creates its persona, researches players, and
        writes strategy + scouting docs. Runs in parallel.
        """
        print("\n" + "=" * 60)
        print("  PHASE 1: IDENTITY & SCOUTING")
        print("=" * 60)

        team_names = list(self.state.teams.keys())

        # Run identity phase for each model (parallel, capped by semaphore)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = {
                pool.submit(self._run_identity_for_team, name): name
                for name in team_names
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  ❌ {name} identity failed: {e}")
                    # Set fallback persona so the draft can continue
                    self._set_fallback_persona(name)

        # Log all personas to markdown
        self.logger.log_personas_intro(self.state.teams)
        print("\n  ✅ Phase 1 complete — all personas created.")

    def _run_identity_for_team(self, team_name: str):
        """Run the full identity phase for one team."""
        team = self.state.teams[team_name]
        print(f"  🎭 {team_name} ({team.slug}) creating persona...")

        start = time.time()
        total_tool_calls = []

        # Step 1: Initial call — model creates persona + requests tools
        messages = build_identity_prompt(team_name, self.state, self.config)
        result = self.api.call_model(
            model=team.slug,
            messages=messages,
            max_tokens=1500,
            temperature=0.8,
        )

        parsed = extract_json(result["content"])

        # Step 2: Execute requested tool calls (if any)
        if parsed and parsed.get("tool_calls"):
            tool_results_text = ""
            for tc in parsed["tool_calls"][:4]:  # Max 4 tool calls in Phase 1
                tool_name = tc.get("tool", "")
                tool_args = {k: v for k, v in tc.items() if k != "tool"}
                print(f"    🔧 {team_name} → {tool_name}: {tool_args.get('query', '')[:50]}")

                tool_result = execute_tool(
                    tool_name, tool_args, team_name,
                    self.state, self.api, self.config,
                )
                tool_results_text += f"\n--- {tool_name} result ---\n{tool_result}\n"
                total_tool_calls.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_preview": tool_result[:200],
                })

            # Step 3: Follow-up call with tool results
            messages.append({"role": "assistant", "content": result["content"]})
            messages.extend(build_identity_followup(tool_results_text))

            result = self.api.call_model(
                model=team.slug,
                messages=messages,
                max_tokens=1500,
                temperature=0.7,
            )
            parsed = extract_json(result["content"])

        # Step 4: Apply persona data to team state
        elapsed = round(time.time() - start, 1)

        if parsed:
            self._apply_persona(team, parsed)
        else:
            # JSON parse failed — try aggressive text extraction
            print(f"    ⚠️ {team_name}: JSON parse failed, attempting text extraction...")
            extracted = self._extract_persona_from_text(result["content"])
            if extracted.get("nickname") and extracted["nickname"] != team.name:
                self._apply_persona(team, extracted)
                print(f"    ✅ {team_name}: Recovered persona from text")
            else:
                print(f"    ⚠️ {team_name}: Using fallback persona")
                self._set_fallback_persona(team_name)

        # Step 5: Validate voice description — retry if missing
        if not team.voice_description or len(team.voice_description.strip()) < 10:
            print(f"    ⚠️ {team_name}: Missing voice description, requesting...")
            for voice_attempt in range(2):
                voice_msg = (
                    f"You are {team.nickname}. Your persona: {team.persona}\n\n"
                    f"You forgot to provide a voice_description for the podcast. "
                    f"Describe your ideal podcast voice in 50-100 characters. "
                    f"Be specific about tone, accent, speed, and style.\n\n"
                    f'Respond with ONLY: {{"voice_description": "your description"}}'
                )
                voice_result = self.api.call_model(
                    model=team.slug,
                    messages=[{"role": "user", "content": voice_msg}],
                    max_tokens=200,
                    temperature=0.7,
                )
                voice_parsed = extract_json(voice_result.get("content", ""))
                if voice_parsed and voice_parsed.get("voice_description"):
                    team.voice_description = voice_parsed["voice_description"][:500]
                    print(f"    ✅ {team_name}: Got voice description on retry {voice_attempt + 1}")
                    break
            else:
                # Meaningful default based on the model's personality
                team.voice_description = (
                    f"Confident, articulate AI voice with distinct personality. "
                    f"Speaks with authority about hockey."
                )
                print(f"    ℹ️ {team_name}: Using generated voice description")

        # Step 6: Enforce unique nicknames — "you snooze you lose" rule
        # If another model (finishing faster) already claimed the nickname,
        # we re-call this model and tell them to pick a new one.
        nickname_lower = team.nickname.lower().strip()
        if nickname_lower in self.used_nicknames:
            original = team.nickname
            print(f"    ⚠️ {team_name}: Nickname '{original}' already taken — asking for a new one...")
            for nick_attempt in range(3):
                nick_msg = (
                    f"You are {team_name}. Your persona: {team.persona}\n\n"
                    f"You chose the nickname \"{original}\" but another AI model "
                    f"beat you to it — they finished their setup faster. "
                    f"You snooze, you lose!\n\n"
                    f"Already taken nicknames: {', '.join(sorted(self.used_nicknames))}\n\n"
                    f"Choose a DIFFERENT 2-3 word draft nickname that fits your "
                    f"personality. Make it memorable and unique.\n\n"
                    f'Respond with ONLY: {{"nickname": "Your New Nickname"}}'
                )
                nick_result = self.api.call_model(
                    model=team.slug,
                    messages=[{"role": "user", "content": nick_msg}],
                    max_tokens=200,
                    temperature=0.9,  # Higher temp for creativity
                )
                nick_parsed = extract_json(nick_result.get("content", ""))
                if nick_parsed and nick_parsed.get("nickname"):
                    new_nick = nick_parsed["nickname"].strip()
                    new_nick_lower = new_nick.lower().strip()
                    if new_nick_lower not in self.used_nicknames:
                        team.nickname = new_nick
                        nickname_lower = new_nick_lower
                        print(f"    ✅ {team_name}: New nickname '{new_nick}' (attempt {nick_attempt + 1})")
                        break
                    else:
                        print(f"    🔄 {team_name}: '{new_nick}' also taken — retrying...")
            else:
                # 3 retries exhausted — use model name as absolute last resort
                team.nickname = f"The {team_name}"
                nickname_lower = team.nickname.lower().strip()
                print(f"    ℹ️ {team_name}: Gave up — using 'The {team_name}'")
        self.used_nicknames.add(nickname_lower)

        # Log
        persona_data = {
            "nickname": team.nickname,
            "persona": team.persona,
            "voice_description": team.voice_description,
            "tiebreaker": team.tiebreaker,
        }
        self.logger.log_persona(
            team_name, persona_data, result["usage"],
            elapsed, total_tool_calls,
        )

        print(f"  ✅ {team_name} → \"{team.nickname}\" ({elapsed}s)")

    def _apply_persona(self, team, parsed: dict):
        """Apply parsed persona data to team state."""
        team.nickname = parsed.get("nickname", team.name)[:60]
        team.persona = parsed.get("persona", "A hockey-loving AI.")[:1000]
        team.voice_description = parsed.get("voice_description", "")[:500]
        team.strategy_doc = parsed.get("strategy_doc", "")[:8000]
        team.scouting_notes = parsed.get("scouting_notes", "")[:16000]
        try:
            team.tiebreaker = int(parsed.get("tiebreaker_prediction", 700))
        except (TypeError, ValueError):
            team.tiebreaker = 700

    def _extract_persona_from_text(self, text: str) -> dict:
        """
        Last-resort extraction: scan raw text for persona fields
        when JSON parsing has completely failed. Looks for quoted values
        after key names, or common patterns.
        """
        import re
        result = {}

        # Try to find nickname — look for patterns like:
        # nickname: "Something", Nickname: Something, **Something**
        patterns = [
            r'"nickname"\s*:\s*"([^"]+)"',
            r'nickname[:\s]+["\']?([A-Z][^"\'\n,]{2,25})',
            r'\*\*([A-Z][^*]{2,25})\*\*',  # Markdown bold
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                result["nickname"] = m.group(1).strip()
                break

        # Try to find persona description
        persona_patterns = [
            r'"persona"\s*:\s*"([^"]+)"',
            r'persona[:\s]+["\']?(.{20,200}?)["\']?\s*(?:\n|$)',
        ]
        for pat in persona_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                result["persona"] = m.group(1).strip()
                break

        # Voice description
        voice_patterns = [
            r'"voice_description"\s*:\s*"([^"]+)"',
            r'voice[:\s]+["\']?(.{20,100}?)["\']?\s*(?:\n|$)',
        ]
        for pat in voice_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                result["voice_description"] = m.group(1).strip()
                break

        # Strategy doc — grab a big block of text if present
        strat_patterns = [
            r'"strategy_doc"\s*:\s*"([^"]+)"',
            r'(?:strategy|game\s*plan)[:\s]+(.{50,1500}?)(?:\n\n|\n[A-Z#▸])',
        ]
        for pat in strat_patterns:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                result["strategy_doc"] = m.group(1).strip()
                break

        # Scouting notes
        scout_patterns = [
            r'"scouting_notes"\s*:\s*"([^"]+)"',
            r'(?:scouting|rankings?|notes)[:\s]+(.{50,2500}?)(?:\n\n[A-Z#▸]|$)',
        ]
        for pat in scout_patterns:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                result["scouting_notes"] = m.group(1).strip()
                break

        # Tiebreaker — look for a standalone number 400-1200
        tb_match = re.search(r'(?:tiebreaker|prediction|total\s*goals?)[:\s]*(\d{3,4})', text, re.IGNORECASE)
        if tb_match:
            result["tiebreaker_prediction"] = int(tb_match.group(1))

        return result

    def _set_fallback_persona(self, team_name: str):
        """Set a generic fallback persona when identity phase fails."""
        team = self.state.teams[team_name]
        team.nickname = f"The {team_name} GM"
        team.persona = f"A no-nonsense hockey analyst powered by {team.provider}."
        team.strategy_doc = "Draft the best available player each round."
        team.tiebreaker = 700

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Snake Draft
    # ═══════════════════════════════════════════════════════════════

    def run_phase_2(self):
        """
        Execute the full snake draft: 10 rounds × 12 teams = 120 picks.
        Sequential — each pick must complete before the next begins.
        """
        print("\n" + "=" * 60)
        print("  PHASE 2: SNAKE DRAFT")
        print("=" * 60)

        pick_overall = 0

        for round_num in range(1, self.state.total_rounds + 1):
            round_order = self.state.get_snake_order(round_num)

            print(f"\n  ── Round {round_num} ──")
            self.logger.log_round_start(round_num, round_order, self.state.teams)

            for team_name in round_order:
                pick_overall += 1
                self._execute_pick(team_name, round_num, pick_overall)

                # Auto-save after every round
                if pick_overall % 12 == 0:
                    self.logger.save(self.api)

        print(f"\n  ✅ Phase 2 complete — {pick_overall} picks made.")

    def _execute_pick(self, team_name: str, round_num: int, pick_overall: int):
        """
        Execute a single draft pick for one team.
        Includes optional tool call, pick validation with escalating retries,
        and chirp reactions.

        Retry strategy:
          Attempts 1-3: Standard retry with error context and player suggestions
          Attempts 4-5: Firm mode — simplified prompt, lower temp, explicit player list
          After 5: HARD FAIL — raises DraftError. No auto-picks, ever.
        """
        team = self.state.teams[team_name]
        start = time.time()
        tool_calls = []
        chirp = ""
        result = {}  # Initialized here so log_pick can safely access it even after exceptions

        try:
            # Build the pick prompt
            messages = build_pick_prompt(
                team_name, self.state, round_num, pick_overall, self.config,
            )

            # Call the model — generous output cap so JSON doesn't get clipped
            result = self.api.call_model(
                model=team.slug,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )
            # Guard against None content from API
            content = result.get("content") or ""
            parsed = extract_json(content)

            # Check for tool call first
            if parsed and parsed.get("tool") and not parsed.get("pick"):
                tool_name = parsed["tool"]
                tool_args = {k: v for k, v in parsed.items() if k != "tool"}
                print(f"    🔧 {team.display_name} → {tool_name}")

                tool_result = execute_tool(
                    tool_name, tool_args, team_name,
                    self.state, self.api, self.config,
                )
                tool_calls.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_preview": tool_result[:200],
                })

                # Follow-up: now make the actual pick
                messages.append({"role": "assistant", "content": result["content"]})
                followup = build_pick_tool_followup(tool_name, tool_result, self.config)
                messages.extend(followup)

                result = self.api.call_model(
                    model=team.slug,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7,
                )
                content = result.get("content") or ""
                parsed = extract_json(content)

            # === PICK VALIDATION WITH ESCALATING RETRIES ===
            # Attempts 1-3: standard retry with error context
            # Attempts 4-5: "firm mode" — simplified prompt with explicit player suggestions
            # After 5: HARD FAIL — no auto-picks ever.
            max_attempts = 5
            last_error_reason = ""
            for attempt in range(max_attempts):
                # Parse the pick from the response
                if not parsed:
                    content = result.get("content") or ""
                    parsed = extract_pick_from_text(content)

                player_name = parsed.get("pick", "") if parsed else ""
                position = parsed.get("position", "").upper() if parsed else ""
                chirp = parsed.get("chirp", "") if parsed else ""

                # Try to resolve the pick
                pick_result = self._try_resolve_pick(
                    team_name, player_name, position, chirp,
                    round_num, pick_overall,
                )

                if pick_result["status"] == "ok":
                    # Valid pick — we're done
                    pick_data = pick_result["pick_data"]
                    break
                elif attempt < max_attempts - 1:
                    # Invalid pick — escalate the retry message
                    last_error_reason = pick_result["reason"]
                    is_firm = attempt >= 3  # Attempts 4-5 get the firm version
                    print(f"    🔄 {team.display_name}: {last_error_reason} — "
                          f"{'FIRM ' if is_firm else ''}retry (attempt {attempt + 2}/{max_attempts})")

                    retry_msg = self._build_retry_message(
                        last_error_reason, team_name,
                        pick_result.get("player_name", ""),
                        firm=is_firm,
                    )
                    messages.append({"role": "assistant", "content": result.get("content") or ""})
                    messages.append({"role": "user", "content": retry_msg})

                    result = self.api.call_model(
                        model=team.slug,
                        messages=messages,
                        max_tokens=800,
                        temperature=0.5 if is_firm else 0.7,  # Lower temp for firm retries
                    )
                    content = result.get("content") or ""
                    parsed = extract_json(content)
                else:
                    # ALL RETRIES EXHAUSTED — HARD FAIL. NO AUTO-PICKS.
                    raw_content = (result.get("content") or "")[:300]
                    raise DraftError(
                        f"DRAFT HALTED: {team.display_name} ({team.slug}) failed to make "
                        f"a valid pick after {max_attempts} attempts.\n"
                        f"  Pick #{pick_overall} (Round {round_num})\n"
                        f"  Last error: {last_error_reason}\n"
                        f"  Last raw response: {raw_content}"
                    )

        except DraftError:
            # Re-raise our own errors — these should halt the draft
            raise
        except Exception as e:
            # Unexpected error — still fail hard, no masking
            raise DraftError(
                f"DRAFT HALTED: Unexpected error during {team.display_name}'s pick.\n"
                f"  Pick #{pick_overall} (Round {round_num})\n"
                f"  Error: {type(e).__name__}: {str(e)[:200]}"
            ) from e

        elapsed = round(time.time() - start, 1)

        # Get chirp reactions from 3 random other GMs
        reactions = self._get_reactions(
            team_name, pick_data, round_num, pick_overall,
        )
        pick_data["reactions"] = reactions

        # Log the pick
        self.logger.log_pick(
            pick_data, team.display_name,
            result.get("usage", {}),
            elapsed, tool_calls, False,
        )

        # Print the pick
        player_info = f"{pick_data['player']} ({pick_data['player_team']}/{pick_data['position']})"
        print(f"  ✅ Pick #{pick_overall}: {team.display_name} → {player_info} ({elapsed}s)")
        if chirp:
            print(f"    💬 \"{chirp[:150]}\"")

    def _build_retry_message(self, error_reason: str, team_name: str,
                              player_name: str, firm: bool = False) -> str:
        """
        Build a retry message telling the model what went wrong.

        Standard mode (attempts 1-3): Explains the error with position hints.
        Firm mode (attempts 4-5): Strips all complexity. Lists 5 exact names
        to pick from. Lower temperature. Basically: "pick one of these."
        """
        team = self.state.teams[team_name]
        roster_format = self.config["draft"]["roster_format"]
        needed = team.positions_needed(roster_format)
        needs_display = ", ".join(f"{v}{k}" for k, v in needed.items())

        # Get top available at each needed position
        suggestions = []
        for pos, count in needed.items():
            players_at_pos = [
                p for p in self.state.available_players if p["position"] == pos
            ]
            top_n = players_at_pos[:5] if firm else players_at_pos[:3]
            for p in top_n:
                suggestions.append(f"  • {p['name']} ({p['team']}/{pos})")
        suggestions_text = "\n".join(suggestions)

        if firm:
            # FIRM MODE — dead simple, no room for interpretation
            return (
                f"YOUR PREVIOUS PICK WAS INVALID. This is your last chance.\n\n"
                f"Error: {error_reason}\n\n"
                f"Pick EXACTLY ONE player from this list:\n{suggestions_text}\n\n"
                f"Respond with ONLY this JSON (nothing else):\n"
                f'{{"pick": "Player Full Name", "position": "F/D/G", '
                f'"chirp": "short comment"}}'
            )
        else:
            return (
                f"⚠️ INVALID PICK: {error_reason}\n\n"
                f"You tried to pick \"{player_name}\" but that pick is not valid.\n\n"
                f"Positions you still need: {needs_display}\n"
                f"Top available players at your needed positions:\n{suggestions_text}\n\n"
                f"Pick a DIFFERENT player who is AVAILABLE. Respond with JSON:\n"
                f"{{\"pick\": \"Player Full Name\", \"position\": \"F/D/G\", "
                f"\"chirp\": \"Your in-character comment\"}}"
            )

    def _try_resolve_pick(self, team_name: str, player_name: str, position: str,
                          chirp: str, round_num: int, pick_overall: int) -> dict:
        """
        Attempt to validate and record a pick. Returns a status dict:
          {"status": "ok", "pick_data": {...}} on success
          {"status": "error", "reason": "...", "player_name": "..."} on failure
        Does NOT auto-pick — the caller decides what to do on error.
        """
        team = self.state.teams[team_name]
        roster_format = self.config["draft"]["roster_format"]

        # No player name at all — parse failure
        if not player_name:
            return {
                "status": "error",
                "reason": "Could not parse a player name from your response",
                "player_name": "",
            }

        # Try to match the player name against available pool
        matched = fuzzy_match_player(player_name, self.state.available_players)

        if not matched:
            return {
                "status": "error",
                "reason": f"\"{player_name}\" is not available (already drafted or not in the player pool)",
                "player_name": player_name,
            }

        actual_position = matched["position"]

        # If model specified wrong position, use the actual position
        if position and position != actual_position:
            position = actual_position

        # Check position cap
        pos_ok, reason = validate_position_for_team(
            actual_position, team, roster_format,
        )

        if not pos_ok:
            return {
                "status": "error",
                "reason": f"{reason} — pick a player at a position you still need",
                "player_name": player_name,
            }

        # Valid pick — record it
        pick_record = self.state.make_pick(
            team_name, matched["name"], round_num, pick_overall,
        )
        pick_record["chirp"] = chirp[:500]
        return {"status": "ok", "pick_data": pick_record}

    # _auto_pick has been intentionally removed.
    # The draft now uses escalating retries (standard → firm) and then
    # hard-fails with DraftError. No silent fallbacks.

    def _get_reactions(self, drafter_name: str, pick_data: dict,
                       round_num: int, pick_overall: int) -> list[dict]:
        """
        Get chirp reactions from N random non-drafting GMs.
        Runs in parallel for speed.
        """
        reaction_count = self.config["reactions"]["count"]
        other_teams = [n for n in self.state.teams if n != drafter_name]
        reactors = random.sample(other_teams, min(reaction_count, len(other_teams)))

        reactions = []

        # Run reactions in parallel (they're independent and fast)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {}
            for reactor_name in reactors:
                future = pool.submit(
                    self._get_single_reaction,
                    reactor_name, drafter_name, pick_data,
                    round_num, pick_overall,
                )
                futures[future] = reactor_name

            for future in concurrent.futures.as_completed(futures):
                reactor_name = futures[future]
                try:
                    rxn = future.result()
                    if rxn:
                        reactions.append(rxn)
                except Exception as e:
                    print(f"    ⚠️ {reactor_name} reaction failed: {str(e)[:50]}")

        return reactions

    def _get_single_reaction(self, reactor_name: str, drafter_name: str,
                              pick_data: dict, round_num: int,
                              pick_overall: int) -> dict | None:
        """Get one chirp reaction from a non-drafting GM."""
        reactor = self.state.teams[reactor_name]

        messages = build_chirp_prompt(
            reactor_name, drafter_name,
            pick_data["player"], pick_data["player_team"],
            pick_data["position"], round_num, pick_overall,
            pick_data.get("chirp", ""),
            self.state, self.config,
        )

        result = self.api.call_model(
            model=reactor.slug,
            messages=messages,
            max_tokens=300,
            temperature=0.9,  # Higher temp for more creative chirps
        )

        # Guard against None content (some models return empty responses)
        content = result.get("content") or ""
        if not content.strip():
            return None

        parsed = extract_json(content)
        chirp_text = ""
        if parsed:
            chirp_text = parsed.get("chirp", "")
        else:
            # Raw text fallback — strip JSON artifacts so we get clean text
            chirp_text = content.strip()
            # Remove code fence wrappers
            chirp_text = re.sub(r'^```(?:json)?\s*', '', chirp_text)
            chirp_text = re.sub(r'\s*```$', '', chirp_text)
            # Remove JSON key wrappers like {"chirp": "actual text"}
            chirp_text = re.sub(r'^\{["\s]*chirp["\s]*:["\s]*', '', chirp_text)
            chirp_text = re.sub(r'["\s]*\}?\s*$', '', chirp_text)
            chirp_text = chirp_text.strip().strip('"')

        if chirp_text:
            max_chars = self.config["reactions"]["max_chars"]
            return {
                "team": reactor_name,
                "nickname": reactor.display_name,
                "chirp": chirp_text[:max_chars],
            }
        return None

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Closing Statements
    # ═══════════════════════════════════════════════════════════════

    def run_phase_3(self):
        """
        Each GM delivers a closing statement after seeing all final rosters.
        Sequential so they can be displayed in draft order.
        """
        print("\n" + "=" * 60)
        print("  PHASE 3: CLOSING STATEMENTS")
        print("=" * 60)

        for team_name in self.state.base_order:
            team = self.state.teams[team_name]
            print(f"  🎤 {team.display_name} delivering closing statement...")

            start = time.time()
            messages = build_closing_prompt(team_name, self.state, self.config)

            try:
                result = self.api.call_model(
                    model=team.slug,
                    messages=messages,
                    max_tokens=800,
                    temperature=0.8,
                )

                parsed = extract_json(result["content"])
                if parsed:
                    statement = parsed.get("closing_statement", "")
                else:
                    # Use raw text as statement — clean up code fence artifacts
                    statement = result["content"].strip()[:2000]

                # Post-process: strip any residual markdown/JSON wrappers
                # Some models (Gemini, Mistral) wrap the entire output in ```json
                if statement.startswith('```'):
                    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', statement)
                    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
                    # If the cleaned text is itself JSON, extract the statement
                    try:
                        inner = json.loads(cleaned)
                        if isinstance(inner, dict) and 'closing_statement' in inner:
                            statement = inner['closing_statement']
                        else:
                            statement = cleaned
                    except (json.JSONDecodeError, TypeError):
                        statement = cleaned

                team.closing_statement = statement
                elapsed = round(time.time() - start, 1)

                self.logger.log_closing(
                    team_name, team.display_name,
                    statement, result["usage"], elapsed,
                )
                print(f"  ✅ {team.display_name}: \"{statement[:60]}...\" ({elapsed}s)")

            except Exception as e:
                print(f"  ❌ {team.display_name} closing failed: {str(e)[:80]}")
                team.closing_statement = "(Statement unavailable)"

        # Write closings + final rosters to markdown
        self.logger.log_closings_section(self.state.teams)
        self.logger.log_final_rosters(self.state.teams)
        print("\n  ✅ Phase 3 complete — all closing statements recorded.")
