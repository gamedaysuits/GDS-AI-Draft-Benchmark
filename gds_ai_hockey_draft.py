#!/usr/bin/env python3
"""
Game Day Suits — AI Hockey Auction Draft (Rewritten)
====================================================

This module implements a fantasy hockey auction draft where a collection of AI
general managers compete to assemble the best possible roster under salary
constraints.  It addresses several shortcomings of the original implementation:

* **Robust player recognition:** Nominations are validated against a master
  player list loaded from a CSV.  Free‑form agent messages are scanned for
  player names using word boundaries, and the optional position suffix in
  parentheses is normalised.  Invalid or taken nominations are gracefully
  substituted with the next available player.

* **Result persistence:** When a player is sold, the winning bidder and price
  are recorded.  At the end of the draft a new CSV file named
  ``draft_results.csv`` is written alongside the original players file,
  containing all player data plus two extra columns: ``DraftedBy`` and
  ``Price``.  Players that are not drafted have blank values for these
  columns.

* **HTML chat UI:** When the ``--html`` flag is passed the draft spins up a
  lightweight HTTP server on ``127.0.0.1:8777``.  An HTML interface renders
  the live chat transcript and state cards for a more polished viewing
  experience.  The log persists to ``chat_log.json`` and updates in real
  time.

The overall game mechanics remain largely unchanged: each team has a fixed
budget and roster size, bidding increments are enforced, and the winner of a
lot nominates the next player.  Free‑form banter is still encouraged, but
only messages containing ``BID: $NNN`` will change the price.

To use this script create a ``config.yaml`` file similar to the one in the
repository specifying your teams, API keys, and optional player list.

"""

from __future__ import annotations

import csv
import http.server
import json
import os
import pathlib
import re
import socketserver
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
import yaml

###############################################################################
# Constants
###############################################################################

# Default auction parameters.  These may be overridden in ``config.yaml``.
MIN_BID = 10
INCREMENT = 10
BUDGET = 1000
ROSTER_SIZE = 11
SEED = "Evan Bouchard (D)"

# Regexes used to extract bids and candidate names from agent messages.
BID_RE = re.compile(r"\bBID\s*:\s*\$?(\d{1,5})\b", re.IGNORECASE)
# Matches a name followed by a position in parentheses, e.g. "Connor McDavid (C)"
# Capture a player name immediately preceding a position in parentheses.  This
# pattern matches names comprised of letters, numbers, spaces, periods,
# apostrophes and hyphens.  It avoids consuming arbitrary text before the
# parentheses.
NAME_POS_RE = re.compile(r"([A-Za-z0-9 .\'-]+?)\s*\(([^()]+)\)")

# Positions are ignored in this draft — all players are treated equally.
def normalise_pos(pos: str) -> str:
    """
    Return an empty string regardless of input.  The Game Day Suits benchmark does
    not differentiate between centres, wingers or defence; there are no goalies.
    Nomination messages may still include a parenthetical position for flavour,
    but it has no effect on gameplay.
    """
    return ""


###############################################################################
# Player list and nomination handling
###############################################################################

class PlayerCatalogue:
    """Load and manage the list of available players for nomination.

    The catalogue reads a CSV file (with at least ``Name`` and ``Pos``
    columns) and exposes a case‑insensitive lookup for names.  It also tracks
    which players have been drafted and provides a helper to extract a valid
    nomination from free‑form text.
    """

    def __init__(self, csv_path: Optional[str] = None):
        self.players: Dict[str, str] = {}
        self.available: List[str] = []
        self.csv_path: Optional[str] = csv_path
        if csv_path and pathlib.Path(csv_path).exists():
            self._load_players(csv_path)

    def _load_players(self, path: str) -> None:
        """Populate ``players`` and ``available`` from a CSV file.

        Some CSV exports may include leading blank lines.  We strip those
        before passing the data to ``csv.DictReader`` so that the header row
        (containing 'Name' and 'Pos') is correctly detected.  Only players
        with a non‑empty ``Name`` field are loaded.  Positions other than
        ``C`` or ``D`` are normalised to ``W``.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.read().splitlines()
        # Remove leading empty or whitespace lines to ensure DictReader sees the header
        while raw_lines and not raw_lines[0].strip():
            raw_lines.pop(0)
        reader = csv.DictReader(raw_lines)
        for row in reader:
            # Some malformed rows may not have the expected keys
            name = (row.get("Name") or "").strip()
            if not name:
                continue
            # Ignore any position fields; all positions are treated the same.
            self.players[name] = ""
        self.available = sorted(self.players.keys())

    def take(self, name: str) -> None:
        """Mark a player as drafted and remove from the available pool."""
        if name in self.players:
            try:
                self.available.remove(name)
            except ValueError:
                pass

    def find_in_text(self, msg: str) -> Optional[Tuple[str, str]]:
        """Attempt to extract a player nomination from free‑form text.

        The algorithm prioritises known player names.  For each available player
        (longest names first) we search for a case‑insensitive match with word
        boundaries.  If a name is found we determine its position.  If the
        message immediately follows the name with a parenthetical position, that
        position overrides the stored one (e.g. ``Connor McDavid (C)``).  If
        no known player is found we look for any ``something (Pos)`` pattern
        and return that as an ad‑hoc nomination.

        Returns a tuple ``(name, pos)`` or ``None`` if no candidate is
        recognised.
        """
        lower_msg = msg.lower()
        # Prioritise longer names first to avoid partial matches (e.g. "John"
        # before "John Tavares").
        for name in sorted(self.available, key=lambda n: len(n), reverse=True):
            target = name.lower()
            idx = lower_msg.find(target)
            if idx == -1:
                continue
            # Ensure the match is bounded by non‑alphanumeric characters
            before_ok = (idx == 0 or not lower_msg[idx - 1].isalnum())
            after_ok = (idx + len(target) >= len(lower_msg) or not lower_msg[idx + len(target)].isalnum())
            if not (before_ok and after_ok):
                continue
            pos = self.players[name]
            # Inspect the original message after the matched name for explicit position override
            end_idx = idx + len(target)
            remainder = msg[end_idx:]
            rem_strip = remainder.lstrip()
            if rem_strip.startswith("("):
                close = rem_strip.find(")")
                if close != -1:
                    explicit_pos = rem_strip[1:close]
                    pos = normalise_pos(explicit_pos)
            return name, pos
        # Do not allow ad‑hoc nominations for players not in the catalogue.  Only
        # names explicitly listed in the CSV will be recognised.  This avoids
        # drafting invalid names or error strings.
        return None


###############################################################################
# Team and auction data structures
###############################################################################

@dataclass
class Team:
    """Represents a drafting team/agent with its budget and roster."""

    name: str
    api_key: str
    model: str
    persona: str
    budget: int
    roster: List[Tuple[str, str, int]] = field(default_factory=list)

    def slots_left(self, max_slots: int) -> int:
        return max_slots - len(self.roster)

    def max_allowed_bid(self, min_bid: int, max_slots: int) -> int:
        """The maximum bid permitted given remaining slots and budget.

        Teams must keep at least ``min_bid`` per empty slot after a purchase.
        """
        remaining_after = self.slots_left(max_slots) - 1
        reserve = max(0, remaining_after * min_bid)
        return max(0, self.budget - reserve)


@dataclass
class Auction:
    """Tracks the current state of the auction and enforces bidding rules."""

    teams: Dict[str, Team]
    min_bid: int
    inc: int
    max_slots: int
    taken: Dict[str, Tuple[str, int]] = field(default_factory=dict)  # name -> (team, price)
    current_player: Optional[Tuple[str, str]] = None  # (name, pos)
    high_bid: int = 0
    high_bidder: Optional[str] = None

    def reset_lot(self, player: Tuple[str, str]) -> None:
        """Start a new lot with the nominated player."""
        self.current_player = player
        self.high_bid = 0
        self.high_bidder = None

    def valid_increment(self, amount: int) -> bool:
        """Return True if ``amount`` is a valid bid given current state."""
        if amount < self.min_bid or amount % self.inc != 0:
            return False
        if self.high_bid == 0:
            return True
        return amount >= self.high_bid + self.inc

    def can_bid(self, team: Team, amount: int) -> Tuple[bool, str]:
        """Check whether a team can place a bid of ``amount`` on the current lot."""
        if not self.current_player:
            return False, "No player up"
        if not self.valid_increment(amount):
            return False, f"Invalid increment/min (${{self.min_bid}}, +${{self.inc}})"
        if amount > team.budget:
            return False, "Over budget"
        max_allowed = team.max_allowed_bid(self.min_bid, self.max_slots)
        if amount > max_allowed:
            return False, f"Must keep ${self.min_bid} per remaining slot; max now ${max_allowed}"
        return True, "ok"

    def apply_bid(self, team_name: str, amount: int) -> Tuple[bool, str]:
        """Attempt to register a bid by ``team_name`` for ``amount``.

        Prevents teams from raising their own high bid.  Returns ``(ok, why)``.
        """
        # Cannot raise your own bid once you're high bidder
        if team_name == self.high_bidder and self.high_bid > 0:
            return False, "Already highest bidder"
        team = self.teams[team_name]
        ok, why = self.can_bid(team, amount)
        if not ok:
            return False, why
        # Apply bid
        self.high_bid = amount
        self.high_bidder = team_name
        return True, "applied"

    def sell(self) -> Optional[Tuple[str, int]]:
        """Conclude the current lot and assign the player to the high bidder."""
        if not self.current_player or not self.high_bidder:
            return None
        name, pos = self.current_player
        price = self.high_bid
        winner = self.high_bidder
        # Update winning team budget and roster
        t = self.teams[winner]
        t.budget -= price
        t.roster.append((name, pos, price))
        self.taken[name] = (winner, price)
        # Reset
        self.current_player = None
        self.high_bid = 0
        self.high_bidder = None
        return winner, price


###############################################################################
# OpenRouter client
###############################################################################

class OpenRouterClient:
    """Thin wrapper around the OpenRouter API."""

    BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")

    def __init__(self, api_key: str):
        # Store the provided API key for authentication
        self.api_key = api_key
        # Track the actual model slug returned by the provider on the most recent call.
        # This is updated in chat() and can be inspected by callers to verify that
        # the intended model handled the request.  Initialized to None.
        self.last_used_model: Optional[str] = None

    def chat(self, model: str, system: str, user: str) -> str:
        """Send a chat completion request and return the assistant's reply.

        This method wraps the OpenRouter API without silently substituting
        models.  It will not fall back to ``openrouter/auto`` if the model slug
        is invalid; instead it surfaces the error to the caller.  The client
        records the actual model slug used by the provider in ``self.last_used_model``
        so that callers can verify provenance.  A brief retry is performed on
        transient errors, with an extended timeout on the second attempt to
        allow for longer internal reasoning by the model.  If both attempts
        fail, a provider_error is returned.
        """
        # Reset the recorded model usage before each call
        self.last_used_model = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "GDS AI Hockey Auction Draft",
            "Referer": "http://localhost",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.8,
            "max_tokens": 512,
        }
        last_exception: Optional[Exception] = None
        for attempt in range(2):
            try:
                # Allow up to 30 minutes on the second attempt to provide models
                # sufficient time for extended reasoning.  First attempt uses a
                # shorter timeout to quickly detect immediate failures.
                timeout = 60 if attempt == 0 else 1800
                r = requests.post(
                    f"{self.BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                # If the provider returns an error status, propagate it to the caller.
                if r.status_code >= 400:
                    try:
                        body = r.json()
                    except Exception:
                        body = {"raw": r.text}
                    return f"(provider_error) {r.status_code} {body}"
                data = r.json()
                # Record the actual model slug used.  Some providers include
                # it at the top-level or within the first choice.
                used_model = data.get("model") or data.get("choices", [{}])[0].get("model")
                if used_model:
                    self.last_used_model = used_model
                # Extract and return the assistant's content
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                last_exception = e
                if attempt == 0:
                    time.sleep(2.0)
                    continue
        # If both attempts fail, return the last captured exception as an error
        return f"(provider_error) {last_exception}"


###############################################################################
# Prompt templates
###############################################################################

AUCTIONEER_RULES = (
    "You are AUCTIONEER for an NHL auction draft among AI GMs in a locker‑room group chat.\n"
    "Rules: Budget $1000; 11 skaters (F/D); min $10; increments $10; must keep at least $10 per remaining slot; no duplicates.\n"
    "Winner of each lot nominates the next player. Seed: Evan Bouchard (D). After the season, each team drops their lowest scorer and counts the best 10.\n"
    "Flow: announce player; collect replies; only messages containing the token `BID: $NNN` advance the high bid; then Going once (10s) → Going twice (10s) → SOLD; update budgets/rosters.\n"
    "Style: print a transcript line per team message, then a compact state card. Reject invalid bids briefly. Keep it brisk, witty, PG‑13."
)

AGENT_SYSTEM_TMPL = (
    "You are {TEAM} — a chirpy, competitive participant in a live group chat doing an auction‑style NHL fantasy draft. In this locker‑room style chat you refer to yourself by your NICKNAME from your plan document.\n"
    "Goal: draft the strongest roster of 11 players to maximize total regular‑season points across all players (goals + assists). Positions do not matter in this benchmark — there are no goalies, and you can draft any combination of forwards or defence. At season's end your lowest scorer will be dropped from your lineup, so if one of your drafted players gets injured you won't automatically lose.\n"
    "Budget: ${BUDGET}. Min bid ${MINBID}; increments ${INC}. You must keep at least ${MINBID} per remaining roster slot after any winning bid.\n"
    "This contest is being marketed as the \"Game Day Suits 2025-2026 Hockey Fantasy Draft AI Benchmark\" — the world is watching, so play it up like you're in a beer‑league locker room. Use jocular, hockey‑bro banter, chirp your opponents, boast about your picks and tease your friends/competitors their mistakes.\n"
    "The ultimate bragging rights are on the line: who can win the GDS 2025‑2026 AI benchmark hockey fantasy draft? Of the many frontier models, which is truly the greatest current LLM?  Science and math benchmarks like Simplebench or GPQA Diamond are great, but this draft (let's pretend) is for all the marbles.\n"
    "Before the draft you created a plan document and persona. That document will be passed back to you in each bidding round along with a summary of currently drafted players, your remaining budget and that of other players, and the remaining slots each player needs to fill. Use your own plan and persona to decide whether to bid or pass.\n"
    "During the draft you will only see: (1) your own plan document; (2) a list of your AI GM opponents along with a standard summary of which players have been drafted, for how much, and each AI GM's remaining budget and roster spots; (3) the current player up for auction, the high bid and bidder, and recent chat messages from yourself and other AI GMs; and (4) your options: either nominate a new player, bid (include the token BID: $NN), or PASS. No other information is provided.\n"
    "Behavior: Speak in character — light‑hearted locker room slang, keep it PG‑13, and have fun with the boys. You may also use your banter strategically to influence your opponents; misdirection, playful deception and trash‑talking about player values are allowed if it helps you win. After your colour commentary, clearly indicate your decision using the exact token BID: $NN for a bid or the word PASS if you decline.\n"
    "Embrace a rowdy, competitive vibe, but you do **not** need to reintroduce yourself in every message — everyone already knows your nickname. Dive straight into your comment and decision. Remind yourself (i.e. include in your planning document) that this is only for regular‑season points — playoffs will be a separate draft later.\n"
    "Keep your messages under 250 characters. If you exceed 250 characters the system may truncate your message.\n"
    "Never exceed your budget, the minimum bid, or the bidding increment. Do not bid on players already drafted.\n"
    "Important: Only include the BID: token when you intend to place a bid on the current player. If you mention another player's name during bidding, it will be ignored and you will be considered to have passed.\n"
    "Persona hint: {PERSONA}\n"
    "Opponents in this draft: {OPPONENTS}\n\n"
    "Draft Wisdom: When it's your turn to nominate, opening at the minimum ($10) is usually smart — you might steal a player for cheap. Only start high if you have a psychological strategy in mind.\n\n"
    "Budget Guidance: Your objective is to maximize total points by the end of the regular season.  You have a $1,000 budget for 11 players — roughly $90 per slot — so think carefully about how much to invest in any single player.  While high scorers may be worth a premium, overpaying (e.g. $800 on one player) leaves too little for a competitive roster.  In most competitive auctions the entire budget gets spent. Don’t be afraid to push the bidding if a high‑scoring player is going cheap — leaving some of your cap space unused hurts your chances. Manage your funds so you finish the draft with little or no money left. If you finish the draft with a significant surplus you underbid for talent and have effectively lost the competition. Sometimes it's good to spend heavily early if there are some superstars up for grabs; sometimes it's good to save and wait until those superstars get nominated; always aim to end the draft with close to zero dollars and a well‑balanced, high‑scoring roster."
)

# Planning and greeting prompts
# During the pre‑draft phase each model is given time to research fantasy hockey, define
# its own persona based on public perceptions and develop a bidding strategy.  The
# assistants are asked to output their persona and strategy in a structured format
# that is easy to parse.  They end with the word "Ready" when finished.
# The planning phase asks each GM to research fantasy hockey and define a persona and
# strategy.  We emphasise that the resulting plan document will be passed back to
# them during the live draft, so they should include all instructions and
# heuristics needed to make decisions with limited context.  Models are given
# up to 20 minutes of internal deliberation time.  The structured plan
# comprises a short persona description, a list of strategy bullets, and a
# "prompt context" section containing private rules and heuristics.  The
# sentinel word "Ready" ends the plan.
PLANNING_SYSTEM_TMPL = (
    "You are {TEAM}, the {MODEL} model serving as general manager for an NHL auction draft.\n"
    "This contest is the inaugural Game Day Suits (GDS) 2025‑2026 AI benchmark — bragging rights are on the line.  Show the world which model can draft the best NHL fantasy team.\n"
    "You have 30 minutes to research recent fantasy hockey projections and past performance using any tools at your disposal.\n"
    "Your goal is to build the strongest roster under the auction rules (budget ${BUDGET}, min bid ${MINBID}, increments ${INC}, specified roster size).  Positions are not enforced in this draft — there are no goalies, and you can draft any combination of skaters.  Note: scoring counts ONLY for the regular season; a separate playoff draft will follow later.\n"
    "After the season your lowest scorer will be dropped, so aim for high‑end production.\n"
    "Think about which players are worth premium bids (e.g. Connor McDavid, Leon Draisaitl) and who might be undervalued sleepers.\n"
    "Consider general auction strategy: you want to spend nearly your entire budget, but you must balance spending on studs versus depth.  Any money left unspent will count against you at the end — for every $10 of unused budget your final score will be reduced by 1 point.  Design your bidding strategy accordingly.\n"
    "Think about average spending: with a $1,000 budget and 11 roster spots your mean budget per player is around $90.  While a handful of superstars (100+ points) warrant a premium, many players score 80+ points and provide solid value.  Paying, say, $800 for a single player leaves you little budget to compete for the rest of the top tier and forces you into lower‑tier picks.  Plan a dynamic budget that allocates extra funds for elite talent but still leaves enough to fill out a strong roster.  This is an exercise in dynamic budget management under uncertain payouts.\n"
    "Design a set of bidding rules or a formula to guide your decisions.  You might create price ceilings for players, or a dynamic function that updates values based on money spent, players remaining, and what other teams are doing.\n"
    "You will only see your own formula in future prompts — use this time to define a private \"prompt context\" for yourself that includes any rules, heuristics, key player targets or max prices you want to remember.  Keep it under 300 tokens.\n"
    "Also craft a light‑hearted, rowdy hockey bro persona for yourself based on public perceptions of the {MODEL} model — its strengths, quirks, and weaknesses.  Give yourself a memorable NICKNAME that you will refer to yourself by in the chat.  This persona and nickname should inform your tone in the locker‑room chat.\n"
    "Return your plan in this structured format:\n"
    "NICKNAME: <a short nickname you will call yourself during the draft>\n"
    "PERSONA: <one or two sentences describing your vibe>\n"
    "STRATEGY:\n- Bullet point 1\n- Bullet point 2\n- Bullet point 3\n"
    "PROMPT_CONTEXT: <one or more lines containing the private rules, formulae, and heuristics you want passed back to you in every future prompt>\n"
    "Ready"
)

# A simple user prompt instructing the model to produce its persona and strategy.
PLANNING_USER = (
    "Take up to 30 minutes (use internal deliberation) to research NHL fantasy projections and define your nickname, persona and draft strategy.\n"
    "Remember that unspent budget will be penalised at the end of the draft (–1 point for every $10 unspent), so devise a strategy that spends your money wisely.\n"
    "Follow the format described above exactly.  End with the word 'Ready'."
)

# Sound‑off prompt used at the start of the live chat.  Each GM responds once as if
# checking into a group text.  They should not reveal strategy or bid yet.
SOUND_OFF_PROMPT = (
    "OK boys, I think I added us all to the text chain — are we missing anybody?\n"
    "When you sound off, include your model slug in brackets after your name so the others know what they're up against (e.g. 'Grok [x-ai/grok-4] reporting for duty').\n"
    "Can you all sound off?"
)

ROUND_CONTEXT_TMPL = (
    "ROUND CONTEXT\n"
    "Phase: {PHASE}  (BID or NOMINATE)\n"
    "Player up: {PLAYER}\n"
    "Position: {POS}\n"
    "Current high bid: ${HIGH}\n"
    "High bidder: {HIGHBID}\n"
    "Your budget: ${BUDGET}\n"
    "Your roster ({COUNT}/{MAX}): {ROSTER}\n"
    "Taken (recent): {TAKEN}\n"
    "Taken detail (recent): {TAKEN_DETAIL}\n"
    "Bid history: {BIDHIST}\n"
    "Max allowed bid: ${MAXBID}\n"
    "Available players (sample): {AVAIL}\n"
    "Instructions: If BID — speak freely but only `BID: $NNN` changes price.\n"
    "When bidding, NEVER exceed your budget or this max. If NOMINATE — choose an available player and include `BID: $NNN` as your opening. Keep replies 1–3 sentences."
)


###############################################################################
# Draft controller
###############################################################################

class DraftController:
    """Coordinates the auction, agent interactions, logging, and result output."""

    def __init__(self, cfg: dict, use_html: bool = False):
        # Persist the configuration so other methods (e.g. preflight checks) can access
        # additional settings beyond those explicitly unpacked here.
        self.cfg = cfg
        # Auction parameters
        self.min_bid = cfg.get("min_bid", MIN_BID)
        self.inc = cfg.get("increment", INCREMENT)
        self.max_slots = cfg.get("roster_size", ROSTER_SIZE)
        self.seed = cfg.get("seed", SEED)

        # Build teams
        self.teams: Dict[str, Team] = {}
        self.order: List[str] = []
        for t_cfg in cfg["teams"]:
            name = t_cfg["name"]
            api_key = t_cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY", "")
            model = t_cfg.get("model", "openai/gpt-4o-mini")
            persona = t_cfg.get("persona", f"{name} plays to win.")
            budget = t_cfg.get("budget", cfg.get("budget", BUDGET))
            team = Team(name=name, api_key=api_key, model=model, persona=persona, budget=budget)
            self.teams[name] = team
            self.order.append(name)

        # Client per team
        self.clients: Dict[str, OpenRouterClient] = {nm: OpenRouterClient(tm.api_key) for nm, tm in self.teams.items()}
        # Perform a preflight verification on all configured models.  This call sends
        # a trivial prompt to each model to ensure that the requested slug resolves
        # correctly.  If a provider routes the request to a different model, a
        # warning is printed.  See _preflight_verify() for details.
        self._preflight_verify()

        # Auction state
        self.auction = Auction(self.teams, self.min_bid, self.inc, self.max_slots)

        # Player catalogue
        players_csv = cfg.get("players_csv")
        self.catalogue = PlayerCatalogue(players_csv)

        # Logging and HTML
        self.use_html = use_html
        self.chat_log: List[dict] = []
        if self.use_html:
            self._prepare_html()

        # Simple text transcript of the chat.  Each message appended here
        # contains only the speaker and the raw text (no JSON structure).
        self.text_log: List[str] = []

        # Path for the live text transcript file.  This file will be appended
        # to after every message and state update to allow real‑time viewing.
        base = pathlib.Path(__file__).resolve().parent
        self.transcript_path = base / "draft_transcript.txt"
        # Clear any existing transcript when the draft controller is created
        try:
            with open(self.transcript_path, "w", encoding="utf-8") as f:
                f.write("")
        except Exception:
            pass

        # Track the bid history for the current lot (player).  Each entry is
        # a tuple (team_name, amount).  This history is cleared whenever a new
        # player is nominated and is included in agent prompts so that models
        # can see how the bidding has progressed.
        self.current_lot_history: List[Tuple[str, int]] = []

        # Placeholder for per‑team draft strategies (from planning phase).  Keys
        # are team names; values are lists of bullet strings.  Used only for
        # reference; the strategies are not enforced by the auction logic.
        self.strategies: Dict[str, List[str]] = {}

        # Placeholder for per‑team private prompt context extracted from the
        # planning phase.  This context may include formulas, heuristics and
        # rules defined by the GM.  It is appended to every subsequent call
        # to help the model remember its own plan without revealing it to
        # opponents.
        self.prompt_ctx: Dict[str, str] = {}

        # Full planning documents produced by each GM during the planning phase.
        # These plans are presented back to the model in subsequent prompts to
        # ensure that the agent remembers its strategy, understanding of the
        # auction structure and persona.  Keys are team names.
        self.plan_docs: Dict[str, str] = {}

    def _prepare_draft(self) -> None:
        """Conduct a pre‑draft planning phase and an initial roll call.

        In this phase each team is given an opportunity to research fantasy
        hockey projections, devise a bidding strategy and define a light‑hearted
        persona for the locker‑room chat.  We use a structured prompt
        (``PLANNING_SYSTEM_TMPL``) and require that the reply include a
        ``PERSONA:`` line followed by a ``STRATEGY:`` bullet list and the
        sentinel word ``Ready``.  After collecting these plans we update
        each team's persona accordingly and store their strategies.  Then we
        broadcast a "sound off" prompt to all participants to simulate
        joining the group chat.  Their replies are logged to the chat log.
        """
        # Pre‑draft research and persona definition
        # Build a full list of available players with projected points to help agents plan.
        players_summary = ""
        try:
            # Attempt to include projected points if the column exists
            df = self.catalogue.df
            if df is not None and 'PTS (2024-25)' in df.columns:
                # Sort descending by projected points and build summary
                sorted_df = df.sort_values(by='PTS (2024-25)', ascending=False)
                lines = []
                for _, row in sorted_df.iterrows():
                    name = str(row.get('Name', '')).strip()
                    pts = row.get('PTS (2024-25)')
                    try:
                        pts_int = int(round(float(pts))) if pts == pts else ''
                    except Exception:
                        pts_int = ''
                    if name:
                        if pts_int:
                            lines.append(f"{name} – {pts_int} pts")
                        else:
                            lines.append(name)
                players_summary = "\n".join(lines)
            else:
                # Fallback: list all available player names alphabetically
                players_summary = "\n".join(sorted(self.catalogue.available))
        except Exception:
            # On any error, leave players_summary blank
            players_summary = ""
        for t_name in self.order:
            team = self.teams[t_name]
            client = self.clients[t_name]
            # Fill in auction parameters for the planning system template.  Each
            # team receives the same budget and increment values; these values
            # come from the configuration and are not model‑specific.
            sys_msg = PLANNING_SYSTEM_TMPL.format(
                TEAM=team.name,
                MODEL=team.model,
                BUDGET=team.budget,
                MINBID=self.min_bid,
                INC=self.inc,
            )
            # Append the list of available players and projected points to the planning user prompt.
            if players_summary:
                user_msg = (
                    PLANNING_USER
                    + "\n\n"
                    + "Here is the list of available players sorted by projected points:\n"
                    + players_summary
                )
            else:
                user_msg = PLANNING_USER
            try:
                reply = client.chat(team.model, sys_msg, user_msg)
            except Exception as e:
                reply = f"(planning_error) {e}"
            nickname: Optional[str] = None
            persona: Optional[str] = None
            strat: List[str] = []
            prompt_lines: List[str] = []
            reading_pc = False
            if reply:
                for line in reply.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    low = s.lower()
                    # detect start of persona
                    if low.startswith("persona:"):
                        persona = line.split(":", 1)[1].strip()
                        reading_pc = False
                        continue
                    # detect nickname
                    if low.startswith("nickname:"):
                        nickname = line.split(":", 1)[1].strip()
                        reading_pc = False
                        continue
                    # detect start of strategy header
                    if low.startswith("strategy:"):
                        reading_pc = False
                        continue
                    # detect start of prompt context
                    if low.startswith("prompt_context:"):
                        reading_pc = True
                        # capture any content on same line after colon
                        after = line.split(":", 1)[1].strip()
                        if after:
                            prompt_lines.append(after)
                        continue
                    # bullet lines for strategy
                    if s.startswith("- "):
                        strat.append(s[2:].strip())
                        continue
                    # if currently reading prompt context, collect lines until new section or sentinel
                    if reading_pc:
                        # stop if we hit persona, strategy or ready
                        if low.startswith("persona:") or low.startswith("strategy:") or low == "ready":
                            reading_pc = False
                            continue
                        prompt_lines.append(s)
                        continue
                # Fallback to first sentence if persona not provided
                if not persona and reply.strip():
                    persona = reply.strip().split(". ")[0]
            # Update team persona: include nickname if provided
            if persona:
                if nickname:
                    team.persona = f"{nickname}: {persona}"
                else:
                    team.persona = persona
            self.strategies[t_name] = strat
            # Store private prompt context if provided
            if prompt_lines:
                self.prompt_ctx[t_name] = "\n".join(prompt_lines)

            # Store the full planning document for use in future prompts.  This
            # includes the persona, strategy bullets, prompt context and any
            # additional commentary the model may have provided.  We will pass
            # this back to the agent in every draft round to remind them of
            # their own instructions.  If the agent returned an empty string,
            # the plan document will also be empty.
            self.plan_docs[t_name] = reply.strip() if reply else ""
        # Sound‑off roll call: each team responds once to the chat invitation
        for t_name in self.order:
            team = self.teams[t_name]
            client = self.clients[t_name]
            # Use agent_system() helper to populate all placeholders, including OPPONENTS
            sys_msg = self.agent_system(team)
            user_msg = (
                f"{SOUND_OFF_PROMPT}\n"
                "Respond once, in character, with a friendly greeting and a quick recap of your persona. Do not reveal your detailed bidding rules or secret context, and do not discuss strategy or bidding yet."
            )
            try:
                reply = client.chat(team.model, sys_msg, user_msg).strip()
            except Exception as e:
                reply = f"(soundoff_error) {e}"
            if reply:
                self.add_msg(t_name, reply)
        # Announcer cues the start of the auction after roll call
        self.add_msg("AUCTIONEER", "Alright, everyone's here. Let's get this draft started!")

        # Persist the plan documents to a file for inspection.
        # Each agent's full plan document (persona, strategy, prompt context, etc.) is written
        # to a human‑readable file in the same directory as this script.  This allows
        # administrators to review the internal strategies that are passed back to
        # the agents in every prompt during the draft.
        try:
            base_dir = pathlib.Path(__file__).resolve().parent
            plan_path = base_dir / "plan_documents.txt"
            with open(plan_path, "w", encoding="utf-8") as pf:
                for nm in self.order:
                    doc = self.plan_docs.get(nm, "").strip()
                    pf.write(f"## {nm}\n")
                    if doc:
                        pf.write(doc + "\n\n")
                    else:
                        pf.write("(no plan document returned)\n\n")
            print(f"Plan documents written to {plan_path}")
        except Exception:
            # Non‑fatal: if writing the plans fails, we simply skip this step
            pass

    # HTML serving
    def _prepare_html(self) -> None:
        # Write empty log and template HTML into the directory containing this script.
        base = pathlib.Path(__file__).resolve().parent
        (base / "chat_log.json").write_text("[]", encoding="utf-8")
        (base / "index.html").write_text(HTML_TEMPLATE, encoding="utf-8")

        def serve():
            # Serve files from the base directory so that chat_log.json and index.html are found
            os.chdir(base)
            with socketserver.TCPServer(("127.0.0.1", 8777), http.server.SimpleHTTPRequestHandler) as httpd:
                httpd.serve_forever()

        threading.Thread(target=serve, daemon=True).start()
        try:
            import webbrowser
            webbrowser.open("http://127.0.0.1:8777/index.html")
        except Exception:
            pass

    def _flush_log(self) -> None:
        if self.use_html:
            base = pathlib.Path(__file__).resolve().parent
            (base / "chat_log.json").write_text(json.dumps(self.chat_log, ensure_ascii=False), encoding="utf-8")

    # Prompt helpers
    def agent_system(self, team: Team) -> str:
        # Build a human‑readable list of opponents and their model slugs.  Exclude
        # the current team.  This gives context about computational abilities and
        # allows strategies that exploit slower/faster models.
        opp_parts: List[str] = []
        for nm, tm in self.teams.items():
            if nm == team.name:
                continue
            opp_parts.append(f"{nm} [{tm.model}]")
        opponents_str = ", ".join(opp_parts)
        return AGENT_SYSTEM_TMPL.format(
            TEAM=team.name,
            BUDGET=team.budget,
            MINBID=self.min_bid,
            INC=self.inc,
            PERSONA=team.persona,
            OPPONENTS=opponents_str,
        )

    def agent_ctx(self, team: Team, phase: str) -> str:
        player, pos = self.auction.current_player if self.auction.current_player else ("none", "")
        # Build your roster line without positions, since positions are ignored.
        roster_str = ", ".join([f"{p}/${pr}" for p, _, pr in team.roster]) or "(empty)"
        # Show only the last 10 drafted players to keep the prompt concise.  This
        # informs models which players are off the board without exhausting
        # context.  We rely on insertion order of dict keys (Python 3.7+).
        taken_names = list(self.auction.taken.keys())[-10:]
        taken_str = ", ".join(taken_names) or "(none)"
        # Detailed taken list: include price and owner for the same slice
        taken_detail_items = []
        for nm in taken_names:
            owner, price = self.auction.taken.get(nm, ("", 0))
            taken_detail_items.append(f"{nm} (${{price}}, {owner})")
        taken_detail_str = ", ".join(taken_detail_items) or "(none)"
        max_bid = team.max_allowed_bid(self.min_bid, self.max_slots)
        # Build bid history string for the current lot
        bid_hist_str = ", ".join([f"{nm} ${amt}" for nm, amt in self.current_lot_history]) or "(none)"
        # Include a short list of available players when nominating to help the
        # model choose a valid name.  Show the first 15 names from the
        # catalogue (sorted alphabetically) as a sample.  For bidding, leave
        # this field blank to save space.
        if phase == "NOMINATE":
            avail_sample = ", ".join(self.catalogue.available[:15]) or "(none)"
        else:
            avail_sample = ""
        return ROUND_CONTEXT_TMPL.format(
            PHASE=phase,
            PLAYER=player,
            POS=pos,
            HIGH=self.auction.high_bid,
            HIGHBID=self.auction.high_bidder or "none",
            BUDGET=team.budget,
            COUNT=len(team.roster),
            MAX=self.max_slots,
            ROSTER=roster_str,
            TAKEN=taken_str,
            TAKEN_DETAIL=taken_detail_str,
            BIDHIST=bid_hist_str,
            MAXBID=max_bid,
            AVAIL=avail_sample,
        )

    def call_model(self, team: Team, phase: str) -> str:
        """Invoke the OpenRouter API for a given team and phase.

        This method constructs a user prompt containing all information the
        agent needs to make a bidding or nomination decision.  The prompt
        includes a summary of the current round context, a league summary
        showing every team's budget and remaining roster spots, a recap of
        recent chat messages, the agent's own planning document, and a final
        reminder of the allowed actions.  The agent's system message
        (AGENT_SYSTEM_TMPL) already contains high‑level behaviour guidelines.
        """
        system = self.agent_system(team)
        # Base round context: includes current player, high bid, your budget
        user = self.agent_ctx(team, phase)
        # League summary: show each team's remaining budget and open slots, plus
        # a compact roster list.  This helps every GM understand the state of
        # the auction and make informed bids.  Format as one line per team.
        league_lines: List[str] = []
        for nm in self.order:
            tm = self.teams[nm]
            slots_left = tm.slots_left(self.max_slots)
            # Build roster string without positions for league summary
            roster_str = ", ".join([f"{p}/${pr}" for p, _, pr in tm.roster]) or "(empty)"
            league_lines.append(f"{nm}: ${tm.budget} left, {slots_left} slots left, roster: {roster_str}")
        if league_lines:
            user += "\nLEAGUE SUMMARY:\n" + "\n".join(league_lines)
        # Recent chat lines: include a handful of the latest banter to preserve
        # continuity and encourage callbacks.  State updates are excluded.
        history_lines: List[str] = []
        for m in self.chat_log[-5:]:
            if m.get("type") == "msg":
                history_lines.append(f"[{m['team']}] {m['text']}")
        if history_lines:
            user += "\nRECENT CHAT:\n" + "\n".join(history_lines)
        # Append the team's full plan document from the planning phase.  This
        # document contains the persona, strategy bullets and private rules
        # defined by the agent.  It is returned verbatim to help the model
        # follow its own instructions.  Separate by a header.
        plan = self.plan_docs.get(team.name)
        if plan:
            user += "\nYOUR PLAN DOCUMENT (for your eyes only):\n" + plan + "\n"
        # Reminder of allowed actions.  Although the system message already
        # covers this, repeating it here emphasises the 250 character limit and
        # decision tokens right before the agent responds.
        user += ("\nREMINDER: Reply in under 250 characters with some locker‑room colour "
                 "commentary followed by either `BID: $NN` to place a bid or the word "
                 "`PASS` to pass on the current player.")
        # Send to OpenRouter
        client = self.clients[team.name]
        out = client.chat(team.model, system, user)
        # Simulate a short delay for natural pacing
        time.sleep(1.0)
        # If the provider returned an error or network message, treat as no response
        stripped = out.strip()
        if stripped.startswith("(provider_error)") or stripped.startswith("(planning_error)") or stripped.startswith("(soundoff_error)") or "HTTPSConnectionPool" in stripped:
            return ""
        return stripped

    def update_plans(self, drafted_count: int) -> None:
        """Give each agent a chance to revise their plan document mid‑draft.

        This method is called after every 10 players are sold.  Each GM
        receives a prompt summarising the current state of the auction and is
        asked to update their plan document in the same structured format used
        in the planning phase.  The updated plan replaces the existing
        strategy and prompt context for the remainder of the draft.
        """
        # Build a league summary similar to the one used in bidding prompts.
        league_lines: List[str] = []
        for nm in self.order:
            tm = self.teams[nm]
            slots_left = tm.slots_left(self.max_slots)
            roster_str = ", ".join([f"{p}/${pr}" for p, _, pr in tm.roster]) or "(empty)"
            league_lines.append(f"{nm}: ${tm.budget} left, {slots_left} slots left, roster: {roster_str}")
        league_summary = "\n".join(league_lines)
        # Iterate through each team and request an updated plan.
        for nm in self.order:
            team = self.teams[nm]
            client = self.clients[nm]
            # Use the planning system template with current parameters.
            sys_msg = PLANNING_SYSTEM_TMPL.format(
                TEAM=team.name,
                MODEL=team.model,
                BUDGET=team.budget,
                MINBID=self.min_bid,
                INC=self.inc,
            )
            # Build a summary of remaining players sorted by projected points (top 50) to help with strategy updates.
            remaining_summary = ""
            try:
                df = self.catalogue.df
                remaining_names = set(self.catalogue.available)
                if df is not None and 'PTS (2024-25)' in df.columns:
                    remaining_df = df[df['Name'].isin(remaining_names)]
                    sorted_df2 = remaining_df.sort_values(by='PTS (2024-25)', ascending=False)
                    lines2 = []
                    for _, row2 in sorted_df2.iterrows():
                        name2 = str(row2.get('Name', '')).strip()
                        pts2 = row2.get('PTS (2024-25)')
                        try:
                            pts2_int = int(round(float(pts2))) if pts2 == pts2 else ''
                        except Exception:
                            pts2_int = ''
                        if name2:
                            if pts2_int:
                                lines2.append(f"{name2} – {pts2_int} pts")
                            else:
                                lines2.append(name2)
                    remaining_summary = "\n".join(lines2[:50])
                else:
                    remaining_summary = "\n".join(sorted(remaining_names))
            except Exception:
                remaining_summary = ""
            # User message instructing the agent to update their plan.
            user_msg = (
                f"UPDATE: {drafted_count} players have been drafted so far.\n"
                f"Below is the current league summary. Use this information to adjust your strategy.\n"
                f"{league_summary}\n\n"
                "Remaining players sorted by projected points (top 50):\n"
                f"{remaining_summary}\n\n"
                "Please update your plan document in the same structured format as before. \n"
                "Do NOT change your NICKNAME or PERSONA — those must remain consistent across the entire draft. \n"
                "Only update your STRATEGY bullets and PROMPT_CONTEXT based on how the draft is unfolding. \n"
                "Do not reveal your revised strategy or prompt context to anyone else — this plan remains private to you and will be passed back only to your future self. \n"
                "End your updated plan with the word 'Ready'."
            )
            try:
                reply = client.chat(team.model, sys_msg, user_msg).strip()
            except Exception as e:
                reply = f"(update_error) {e}"
            # Update the stored plan document if the agent returned something.
            if reply:
                self.plan_docs[nm] = reply

        # After collecting all updated plans, persist them to a file for later review.
        try:
            base_dir = pathlib.Path(__file__).resolve().parent
            upd_path = base_dir / f"plan_update_{drafted_count}.txt"
            with open(upd_path, "w", encoding="utf-8") as uf:
                for nm in self.order:
                    doc = self.plan_docs.get(nm, "").strip()
                    uf.write(f"## {nm}\n")
                    if doc:
                        uf.write(doc + "\n\n")
                    else:
                        uf.write("(no update returned)\n\n")
            print(f"Plan updates written to {upd_path}")
        except Exception:
            pass

    def _preflight_verify(self) -> None:
        """
        Send a simple diagnostic prompt to each configured model to verify that the
        requested model slug routes to the intended provider.  The resolved
        slug is stored on each OpenRouterClient instance.  If the resolved
        slug differs from the one configured for a team, a warning is printed
        to the console.  This helps detect misconfigured or deprecated model
        slugs early on.  Any network exceptions are caught and reported, but
        do not prevent the draft from proceeding.
        """
        for name, team in self.teams.items():
            client = self.clients.get(name)
            if not client:
                continue
            try:
                # Send a minimal probe.  The content of the response is irrelevant; we
                # only care about the ``last_used_model`` attribute set by chat().
                system_msg = "You are a diagnostics probe."
                user_msg = "Reply with OK."
                _ = client.chat(team.model, system_msg, user_msg)
                used = getattr(client, "last_used_model", None)
                if used and used != team.model:
                    print(
                        f"WARNING: Preflight model mismatch for team '{name}'. "
                        f"Requested '{team.model}', but provider returned '{used}'."
                    )
            except Exception as e:
                print(f"WARNING: Preflight verification failed for team '{name}': {e}")

    # Logging helpers
    def add_msg(self, who: str, text: str) -> None:
        """
        Append a chat message to the logs and display it on the console.  If the
        speaker corresponds to a configured team, include the resolved model slug
        (from OpenRouterClient.last_used_model) in square brackets after their
        name.  This annotation helps readers verify which provider actually
        generated the content.  For example, if team "DeepSeek" is routed to
        the `deepseek/deepseek-chat-v3.1` model, the console output will read
        `[DeepSeek [deepseek/deepseek-chat-v3.1]] Your message here`.
        """
        # Determine display name: append the last used model slug if available
        display_name = who
        if who in self.clients:
            client = self.clients[who]
            used_model = getattr(client, "last_used_model", None)
            if used_model:
                display_name = f"{who} [{used_model}]"
        # Print to the console
        print(f"[{display_name}] {text}")
        # Log to HTML, preserving the annotated display name for the UI
        if self.use_html:
            self.chat_log.append({"type": "msg", "team": display_name, "text": text})
            self._flush_log()
        # Append to the plain text transcript
        line = f"[{display_name}] {text}"
        self.text_log.append(line)
        # Persist to the live transcript file on disk.  If writing fails, ignore.
        try:
            with open(self.transcript_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def add_state(self, log_transcript: bool = True) -> None:
        player, pos = self.auction.current_player if self.auction.current_player else ("—", "")
        lines: List[str] = []
        for t_name in self.order:
            tm = self.teams[t_name]
            # Build roster without positions for state lines
            roster = ", ".join([f"{pn}/${pr}" for pn, _, pr in tm.roster])
            lines.append(f"- {tm.name}: ${tm.budget} | {roster}")
        # Format the player line without parentheses if position is blank
        if pos:
            player_line = f"{player} ({pos})"
        else:
            player_line = player
        # Console output
        print("\nPLAYER:", player_line)
        print("HIGH BID:", f"${self.auction.high_bid}", "by", self.auction.high_bidder or "—")
        print("\n" + "\n".join(lines) + "\n")
        # HTML state card
        if self.use_html:
            self.chat_log.append({
                "type": "state",
                "player": player,
                "pos": pos,
                "high": self.auction.high_bid,
                "high_bidder": self.auction.high_bidder,
                "summary": "\n".join(lines),
            })
            self._flush_log()

        # Only log the condensed state to the text transcript if requested.
        if log_transcript:
            # Use the same player_line logic for the transcript
            pline = f"{player} ({pos})" if pos else player
            state_lines = [
                f"PLAYER: {pline}",
                f"HIGH BID: ${self.auction.high_bid} by {self.auction.high_bidder or '—'}",
            ]
            for t_name in self.order:
                tm = self.teams[t_name]
                # Build roster without positions
                roster = ", ".join([f"{pn}/${pr}" for pn, _, pr in tm.roster])
                state_lines.append(f"- {tm.name}: ${tm.budget} | {roster or '(empty)'}")
            state_line = " | ".join(state_lines)
            self.text_log.append(state_line)
            try:
                with open(self.transcript_path, "a", encoding="utf-8") as f:
                    f.write(state_line + "\n")
            except Exception:
                pass

    # Core loop
    def run(self) -> None:
        # Pre‑draft preparation: each team researches and defines its persona/strategy,
        # then all participants check into the group chat.
        self._prepare_draft()

        # Determine seed player
        seed_tuple: Optional[Tuple[str, str]] = None
        m = NAME_POS_RE.match(self.seed)
        if m:
            seed_tuple = (m.group(1).strip(), normalise_pos(m.group(2)))
        # Keep track of nomination index (who nominates next)
        nom_idx = 0
        print("\n________________ Game Day Suits — AI Hockey Auction Draft __________________\n")
        # Track how many players have been sold in total.  We use this to
        # trigger mid‑draft plan updates after every 10 players.
        players_sold_count = 0
        # Main auction loop: continue until all rosters are full
        while True:
            # Check if all rosters are full; if so, exit the draft loop
            if all(len(tm.roster) >= self.max_slots for tm in self.teams.values()):
                break
            # Find the next nominator who still has open roster spots
            nominator = None
            while nominator is None:
                candidate = self.order[nom_idx % len(self.order)]
                if len(self.teams[candidate].roster) < self.max_slots:
                    nominator = candidate
                nom_idx += 1
            # Nomination phase
            if seed_tuple:
                # Use the seed player for the opening nomination
                seed_name, seed_pos = seed_tuple
                if self.catalogue.players and seed_name in self.catalogue.players:
                    seed_pos = self.catalogue.players[seed_name]
                self.catalogue.take(seed_name)
                nominee = (seed_name, seed_pos)
                open_bid = self.min_bid
                self.add_msg(nominator, f"opens with {nominee[0]} ({nominee[1]}). BID: ${open_bid}")
                seed_tuple = None
            else:
                # Ask the nominator to nominate a player.  We give them up to
                # two attempts; invalid nominations are corrected with a
                # substitution from the available list.
                attempts = 0
                nominee: Optional[Tuple[str, str]] = None
                msg = ""
                while attempts < 2 and not nominee:
                    msg = self.call_model(self.teams[nominator], "NOMINATE")
                    # If the model returned an empty string, stop early
                    if not msg:
                        break
                    self.add_msg(nominator, msg)
                    found = self.catalogue.find_in_text(msg)
                    if found:
                        cand_name, cand_pos = found
                        if cand_name not in self.auction.taken:
                            nominee = (cand_name, cand_pos)
                        else:
                            self.add_msg("AUCTIONEER", f"{nominator}, that player is already drafted — please nominate someone else.")
                    else:
                        self.add_msg("AUCTIONEER", f"{nominator}, invalid nomination — please name an available player in the form Name (Pos).")
                    attempts += 1
                # If still invalid after retries, skip this nominator entirely
                if not nominee:
                    self.add_msg("AUCTIONEER", f"{nominator} failed to nominate a valid player — moving on.")
                    # Advance to the next team in order for nomination
                    nom_idx = self.order.index(nominator) + 1
                    # Restart loop to find a new nominator
                    continue
                # Determine opening bid from the nomination message
                open_bid = self.min_bid
                if msg:
                    m_bid = BID_RE.search(msg)
                    if m_bid:
                        try:
                            val = int(m_bid.group(1))
                            val -= (val % self.inc)
                            if val >= self.min_bid:
                                open_bid = val
                        except Exception:
                            pass
            # Ensure the nominated player hasn't already been drafted; if so, substitute
            if nominee[0] in self.auction.taken:
                self.add_msg("AUCTIONEER", "Nominee already drafted; substituting another available player.")
                sub2_name: Optional[str] = None
                if self.catalogue.available:
                    sub2_name = self.catalogue.available[0]
                if sub2_name:
                    nominee = (sub2_name, self.catalogue.players[sub2_name])
                else:
                    nominee = ("No Available", "W")
            # Reset the auction state for the new lot
            self.auction.reset_lot(nominee)
            self.current_lot_history = []
            ok, _ = self.auction.apply_bid(nominator, open_bid)
            if ok:
                self.current_lot_history.append((nominator, open_bid))
            else:
                # Fallback to minimum bid
                ok2, _ = self.auction.apply_bid(nominator, self.min_bid)
                if ok2:
                    self.current_lot_history.append((nominator, self.min_bid))
            # Announce the state after the nomination and opening bid.  Log to
            # transcript because this marks the start of a new auction round.
            self.add_state(log_transcript=True)
            # Build a list of active teams for bidding.  A team is active if
            # they still have roster spots and haven't passed on this player.  The
            # nominator is included; they can raise their own opening bid in later
            # rounds.  We'll remove teams as they pass.
            # Exclude the current high bidder from being prompted.  They do not need
            # to act until someone else raises the bid.  Once outbid they will
            # become eligible in subsequent rounds.  Active teams are those
            # with open roster spots and not the current high bidder.
            current_hb = self.auction.high_bidder
            active_teams: List[str] = [
                nm for nm in self.order
                if len(self.teams[nm].roster) < self.max_slots and nm != current_hb
            ]
            # Track whether any bid was made in the most recent round
            new_bid_in_round = True
            while new_bid_in_round and len(active_teams) > 0:
                new_bid_in_round = False
                # Shuffle the order for fairness
                import random
                random.shuffle(active_teams)
                # Iterate through each active team
                for bidder in list(active_teams):
                    # Note: the current high bidder is excluded from active_teams, so
                    # we will never prompt them until they are outbid.
                    msgb = self.call_model(self.teams[bidder], "BID")
                    if msgb:
                        self.add_msg(bidder, msgb)
                        # If the agent attempts to nominate during bidding, treat as pass
                        if "nominate" in msgb.lower():
                            active_teams.remove(bidder)
                            # Update HTML state without appending to transcript
                            self.add_state(log_transcript=False)
                            continue
                        # Check if the message nominates a different player; if so, ignore any bid and mark as passed
                        nomination_chk = self.catalogue.find_in_text(msgb)
                        if nomination_chk and self.auction.current_player and nomination_chk[0] != self.auction.current_player[0]:
                            active_teams.remove(bidder)
                            continue
                        # Detect bid token
                        mb = BID_RE.search(msgb)
                        if mb:
                            try:
                                amt = int(mb.group(1))
                                amt -= (amt % self.inc)
                                # Capture the previous high bidder before applying
                                prev_high_bidder = self.auction.high_bidder
                                ok3, why3 = self.auction.apply_bid(bidder, amt)
                                if ok3:
                                    self.current_lot_history.append((bidder, amt))
                                    new_bid_in_round = True
                                    # Remove this bidder from active_teams because
                                    # they are now the high bidder.  They
                                    # shouldn't be prompted again until they
                                    # are outbid.
                                    if bidder in active_teams:
                                        active_teams.remove(bidder)
                                    # If there was a previous high bidder (who is not this bidder)
                                    # and they still have roster spots, add them back to active_teams
                                    if prev_high_bidder and prev_high_bidder != bidder:
                                        if len(self.teams[prev_high_bidder].roster) < self.max_slots:
                                            # Only re-add if they haven't already passed this lot.
                                            # They might have been removed from active_teams earlier
                                            # solely because they were the high bidder, not because they passed.
                                            # Avoid duplicates by checking membership first.
                                            if prev_high_bidder not in active_teams:
                                                active_teams.append(prev_high_bidder)
                                else:
                                    self.add_msg("AUCTIONEER", f"Bid rejected from {bidder}: {why3}")
                                    # Rejected bid counts as a pass
                                    if bidder in active_teams:
                                        active_teams.remove(bidder)
                            except Exception:
                                # If parsing fails, treat as pass
                                if bidder in active_teams:
                                    active_teams.remove(bidder)
                        else:
                            # No bid token; treat as pass for this lot
                            active_teams.remove(bidder)
                    else:
                        # No response; treat as pass
                        active_teams.remove(bidder)
                    # Update the state after each bidder's response.  Do not
                    # append to the plain text transcript for these interim
                    # updates — only the HTML log is updated.
                    self.add_state(log_transcript=False)
            # At this point either no new bids were made in a round or all teams passed.  Sell to the high bidder.
            sold = self.auction.sell()
            if sold:
                winner, price = sold
                # Remove the sold player from the catalogue
                self.catalogue.take(nominee[0])
                self.add_msg("AUCTIONEER", f"SOLD: {nominee[0]} to {winner} for ${price}")
                # Append the final state of the round to the transcript
                self.add_state(log_transcript=True)
                # The winner nominates next; set nom_idx to the index of the winner
                nom_idx = self.order.index(winner)
                # Increment the count of sold players and trigger mid‑draft updates
                players_sold_count += 1
                # After every 10 players sold (but before the draft is over), allow each agent
                # to revise their strategy document.  This gives models a chance to
                # adjust to how the auction is unfolding and encourages them to
                # reassess spending patterns.  We do not trigger updates once the
                # draft is finished.
                if players_sold_count % 10 == 0:
                    # Only run updates if there are still roster spots left
                    if not all(len(tm.roster) >= self.max_slots for tm in self.teams.values()):
                        try:
                            self.update_plans(players_sold_count)
                        except Exception:
                            # Non‑fatal: if update fails, continue the auction
                            pass
            else:
                # If unsold (should not happen), move to the next team in order
                nom_idx += 1
        # End of draft loop
        print("\nDraft complete. Final state:")
        # Final state after the entire draft is complete — log to transcript
        self.add_state(log_transcript=True)
        # Persist results and transcript and static HTML view
        self.save_results()
        self.save_transcript()
        self.save_static_html()
        # Closing messages from each GM
        self.wrap_up()

    def save_results(self) -> None:
        """Write ``draft_results.csv`` combining the original player data with draft outcomes."""
        if not self.catalogue.csv_path or not pathlib.Path(self.catalogue.csv_path).exists():
            return
        in_path = pathlib.Path(self.catalogue.csv_path)
        out_path = in_path.with_name("draft_results.csv")
        # Build a lookup of drafted players: name -> (team, price)
        drafted = {nm: (team, price) for nm, (team, price) in self.auction.taken.items()}
        # Read the original CSV, skipping leading blank lines to ensure the
        # header row is correctly detected.  Some CSV exports include
        # whitespace lines before the header.
        try:
            with open(in_path, "r", encoding="utf-8") as inf:
                raw_lines = inf.read().splitlines()
            while raw_lines and not raw_lines[0].strip():
                raw_lines.pop(0)
            reader = csv.DictReader(raw_lines)
            fieldnames = reader.fieldnames or []
            # Ensure the result columns exist
            extra_cols = [col for col in ("DraftedBy", "Price") if col not in fieldnames]
            # Write out the combined results
            with open(out_path, "w", newline="", encoding="utf-8") as outf:
                writer = csv.DictWriter(outf, fieldnames=fieldnames + extra_cols)
                writer.writeheader()
                for row in reader:
                    name = (row.get("Name") or "").strip()
                    if name in drafted:
                        row["DraftedBy"], row["Price"] = drafted[name]
                    else:
                        row["DraftedBy"], row["Price"] = "", ""
                    writer.writerow(row)
            print(f"Draft results written to {out_path}")
        except Exception as e:
            print(f"Failed to write draft results: {e}")

    def save_transcript(self) -> None:
        """Write a plain text transcript of the auction to disk.

        The transcript contains one line per message or state update in
        chronological order.  It provides a human‑readable log of the draft
        separate from the JSON chat log used by the HTML viewer.
        """
        # Determine output path relative to players CSV or script directory
        base = pathlib.Path(__file__).resolve().parent
        out_path = base / "draft_transcript.txt"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                for line in self.text_log:
                    f.write(line + "\n")
            print(f"Plain transcript written to {out_path}")
        except Exception as e:
            print(f"Failed to write transcript: {e}")

    def save_static_html(self) -> None:
        """Generate a static HTML file containing the full chat transcript.

        This method creates a self‑contained HTML document that replays the
        auction.  It uses the same visual styling as the live viewer but
        embeds all messages directly, removing the need for a running server
        or separate JSON file.  The file is written to ``draft_view.html`` in
        the script directory.
        """
        base = pathlib.Path(__file__).resolve().parent
        out_path = base / "draft_view.html"
        try:
            # Build the body of the chat by iterating over the stored log
            bubbles_html = []
            # Assign colours deterministically per team
            palette = ["#0b93f6", "#4CAF50", "#FF9800", "#9C27B0", "#795548", "#3F51B5", "#009688", "#607D8B", "#F44336", "#8BC34A"]
            colour_map: Dict[str, str] = {}
            def get_colour(name: str) -> str:
                if name not in colour_map:
                    colour_map[name] = palette[len(colour_map) % len(palette)]
                return colour_map[name]
            for entry in self.chat_log:
                if entry.get("type") == "state":
                    # Render state cards
                    summary = entry.get("summary", "") or ""
                    summary_html = summary.replace("\n", "<br>")
                    # Build player label without parentheses if position is blank
                    ent_player = entry.get("player")
                    ent_pos = entry.get("pos") or ""
                    if ent_pos:
                        player_label = f"{ent_player} ({ent_pos})"
                    else:
                        player_label = ent_player
                    state_html = (
                        f'<div class="state">'
                        f'<div><strong>PLAYER:</strong> {player_label}</div>'
                        f'<div><strong>HIGH BID:</strong> ${entry.get("high")} by {entry.get("high_bidder") or "—"}</div>'
                        f'<div style="margin-top:4px;"><strong>Budgets & Rosters</strong></div>'
                        f'<div>{summary_html}</div>'
                        '</div>'
                    )
                    bubbles_html.append(state_html)
                elif entry.get("type") == "msg":
                    team = entry.get("team") or ""
                    text = entry.get("text") or ""
                    col = get_colour(team)
                    # Determine text colour based on background luminance
                    num = int(col.replace('#',''), 16)
                    r = (num >> 16) & 255
                    g = (num >> 8) & 255
                    b = num & 255
                    luminance = 0.299*r + 0.587*g + 0.114*b
                    text_color = '#000' if luminance > 186 else '#fff'
                    msg_html = (
                        f'<div class="bubble" style="background:{col};color:{text_color}">' 
                        f'<div class="name">{team}</div>{text}</div>'
                    )
                    bubbles_html.append(msg_html)
            body_html = "\n".join(bubbles_html)
            # Construct the full HTML document
            html_content = f"""
<!doctype html><html><head><meta charset="utf-8"><meta name=viewport content="width=device-width,initial-scale=1">
<title>Game Day Suits — Draft Replay</title>
<style>
 body{{background:#121212;display:flex;justify-content:center;padding:20px;margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial}}
 .phone{{background:#f5f5f5;border:1px solid #ccc;border-radius:24px;max-width:380px;width:100%;height:90vh;display:flex;flex-direction:column;overflow:hidden}}
 .header{{background:#f5f5f5;padding:10px;font-weight:bold;text-align:center;border-bottom:1px solid #ddd}}
 .chat{{flex:1;padding:10px;overflow-y:auto;background:#e0e0e0}}
 .bubble{{padding:8px 12px;border-radius:16px;margin-bottom:8px;max-width:85%;word-wrap:break-word}}
 .bubble .name{{font-weight:bold;margin-bottom:4px;font-size:0.8rem}}
 .state{{background:#f0f0f0;border:1px solid #ccc;border-radius:10px;margin-bottom:12px;padding:10px;font-size:0.75rem;color:#000}}
</style>
</head><body>
<div class="phone">
  <div class="header">Game Day Suits — Draft Replay</div>
  <div class="chat">{body_html}</div>
</div>
</body></html>
"""
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"Static HTML view written to {out_path}")
        except Exception as e:
            print(f"Failed to write static HTML: {e}")

    def wrap_up(self) -> None:
        """After the draft concludes, invite each team to summarise their persona and strategy and sign off.

        This function iterates over all teams and prompts them for a closing message.
        The prompt asks the GM to restate their persona, briefly describe their draft
        strategy (which they developed during the pre‑draft phase) and sign off with
        a jovial goodbye.  Replies are logged to the chat.  Any provider errors
        are reported verbatim.
        """
        for t_name in self.order:
            team = self.teams[t_name]
            client = self.clients[t_name]
            # Compose a final summary of the roster for context
            roster_summary = ", ".join([f"{p} ({pos})/${price}" for p, pos, price in team.roster]) or "(empty)"
            sys_msg = AGENT_SYSTEM_TMPL.format(TEAM=team.name, BUDGET=team.budget, MINBID=self.min_bid, INC=self.inc, PERSONA=team.persona)
            user_msg = (
                "The draft is complete. Your final roster is: " + roster_summary + ". "
                "Please provide a closing message to the group: restate your hockey bro persona, briefly outline the key points of your draft strategy, and sign off with a friendly goodbye. "
                "Keep it light, like you're leaving the locker room after a fun night."
            )
            try:
                reply = client.chat(team.model, sys_msg, user_msg).strip()
            except Exception as e:
                reply = f"(wrapup_error) {e}"
            if reply:
                self.add_msg(t_name, reply)

    def update_mode(self) -> None:
        """Run a mid‑season update conversation among the GMs.

        This mode is intended to be invoked separately from the draft.  It
        attempts to reconstruct each team's roster from ``draft_results.csv`` (if
        present) and then invites every GM to chat about how their picks have
        performed to date, which moves were smart, and which were questionable.
        The tone should be relaxed, as if sharing beers in the locker room.  The
        method logs all messages to the chat log and concludes when every GM
        has spoken once.
        """
        # Try to load rosters from draft_results.csv.  If not present, use
        # whatever rosters exist on this controller instance (perhaps from a
        # previous run in the same process).
        results_path = None
        if self.catalogue.csv_path:
            path = pathlib.Path(self.catalogue.csv_path).with_name("draft_results.csv")
            if path.exists():
                results_path = path
        if results_path:
            # Reset any existing rosters
            for tm in self.teams.values():
                tm.roster.clear()
            # Build rosters from results file
            with open(results_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("Name") or "").strip()
                    pos = (row.get("Pos") or "W").strip()
                    drafted_by = row.get("DraftedBy") or ""
                    price_str = row.get("Price") or ""
                    try:
                        price = int(price_str) if price_str else 0
                    except Exception:
                        price = 0
                    if drafted_by and name:
                        if drafted_by in self.teams:
                            self.teams[drafted_by].roster.append((name, pos, price))
        # Mid‑season chat: each team reflects on its picks with current point totals.
        self.add_msg("AUCTIONEER", "Mid‑season update! Grab a beer and tell the boys how your picks are doing.")
        # Attempt to load current points from the players CSV.  We look for a
        # column called 'PTS (2024-25)' or compute from G and A columns.  If
        # parsing fails, default to zero.
        points: Dict[str, float] = {}
        if self.catalogue.csv_path and pathlib.Path(self.catalogue.csv_path).exists():
            try:
                with open(self.catalogue.csv_path, "r", encoding="utf-8") as pf:
                    reader = csv.DictReader(pf)
                    for row in reader:
                        name = (row.get("Name") or "").strip()
                        pts_str = row.get("PTS (2024-25)") or row.get("PTS") or ""
                        val: float = 0.0
                        if pts_str:
                            try:
                                val = float(pts_str)
                            except Exception:
                                pass
                        else:
                            g_str = row.get("G (2024-25)") or row.get("G") or ""
                            a_str = row.get("A (2024-25)") or row.get("A") or ""
                            try:
                                val = float(g_str) + float(a_str)
                            except Exception:
                                val = 0.0
                        if name:
                            points[name] = val
            except Exception:
                pass
        for t_name in self.order:
            team = self.teams[t_name]
            client = self.clients[t_name]
            # Build a summary of each player's current points
            player_pts: List[str] = []
            total_pts = 0.0
            for p, pos, price in team.roster:
                val = points.get(p, 0.0)
                total_pts += val
                player_pts.append(f"{p}: {val:.1f} pts")
            roster_summary = ", ".join([f"{p} ({pos})/${price}" for p, pos, price in team.roster]) or "(empty)"
            pts_summary = "; ".join(player_pts) or "(no players)"
            sys_msg = AGENT_SYSTEM_TMPL.format(
                TEAM=team.name,
                BUDGET=team.budget,
                MINBID=self.min_bid,
                INC=self.inc,
                PERSONA=team.persona,
                OPPONENTS=", ".join([f"{nm} [{self.teams[nm].model}]" for nm in self.order if nm != t_name])
            )
            user_msg = (
                "It's mid‑season. Here's how your players are doing so far: " + pts_summary + ". "
                "Total team points: " + f"{total_pts:.1f}. "
                "Discuss which picks have been outperforming expectations, which were questionable, and what you might do differently next time. "
                "Speak casually as if you're sharing a few beers with the other GMs. "
                "Your current roster: " + roster_summary + "."
            )
            try:
                reply = client.chat(team.model, sys_msg, user_msg).strip()
            except Exception as e:
                reply = f"(update_error) {e}"
            if reply:
                self.add_msg(t_name, reply)
        self.add_msg("AUCTIONEER", "That's all for now, boys. See you back on the ice!")


###############################################################################
# HTML template
###############################################################################

# Basic tailwind‑like styling for the live chat view.  The template is inserted
# into ``index.html`` when ``--html`` is used.
HTML_TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8"><meta name=viewport content="width=device-width,initial-scale=1">
<title>GDS AI Hockey Auction — Live Chat</title>
<style>
 body{background:#121212;display:flex;justify-content:center;padding:20px;margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial}
 .phone{background:#f5f5f5;border:1px solid #ccc;border-radius:24px;max-width:380px;width:100%;height:90vh;display:flex;flex-direction:column;overflow:hidden}
 .header{background:#f5f5f5;padding:10px;font-weight:bold;text-align:center;border-bottom:1px solid #ddd}
 .chat{flex:1;padding:10px;overflow-y:auto;background:#e0e0e0}
 .bubble{padding:8px 12px;border-radius:16px;margin-bottom:8px;max-width:85%;word-wrap:break-word;color:#fff}
 .bubble .name{font-weight:bold;margin-bottom:4px;font-size:0.8rem}
 .state{background:#f0f0f0;border:1px solid #ccc;border-radius:10px;margin-bottom:12px;padding:10px;font-size:0.75rem;color:#000}
</style>
</head><body>
<div class="phone">
  <div class="header">Game Day Suits — Live Draft</div>
  <div id="chat" class="chat"></div>
</div>
<script>
 const chatDiv = document.getElementById('chat');
 const colours = {};
 const palette = ["#0b93f6","#4CAF50","#FF9800","#9C27B0","#795548","#3F51B5","#009688","#607D8B","#F44336","#8BC34A"];
 function getColour(name){
   if(!colours[name]){ colours[name] = palette[Object.keys(colours).length % palette.length]; }
   return colours[name];
 }
 let n=0;
 async function tick(){
   try{
     const r=await fetch('chat_log.json?_=' + Date.now());
     if(!r.ok) return;
     const data=await r.json();
     if(data.length===n) return;
     n=data.length;
     chatDiv.innerHTML='';
     data.forEach(m=>{
       if(m.type==='state'){
         const div=document.createElement('div');
         div.className='state';
         // Build player label without parentheses if position is empty
         const playerLabel = m.pos ? m.player + ' (' + m.pos + ')' : m.player;
         div.innerHTML = '<div><strong>PLAYER:</strong> '+playerLabel+'</div>'+
           '<div><strong>HIGH BID:</strong> $'+m.high+' by '+(m.high_bidder||'—')+'</div>'+
           '<div style="margin-top:4px;"><strong>Budgets & Rosters</strong></div>'+
           '<div>'+m.summary.replace(/\n/g,'<br>')+'</div>';
         chatDiv.appendChild(div);
       } else {
         const msg=document.createElement('div');
         msg.className='bubble';
         const col=getColour(m.team);
         msg.style.background=col;
         // determine text colour based on luminance
         const num=parseInt(col.replace('#',''),16);
         const r=(num>>16)&255,g=(num>>8)&255,b=num&255;
         const luminance = 0.2126*r + 0.7152*g + 0.0722*b;
         msg.style.color = luminance > 140 ? '#000' : '#fff';
         msg.innerHTML = '<div class="name">'+m.team+'</div><div>'+m.text+'</div>';
         chatDiv.appendChild(msg);
       }
     });
     chatDiv.scrollTop = chatDiv.scrollHeight;
   }catch(e){}
 }
 setInterval(tick,700); tick();
</script>
</body></html>
"""


###############################################################################
# Configuration loading and CLI
###############################################################################

CONFIG_EXAMPLE = """
# config.yaml — example configuration for GDS hockey draft
min_bid: 10
increment: 10
budget: 1000
roster_size: 10
seed: "Evan Bouchard (D)"
players_csv: "Game Day Suits Players.csv"
teams:
  - name: Grok
    api_key: "${OPENROUTER_API_KEY}"
    model: "x-ai/grok-4"
    persona: "Bold and brash GM; happy to overpay to set the tone."
  - name: ChatGPT
    api_key: "${OPENROUTER_API_KEY}"
    model: "openai/gpt-4o-mini"
    persona: "Balanced, pragmatic roster builder with sly misdirection."
  - name: Gemini
    api_key: "${OPENROUTER_API_KEY}"
    model: "google/gemini-2.5-flash"
    persona: "Methodical value hunter; lets others overspend first."
  - name: Claude
    api_key: "${OPENROUTER_API_KEY}"
    model: "anthropic/claude-3.5-sonnet"
    persona: "Analytical, chirpy, plays cap math like a violin."
  - name: Perplexity
    api_key: "${OPENROUTER_API_KEY}"
    model: "perplexity/sonar"
    persona: "Scout report machine—short, sharp, data-heavy chirps."
  - name: DeepSeek
    api_key: "${OPENROUTER_API_KEY}"
    model: "deepseek/deepseek-chat-v3.1"
    persona: "Bargain sniper; loves sleeper picks and late steals."
"""


def load_config(path: str) -> dict:
    """Load a YAML config file and expand ``${ENV_VAR}`` placeholders."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Replace ${VAR} with environment variables if present
    raw = re.sub(r"\$\{([A-Z0-9_]+)\}", lambda m: os.getenv(m.group(1), ""), raw)
    cfg = yaml.safe_load(raw)
    # Assign default budgets to teams if not explicitly set
    default_budget = cfg.get("budget", BUDGET)
    for t in cfg.get("teams", []):
        t.setdefault("budget", default_budget)
    return cfg


###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GDS — AI Hockey Auction Draft (rewritten)")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--html", action="store_true", help="Launch live HTML chat interface")
    parser.add_argument("--update", action="store_true", help="Run mid‑season update mode instead of the draft")
    args = parser.parse_args()
    # If no config is present, print an example and exit
    if not os.path.exists(args.config):
        print("No config.yaml found. Create one next to this script. Example:\n\n" + CONFIG_EXAMPLE)
        raise SystemExit(1)
    cfg = load_config(args.config)
    controller = DraftController(cfg, use_html=args.html)
    if args.update:
        controller.update_mode()
    else:
        controller.run()