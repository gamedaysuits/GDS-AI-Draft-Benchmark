#!/usr/bin/env python3
"""
GDS AI Playoff Draft 2026 — Entry Point

Usage:
    python run.py                  # Full draft (all 3 phases)
    python run.py --phase1-only    # Just identity/scouting
    python run.py --mini           # Mini draft (2 rounds, 3 models) for testing
"""

import os
import sys
import yaml
import argparse
from dotenv import load_dotenv

from src.draft_state import DraftState
from src.api_client import APIClient
from src.orchestrator import Orchestrator
from src.logger import DraftLogger


def load_config(path: str = "config.yaml") -> dict:
    """Load and validate the config file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    print(f"  📋 Config loaded: {len(config['models'])} models, "
          f"{config['draft']['rounds']} rounds")
    return config


def main():
    parser = argparse.ArgumentParser(description="GDS AI Playoff Draft 2026")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Run only Phase 1 (identity/scouting)")
    parser.add_argument("--mini", action="store_true",
                        help="Mini draft for testing (2 rounds, 3 models)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--players", default="data/playoff_players.csv",
                        help="Path to player CSV")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  🏒 GDS AI PLAYOFF DRAFT 2026")
    print("=" * 60)

    # Load .env file if present (for API keys)
    load_dotenv()

    # Load config
    config = load_config(args.config)

    # Get API key from environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("  ❌ OPENROUTER_API_KEY not set!")
        sys.exit(1)

    # Mini mode: reduce to 3 models and 2 rounds for testing
    if args.mini:
        print("  🧪 MINI MODE — 3 models, 2 rounds")
        model_names = list(config["models"].keys())[:3]
        config["models"] = {k: config["models"][k] for k in model_names}
        config["draft"]["rounds"] = 2
        # Adjust roster format for 2 rounds
        config["draft"]["roster_format"] = {"F": 1, "D": 1, "G": 0}

    # Initialize components
    draft_state = DraftState(config)
    api_client = APIClient(config, api_key)

    # Determine output directory
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        config["output"]["dir"],
    )
    logger = DraftLogger(config, output_dir)

    # Load player pool
    players_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.players,
    )
    print(f"\n  📂 Loading players from: {players_path}")
    draft_state.load_players(players_path)

    # Set draft order (defending champ first, rest randomized)
    defending_champ = config.get("defending_champion", "Grok")
    if defending_champ not in draft_state.teams:
        # If mini mode excluded the champ, just randomize everything
        defending_champ = list(draft_state.teams.keys())[0]
    draft_state.set_draft_order(defending_champ)
    logger.log_draft_order(draft_state.base_order, draft_state.teams)

    # Create orchestrator
    orchestrator = Orchestrator(config, draft_state, api_client, logger)

    # ── Execute ───────────────────────────────────────────────
    # Phase 1: Identity & Scouting
    orchestrator.run_phase_1()

    if args.phase1_only:
        print("\n  ⏹️ Phase 1 only mode — stopping here.")
        logger.save(api_client)
        return

    # Phase 2: Snake Draft
    orchestrator.run_phase_2()

    # Phase 3: Closing Statements
    orchestrator.run_phase_3()

    # ── Save Everything ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  💾 SAVING OUTPUT")
    print("=" * 60)

    logger.save(api_client)
    logger.save_results_csv(draft_state.teams)

    # Print final summary
    print("\n" + "=" * 60)
    print("  🏆 DRAFT COMPLETE")
    print("=" * 60)

    usage = api_client.get_usage_summary()
    print(f"  Total API calls: {usage['total_calls']}")
    print(f"  Total tokens: {usage['total_input_tokens']:,} in / "
          f"{usage['total_output_tokens']:,} out")
    if usage["total_reasoning_tokens"]:
        print(f"  Reasoning tokens: {usage['total_reasoning_tokens']:,}")

    # Per-model breakdown for cost reporting
    if usage.get("per_model"):
        print(f"\n  📊 PER-MODEL TOKEN USAGE:")
        print(f"  {'Model':<45} {'Calls':>6} {'Input':>10} {'Output':>10} {'Reasoning':>10}")
        print(f"  {'─' * 85}")
        for model_slug, m_usage in sorted(usage["per_model"].items()):
            reasoning_str = f"{m_usage['reasoning_tokens']:,}" if m_usage['reasoning_tokens'] else "—"
            print(f"  {model_slug:<45} {m_usage['calls']:>6} "
                  f"{m_usage['input_tokens']:>10,} {m_usage['output_tokens']:>10,} "
                  f"{reasoning_str:>10}")
        print(f"  {'─' * 85}")

    print(f"\n  📄 JSON log:  {logger.json_path}")
    print(f"  📝 Markdown:  {logger.md_path}")
    print(f"  📊 Results:   {logger.csv_path}")
    print(f"  🎭 Personas:  {logger.personas_path}")
    print()


if __name__ == "__main__":
    main()
