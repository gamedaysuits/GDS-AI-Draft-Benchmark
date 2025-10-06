# GDS-AI-Draft-Benchmark
Game Day Suits – AI Hockey Auction Draft

Game Day Suits is an AI-driven fantasy hockey auction draft benchmark.
Ten large-language models (LLMs) act as general managers (GMs), each with its own personality, strategy, and reasoning style. They compete to assemble the best possible NHL roster under a shared salary cap — chirping, bluffing, and negotiating like real hockey GMs.

This project serves both as an LLM reasoning benchmark and a showcase of multi-agent interaction in a structured, competitive environment.

Features

Real-time multi-model competition – Each GM runs on a distinct LLM endpoint (ChatGPT, Claude, DeepSeek, etc.).

True auction dynamics – Enforces salary caps, bid increments, and roster limits.

Robust player validation – Nominations checked against a CSV roster of available NHL players.

Persistent results – Outputs a draft_results.csv summarizing each player, price, and winning model.

Live HTML interface – Optional --html flag spins up a local dashboard at http://127.0.0.1:8777
 with a scrolling chat and state updates. note: not functioning

No “fake” delays – All models are invoked in real time with stable OpenRouter API calls.

Project Structure
├── gds_ai_hockey_draft.py       # Main draft controller and logic
├── config.yaml                  # Team, model, and auction settings
├── Game Day Suits Players.csv   # Player list with projected points
├── chat_log.json                # Live chat transcript (generated)
├── draft_results.csv            # Final results (generated)
└── README.md                    # This file

Setup
1. Clone the repository
git clone https://github.com/<your-username>/game-day-suits-ai-draft.git
cd game-day-suits-ai-draft

2. Install dependencies
pip install -r requirements.txt


If no requirements.txt is provided, install the basics manually:

pip install requests pyyaml pillow

3. Set your API key

This project uses OpenRouter to access multiple models.
Set your key in your shell before running:

export OPENROUTER_API_KEY="your_openrouter_api_key_here"

Running the Draft

python3 -m venv venv
source venv/bin/activate
pip install -U rich pyyaml requests
export OPENROUTER_API_KEY=="your_openrouter_api_key_here"
python3 gds_ai_hockey_draft.py --config config.yaml --html



Configuration

Edit config.yaml to define:

teams: Each GM, its model slug, and persona.

players_csv: Player list (any CSV with columns Name and Pos. with or without some additional statistical data e.g. Corsi, etc.).

budget, increment, roster_size, seed

Example team configuration:

teams:
  - name: DeepSeek
    model: "deepseek/deepseek-chat-v3.1"
    
  - name: Claude
    model: "anthropic/claude-3.5-sonnet"
    
Optionally a seed persona may be supplied by adding "persona: "XXXX"", but models are told to choose their own persona in the initial prompt.
🧠 How It Works

Preflight Verification – Each model is pinged once to confirm it’s running under the correct slug (no silent fallbacks).

Planning Phase – Every GM defines a private strategy and persona before bidding starts.

Auction Loop – Players are nominated, bids are placed using BID: $NNN, and results are logged.

Persistence – Draft results and transcripts are written to disk after completion.

🖥️ Output Files

chat_log.json – Full structured chat history for replay.

draft_transcript.txt – Human-readable transcript.

draft_results.csv – Final team rosters and prices.

🤝 Contributing

Pull requests are welcome!
If you add support for new models or visualization features, please include:

Updated config.yaml examples

A note in this README

🧾 License

This project is released under the MIT License.
Use, modify, and distribute freely — attribution appreciated.

🧊 Acknowledgements

Built for Game Day Suits, a Canadian-founded bespoke menswear brand that believes in bringing hockey-locker energy to businesswear.  Somehow we are also now a software company?  Do suit companies usually have githubs?
“Suits are for game day — whatever your game day looks like.”
