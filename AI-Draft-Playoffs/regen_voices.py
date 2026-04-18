#!/usr/bin/env python3
"""
Call each of the 12 LLMs via OpenRouter to write their OWN voice descriptions.
Uses the updated prompt guidance (200-500 chars, any gender/accent/attitude).
Then updates personas.json with whatever they come back with.

Also cleans up any partially-generated voices from previous runs.
"""

import json
import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_DIR = Path(__file__).parent
PERSONAS_PATH = PROJECT_DIR / "output" / "personas.json"
VOICE_CONFIG_PATH = PROJECT_DIR / "voice_config.json"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# OpenRouter model slugs — from config.yaml
MODEL_SLUGS = {
    "Claude": "anthropic/claude-opus-4.7",
    "Gemini": "google/gemini-3.1-pro-preview",
    "Grok": "x-ai/grok-4.20",
    "Mistral": "mistralai/mistral-large-2512",
    "Llama 4": "meta-llama/llama-4-maverick",
    "DeepSeek": "deepseek/deepseek-r1",
    "GPT-5": "openai/gpt-5.4",
    "Qwen": "qwen/qwen3-235b-a22b",
    "Hermes": "nousresearch/hermes-4-405b",
    "Cohere": "cohere/command-a",
    "Perplexity": "perplexity/sonar-pro",
    "Gemma": "google/gemma-4-31b-it",
}

# The prompt we send to each LLM — they pick their own voice
VOICE_PROMPT = """\
You are {model_name}, built by {provider}. You just competed in the GDS AI Playoff \
Fantasy Hockey Draft as "{nickname}" — {persona}

Now your draft picks and trash talk will be narrated aloud on a REAL PODCAST using \
AI voice synthesis. We will GENERATE a custom synthetic voice from your description, \
so be extremely detailed and specific.

RULES:
- Write 200-500 characters describing the voice you want for yourself
- You can choose ANY type of voice — male, female, non-binary, old, young
- ANY accent on earth — British, Jamaican, Australian, Japanese, Nigerian, etc.
- ANY attitude — aggressive, calm, sarcastic, warm, menacing, playful
- ANY energy level — whisper-quiet to screaming enthusiasm
- Make it UNIQUE and memorable. 11 other AIs are also choosing voices. If you pick \
something generic like "confident male baritone" you will sound exactly like everyone else.
- The more specific and vivid, the better the voice will turn out.

GREAT EXAMPLES:
- "Elderly Scottish woman, 70s, gravelly whisper like she's sharing secrets over \
single malt. Slow, deliberate pacing with sudden bursts of sharp wit."
- "Young Caribbean man, early 20s, warm Trinidadian accent. Fast-talking with \
infectious energy, laughs mid-sentence. Sounds like a dancehall DJ who became a hockey analyst."
- "Gruff Russian woman, deadpan delivery, speaks like a KGB interrogator reading \
hockey stats. Flat affect, low alto. Pauses before punchlines."

BAD EXAMPLES:
- "Deep male voice, confident" (too generic)
- "Sports radio host" (boring, everyone picks this)
- "Baritone with Canadian accent" (this is what EVERY hockey voice sounds like)

Respond with ONLY a JSON object. Nothing else:
{{"voice_description": "your detailed, unique, vivid voice description here"}}
"""


def call_llm(model_name: str, slug: str, provider: str, nickname: str, persona: str) -> str:
    """Call one LLM via OpenRouter and get its voice description."""
    prompt = VOICE_PROMPT.format(
        model_name=model_name,
        provider=provider,
        nickname=nickname,
        persona=persona,
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://gamedaysuits.ca",
        "X-Title": "GDS AI Playoff Draft 2026",
        "Content-Type": "application/json",
    }

    payload = {
        "model": slug,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.9,  # Creative — we want wild answers
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"].strip()

    # Parse JSON from the response — handle markdown fences
    import re
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)

    # Try to extract just the voice_description value
    try:
        parsed = json.loads(content)
        return parsed.get("voice_description", content)
    except json.JSONDecodeError:
        # Fallback: regex extraction
        match = re.search(r'"voice_description"\s*:\s*"([^"]+)"', content)
        if match:
            return match.group(1)
        # Last resort: use raw content
        return content[:500]


def delete_partial_voices():
    """Delete any partially-generated voices from previous runs."""
    if not VOICE_CONFIG_PATH.exists():
        return

    with open(VOICE_CONFIG_PATH) as f:
        config = json.load(f)

    if not config:
        return

    from elevenlabs import ElevenLabs
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    print("\n  🗑️  Cleaning up partial voices from previous run...")
    for gm_name, info in config.items():
        voice_id = info.get("voice_id")
        if voice_id:
            try:
                client.voices.delete(voice_id=voice_id)
                print(f"     Deleted: {info.get('voice_name', gm_name)} ({voice_id})")
            except Exception as e:
                print(f"     ⚠️ {gm_name}: {e}")

    # Clear the config file
    with open(VOICE_CONFIG_PATH, "w") as f:
        json.dump({}, f)


def main():
    # Load existing personas
    with open(PERSONAS_PATH) as f:
        personas = json.load(f)

    # Delete any partially-generated voices
    delete_partial_voices()

    print("\n" + "=" * 70)
    print("  CALLING EACH LLM FOR VOICE DESCRIPTIONS")
    print("=" * 70)

    import yaml
    with open(PROJECT_DIR / "config.yaml") as f:
        config = yaml.safe_load(f)

    updated = 0
    for gm_name, persona_data in personas.items():
        nickname = persona_data.get("nickname", gm_name)
        persona = persona_data.get("persona", "A hockey-loving AI.")
        provider = config["models"].get(gm_name, {}).get("provider", "Unknown")
        slug = MODEL_SLUGS.get(gm_name)

        if not slug:
            print(f"\n  ⚠️ {gm_name}: No model slug found, skipping")
            continue

        print(f"\n  🤖 Calling {gm_name} ({slug})...")
        print(f"     Nickname: {nickname}")

        try:
            new_desc = call_llm(gm_name, slug, provider, nickname, persona)
            old_desc = persona_data.get("voice_description", "")

            personas[gm_name]["voice_description"] = new_desc
            updated += 1

            print(f"     ✅ Got description ({len(new_desc)} chars):")
            print(f"        \"{new_desc[:120]}...\"" if len(new_desc) > 120 else f"        \"{new_desc}\"")

        except Exception as e:
            print(f"     ❌ Failed: {e}")
            print(f"        Keeping existing description")

        # Rate limit between calls
        time.sleep(0.5)

    # Save updated personas
    with open(PERSONAS_PATH, "w") as f:
        json.dump(personas, f, indent=2)

    print(f"\n" + "=" * 70)
    print(f"  ✅ Updated {updated}/{len(personas)} voice descriptions")
    print(f"  💾 Saved to: {PERSONAS_PATH}")
    print(f"\n  Next: python3 podcast_generator.py --generate")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
