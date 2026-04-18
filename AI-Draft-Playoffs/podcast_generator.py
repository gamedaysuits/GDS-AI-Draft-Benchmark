#!/usr/bin/env python3
"""
GDS AI Playoff Draft — Podcast Generator

Converts the draft_log.md and personas.json into a multi-voice podcast
using ElevenLabs TTS. Each AI GM speaks with a custom voice generated
from the voice_description they authored during Phase 1.

Pipeline stages:
  1. --voices     List existing ElevenLabs voices
  2. --generate   Generate custom voices from GM descriptions
  3. --parse      Dry-run: parse draft log, show segments
  4. --test GM    Generate a test clip for one GM
  5. --render     Render all TTS clips (cached)
  6. --stitch     Stitch clips into final podcast MP3
  7. --all        Run generate → render → stitch

Usage:
  python3 podcast_generator.py --voices
  python3 podcast_generator.py --generate
  python3 podcast_generator.py --test "Grok"
  python3 podcast_generator.py --render
  python3 podcast_generator.py --stitch
  python3 podcast_generator.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

# ─── Load environment ───────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
if not ELEVENLABS_API_KEY:
    print("❌ ELEVENLABS_API_KEY not found in .env")
    sys.exit(1)

# ─── Paths ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "output"
PERSONAS_PATH = OUTPUT_DIR / "personas.json"
DRAFT_LOG_PATH = OUTPUT_DIR / "draft_log.md"
VOICE_CONFIG_PATH = PROJECT_DIR / "voice_config.json"
CLIPS_DIR = PROJECT_DIR / "podcast_clips"
PODCAST_OUTPUT = PROJECT_DIR / "gds_playoff_draft_podcast.mp3"


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SpeechSegment:
    """One unit of speech in the podcast."""
    index: int                  # Global ordering (001, 002, ...)
    speaker: str                # GM name from personas.json (e.g. "Grok")
    nickname: str               # GM's chosen nickname (e.g. "Stats Czar")
    text: str                   # The actual speech content, cleaned for TTS
    segment_type: str           # "pick", "reaction", "closing", "narrator"
    pick_number: int = 0        # Which pick this relates to (0 for closings)
    round_number: int = 0       # Which round


# ═══════════════════════════════════════════════════════════════════
# STAGE 1: VOICE DISCOVERY
# ═══════════════════════════════════════════════════════════════════

def list_voices():
    """List all available ElevenLabs voices with their details."""
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    response = client.voices.get_all()

    print("\n" + "=" * 70)
    print("  AVAILABLE ELEVENLABS VOICES")
    print("=" * 70)

    voices = response.voices
    print(f"\n  Found {len(voices)} voices:\n")

    for v in sorted(voices, key=lambda x: x.name):
        # Build a description from labels
        labels = v.labels or {}
        desc_parts = []
        if labels.get("accent"):
            desc_parts.append(labels["accent"])
        if labels.get("gender"):
            desc_parts.append(labels["gender"])
        if labels.get("age"):
            desc_parts.append(labels["age"])
        if labels.get("description"):
            desc_parts.append(labels["description"])
        if labels.get("use_case"):
            desc_parts.append(f"[{labels['use_case']}]")

        desc = ", ".join(desc_parts) if desc_parts else "(no labels)"
        print(f"  {v.voice_id[:12]}...  {v.name:<25} {desc}")

    print(f"\n  Total: {len(voices)} voices")
    return voices


# ═══════════════════════════════════════════════════════════════════
# STAGE 2: VOICE GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_voices():
    """
    Generate a custom ElevenLabs voice for each GM using the Voice Design API.

    Each AI GM authored their own voice_description during persona creation.
    We feed that description directly into ElevenLabs' voice generation to
    create a unique, permanent voice for each character.

    Flow per GM:
      1. Call create_previews(voice_description=...) → 3 preview options
      2. Save preview audio locally for review
      3. Save the first preview as a permanent voice via create()
      4. Store the resulting voice_id in voice_config.json

    Voices are generated once. If voice_config.json already has a voice_id
    for a GM, that GM is skipped (idempotent / cost-safe).
    """
    from elevenlabs import ElevenLabs
    import base64

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # Load personas
    with open(PERSONAS_PATH) as f:
        personas = json.load(f)

    # Load existing config if present (for incremental runs)
    existing_config = {}
    if VOICE_CONFIG_PATH.exists():
        with open(VOICE_CONFIG_PATH) as f:
            existing_config = json.load(f)

    print("\n" + "=" * 70)
    print("  VOICE GENERATION (ElevenLabs Voice Design)")
    print("=" * 70)

    # Directory for preview audio samples
    previews_dir = PROJECT_DIR / "voice_previews"
    previews_dir.mkdir(exist_ok=True)

    voice_config = dict(existing_config)
    generated = 0
    skipped = 0

    for gm_name, persona in personas.items():
        nickname = persona.get("nickname", gm_name)
        raw_desc = persona.get("voice_description", "")

        print(f"\n  🎭 {nickname} ({gm_name})")
        print(f"     Description: \"{raw_desc}\"")

        # ── Skip if already generated ──
        if gm_name in existing_config and existing_config[gm_name].get("voice_id"):
            print(f"     ⏭️  Already generated — voice_id: "
                  f"{existing_config[gm_name]['voice_id'][:16]}...")
            skipped += 1
            continue

        # ── Format the description for Voice Design API ──
        # The raw description from the LLM is already detailed and specific.
        # We just append studio context for audio quality.
        design_prompt = (
            f"{raw_desc} "
            f"Studio quality audio. Sports podcast personality."
        )

        print(f"     🔧 Design prompt: \"{design_prompt}\"")
        print(f"     ⏳ Generating previews...")

        try:
            # Generate 3 voice previews from the description
            preview_response = client.text_to_voice.create_previews(
                voice_description=design_prompt,
                text=(
                    f"Welcome to the GDS AI Playoff Draft! I'm {nickname}, "
                    f"and I'm about to show you how a real GM builds a roster. "
                    f"Let's get this started."
                ),
            )

            previews = preview_response.previews
            print(f"     ✅ Got {len(previews)} previews "
                  f"({previews[0].duration_secs:.1f}s each)")

            # Save all preview audio files for manual review
            for idx, preview in enumerate(previews):
                safe_name = re.sub(r'[^a-z0-9]+', '_', gm_name.lower())
                preview_path = previews_dir / f"{safe_name}_v{idx+1}.mp3"
                audio_bytes = base64.b64decode(preview.audio_base_64)
                with open(preview_path, "wb") as af:
                    af.write(audio_bytes)
                print(f"     📁 Preview {idx+1}: {preview_path.name} "
                      f"({preview.duration_secs:.1f}s)")

            # ── Save the first preview as a permanent voice ──
            # The user can review previews and re-run with a different
            # selection later if needed.
            chosen = previews[0]
            voice_name = f"GDS Draft - {nickname}"

            print(f"     💾 Saving voice as \"{voice_name}\"...")
            saved_voice = client.text_to_voice.create(
                voice_name=voice_name,
                voice_description=raw_desc,
                generated_voice_id=chosen.generated_voice_id,
                labels={
                    "project": "GDS AI Draft",
                    "character": nickname,
                    "model": gm_name,
                },
            )

            voice_config[gm_name] = {
                "nickname": nickname,
                "voice_id": saved_voice.voice_id,
                "voice_name": voice_name,
                "voice_description": raw_desc,
                "generated_voice_id": chosen.generated_voice_id,
            }
            generated += 1

            print(f"     ✅ Voice created: {saved_voice.voice_id}")

            # Save config after each voice (in case of mid-run failure)
            with open(VOICE_CONFIG_PATH, "w") as f:
                json.dump(voice_config, f, indent=2)

            # Polite rate limiting between API calls
            time.sleep(1.0)

        except Exception as e:
            print(f"     ❌ Failed: {e}")
            print(f"        Skipping {gm_name} — re-run to retry")
            continue

    # Final save
    with open(VOICE_CONFIG_PATH, "w") as f:
        json.dump(voice_config, f, indent=2)

    print(f"\n  💾 Voice config saved to: {VOICE_CONFIG_PATH}")
    print(f"     Generated: {generated} new voices")
    print(f"     Skipped (existing): {skipped}")
    print(f"     Total: {len(voice_config)} voices configured")
    print(f"\n  🎧 Preview audio saved to: {previews_dir}/")
    print("  ℹ️  Listen to previews, then edit voice_config.json to swap if needed.\n")
    return voice_config


# ═══════════════════════════════════════════════════════════════════
# STAGE 3: DRAFT LOG PARSER
# ═══════════════════════════════════════════════════════════════════

def parse_draft_log() -> list[SpeechSegment]:
    """
    Parse the draft_log.md into ordered speech segments.

    Extracts:
      - Pick chirps (the quoted text under each ### Pick N)
      - Reactions (the blockquoted responses from other GMs)
      - Round headers (for narrator segments)

    Does NOT extract:
      - The "Meet the GMs" section (persona intros — not speech)
      - Draft stats section
      - Draft order section
    """
    with open(PERSONAS_PATH) as f:
        personas = json.load(f)

    # Build a nickname → GM name lookup
    nick_to_gm: dict[str, str] = {}
    for gm_name, data in personas.items():
        nickname = data.get("nickname", gm_name)
        nick_to_gm[nickname] = gm_name

    with open(DRAFT_LOG_PATH) as f:
        content = f.read()

    segments: list[SpeechSegment] = []
    segment_index = 0

    # Split into lines for sequential parsing
    lines = content.split("\n")
    i = 0
    current_round = 0
    current_pick = 0
    current_drafter_nick = ""
    current_drafter_gm = ""

    # Skip everything before "## Round 1"
    while i < len(lines):
        if lines[i].strip().startswith("## Round "):
            break
        i += 1

    while i < len(lines):
        line = lines[i].strip()

        # ── Round header ──
        # "## Round N"
        round_match = re.match(r"^## Round (\d+)", line)
        if round_match:
            current_round = int(round_match.group(1))
            i += 1
            continue

        # ── Pick header ──
        # "### Pick N — Nickname"
        pick_match = re.match(r"^### Pick (\d+) — (.+)$", line)
        if pick_match:
            current_pick = int(pick_match.group(1))
            current_drafter_nick = pick_match.group(2).strip()
            current_drafter_gm = nick_to_gm.get(current_drafter_nick, current_drafter_nick)

            # The pick chirp follows in a blockquote block.
            # Pattern: > **Player Name** (TEAM / POS) — STATS
            #          >
            #          > *"chirp text"*
            i += 1
            chirp_text = _extract_pick_chirp(lines, i)

            if chirp_text:
                segment_index += 1
                segments.append(SpeechSegment(
                    index=segment_index,
                    speaker=current_drafter_gm,
                    nickname=current_drafter_nick,
                    text=_clean_for_tts(chirp_text),
                    segment_type="pick",
                    pick_number=current_pick,
                    round_number=current_round,
                ))
                # Advance past the blockquote
                while i < len(lines) and (lines[i].strip().startswith(">") or not lines[i].strip()):
                    i += 1
            continue

        # ── Reactions block ──
        # "**Reactions:**"
        if line == "**Reactions:**":
            i += 1
            # Each reaction is: > **Nickname:** "text"
            while i < len(lines):
                rline = lines[i].strip()
                if not rline.startswith(">"):
                    break

                reaction_match = re.match(
                    r'>\s*\*\*(.+?):\*\*\s*["\u201c]?(.*)',
                    rline
                )
                if reaction_match:
                    reactor_nick = reaction_match.group(1).strip()
                    reactor_gm = nick_to_gm.get(reactor_nick, reactor_nick)
                    reaction_text = reaction_match.group(2).strip()

                    # Reaction might continue on next lines (still blockquoted)
                    i += 1
                    while i < len(lines) and lines[i].strip().startswith(">"):
                        # Check if this is a new reactor (new > **Name:**)
                        if re.match(r'>\s*\*\*(.+?):\*\*', lines[i].strip()):
                            break
                        # Continuation of current reaction
                        cont = lines[i].strip().lstrip(">").strip()
                        reaction_text += " " + cont
                        i += 1

                    # Clean trailing quote mark
                    reaction_text = reaction_text.rstrip('"').rstrip('\u201d').strip()

                    if reaction_text:
                        segment_index += 1
                        segments.append(SpeechSegment(
                            index=segment_index,
                            speaker=reactor_gm,
                            nickname=reactor_nick,
                            text=_clean_for_tts(reaction_text),
                            segment_type="reaction",
                            pick_number=current_pick,
                            round_number=current_round,
                        ))
                else:
                    i += 1
            continue

        # ── Draft Stats — skip past these running checkpoints ──
        # The draft log contains stats snapshots after each round.
        # We skip the entire block until the next "## Round" or end-of-file.
        if line.startswith("## Draft Stats"):
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith("## Round"):
                    break  # Let the outer loop handle the new round
                i += 1
            continue

        # ── Closing statements ──
        # These appear after Phase 3 if the draft completes.
        # Format: ### Closing — Nickname
        closing_match = re.match(r"^### Closing — (.+)$", line)
        if closing_match:
            closer_nick = closing_match.group(1).strip()
            closer_gm = nick_to_gm.get(closer_nick, closer_nick)
            i += 1
            closing_text = _extract_blockquote(lines, i)
            if closing_text:
                segment_index += 1
                segments.append(SpeechSegment(
                    index=segment_index,
                    speaker=closer_gm,
                    nickname=closer_nick,
                    text=_clean_for_tts(closing_text),
                    segment_type="closing",
                ))
                while i < len(lines) and (lines[i].strip().startswith(">") or not lines[i].strip()):
                    i += 1
            continue

        i += 1

    print(f"\n  📝 Parsed {len(segments)} speech segments from draft log")
    print(f"     Picks: {sum(1 for s in segments if s.segment_type == 'pick')}")
    print(f"     Reactions: {sum(1 for s in segments if s.segment_type == 'reaction')}")
    print(f"     Closings: {sum(1 for s in segments if s.segment_type == 'closing')}")

    # Character count estimate
    total_chars = sum(len(s.text) for s in segments)
    print(f"     Total characters for TTS: ~{total_chars:,}")

    return segments


def _extract_pick_chirp(lines: list[str], start: int) -> str:
    """
    Extract the pick chirp text from blockquoted lines starting at `start`.

    The pick block looks like:
      > **Player Name** (TEAM / POS) — stats
      >
      > *"chirp text that may span multiple lines"*

    We want just the chirp text (the italicized/quoted part), not the player info.
    """
    chirp_parts = []
    in_chirp = False
    i = start

    while i < len(lines):
        rline = lines[i].strip()

        # End of blockquote block
        if not rline.startswith(">") and rline and not rline == "":
            break
        if not rline:
            if in_chirp:
                break
            i += 1
            continue

        # Strip the blockquote marker
        text = rline.lstrip(">").strip()

        # Skip the player info line (starts with **Player Name**)
        if text.startswith("**") and ("/" in text) and ("—" in text or "–" in text):
            i += 1
            continue

        # Skip empty blockquote lines
        if not text:
            i += 1
            continue

        # The chirp line — usually wrapped in *"..."*
        if text.startswith('*"') or text.startswith("*'") or text.startswith('*\u201c'):
            in_chirp = True
            # Strip the italic/quote markers
            text = text.strip("*").strip('"').strip("'").strip('\u201c').strip('\u201d')
            chirp_parts.append(text)
        elif in_chirp:
            # Continuation of multi-line chirp
            text = text.strip("*").strip('"').strip("'").strip('\u201c').strip('\u201d')
            chirp_parts.append(text)
        else:
            # Fallback: treat as chirp text even if format is unexpected
            # (some models don't use the exact *"..."* format)
            if not text.startswith("**"):
                in_chirp = True
                text = text.strip("*").strip('"').strip("'")
                chirp_parts.append(text)

        i += 1

    return " ".join(chirp_parts).strip()


def _extract_blockquote(lines: list[str], start: int) -> str:
    """Extract all blockquoted text starting from `start`."""
    parts = []
    i = start
    while i < len(lines):
        rline = lines[i].strip()
        if rline.startswith(">"):
            text = rline.lstrip(">").strip()
            parts.append(text)
        elif rline:
            break
        i += 1
    return " ".join(parts).strip()


def _clean_for_tts(text: str) -> str:
    """
    Clean markdown/formatting artifacts for cleaner TTS output.
    Preserves the actual speech content but removes formatting noise.
    """
    # Remove markdown bold (**text**)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # Remove markdown italic (*text*)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # Remove markdown links [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove citation markers like [1], [3]
    text = re.sub(r'\[\d+\]', '', text)
    # Remove hashtags (#MoneyMistake → MoneyMistake)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove emoji (common ones in the draft log)
    text = re.sub(r'[🏒🏆⚡️💨❄️🎤🎙️💬🔧✅❌⚠️🔄]', '', text)
    # Normalize quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Clean up dashes — use natural pause
    text = text.replace(' — ', ', ')
    text = text.replace('—', ', ')

    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# STAGE 4: TTS RENDERING
# ═══════════════════════════════════════════════════════════════════

def render_test_clip(gm_name: str):
    """Generate a single test clip for one GM to verify voice quality."""
    voice_config = _load_voice_config()

    if gm_name not in voice_config:
        # Try matching by nickname
        for key, val in voice_config.items():
            if val["nickname"].lower() == gm_name.lower():
                gm_name = key
                break
        else:
            print(f"  ❌ GM '{gm_name}' not found in voice config.")
            print(f"     Available: {', '.join(voice_config.keys())}")
            return

    cfg = voice_config[gm_name]
    print(f"\n  🎙️ Test clip for {cfg['nickname']} ({gm_name})")
    print(f"     Voice: {cfg['voice_name']} ({cfg['voice_id'][:12]}...)")

    # Generate a test line that captures the persona
    with open(PERSONAS_PATH) as f:
        personas = json.load(f)
    persona = personas.get(gm_name, {})

    test_text = (
        f"Hey, I'm {cfg['nickname']}, and I'm here to dominate this draft. "
        f"{persona.get('persona', '')[:200]}"
    )

    CLIPS_DIR.mkdir(exist_ok=True)
    output_path = CLIPS_DIR / f"test_{gm_name.lower().replace(' ', '_')}.mp3"

    _generate_clip(cfg["voice_id"], test_text, output_path)
    print(f"  ✅ Test clip saved: {output_path}")


def render_all_clips(segments: list[SpeechSegment] | None = None):
    """
    Render all speech segments to individual MP3 clips via ElevenLabs.
    Skips clips that already exist (cache-aware for cost savings).
    """
    voice_config = _load_voice_config()

    if segments is None:
        segments = parse_draft_log()

    CLIPS_DIR.mkdir(exist_ok=True)

    print(f"\n  🎤 Rendering {len(segments)} clips...")
    total_chars = 0
    rendered = 0
    skipped = 0

    for seg in segments:
        filename = _clip_filename(seg)
        output_path = CLIPS_DIR / filename

        # Cache check — skip if clip already exists
        if output_path.exists() and output_path.stat().st_size > 0:
            skipped += 1
            continue

        # Resolve voice ID for this speaker
        cfg = voice_config.get(seg.speaker, {})
        voice_id = cfg.get("voice_id")
        if not voice_id:
            print(f"    ⚠️ No voice assigned for '{seg.speaker}' — skipping")
            continue

        # Render
        print(f"    [{seg.index:03d}] {seg.nickname} ({seg.segment_type}): "
              f"\"{seg.text[:60]}...\"")

        _generate_clip(voice_id, seg.text, output_path)
        total_chars += len(seg.text)
        rendered += 1

        # Rate limiting — be polite to the API
        time.sleep(0.5)

    print(f"\n  ✅ Rendering complete:")
    print(f"     Rendered: {rendered} clips ({total_chars:,} characters)")
    print(f"     Skipped (cached): {skipped} clips")
    print(f"     Total: {rendered + skipped} clips in {CLIPS_DIR}")


def _generate_clip(voice_id: str, text: str, output_path: Path):
    """Generate a single TTS clip using ElevenLabs API."""
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    # Use eleven_multilingual_v2 for best quality character voices
    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    # The API returns a generator — consume it into bytes
    audio_bytes = b""
    for chunk in audio_generator:
        audio_bytes += chunk

    with open(output_path, "wb") as f:
        f.write(audio_bytes)


def _clip_filename(seg: SpeechSegment) -> str:
    """
    Generate a descriptive filename for a clip.
    Format: 001_pick01_stats_czar.mp3
    """
    # Sanitize nickname for filename
    safe_nick = re.sub(r'[^a-z0-9]+', '_', seg.nickname.lower()).strip('_')
    type_prefix = seg.segment_type

    if seg.pick_number:
        return f"{seg.index:03d}_{type_prefix}{seg.pick_number:02d}_{safe_nick}.mp3"
    else:
        return f"{seg.index:03d}_{type_prefix}_{safe_nick}.mp3"


# ═══════════════════════════════════════════════════════════════════
# STAGE 5: STITCHER
# ═══════════════════════════════════════════════════════════════════

def stitch_podcast(segments: list[SpeechSegment] | None = None):
    """
    Concatenate all rendered clips into one final podcast MP3.

    Cadence rules:
      - 400ms silence between reactions to the same pick
      - 800ms silence between different picks
      - 1500ms silence between rounds
      - 1500ms silence at start and end
    """
    from pydub import AudioSegment

    if segments is None:
        segments = parse_draft_log()

    print(f"\n  🧵 Stitching {len(segments)} clips into podcast...")

    # Build silence segments
    silence_400 = AudioSegment.silent(duration=400)
    silence_800 = AudioSegment.silent(duration=800)
    silence_1500 = AudioSegment.silent(duration=1500)

    # Start with opening silence
    podcast = AudioSegment.silent(duration=1500)

    prev_pick = 0
    prev_round = 0
    clips_loaded = 0
    clips_missing = 0

    for seg in segments:
        filename = _clip_filename(seg)
        clip_path = CLIPS_DIR / filename

        if not clip_path.exists():
            print(f"    ⚠️ Missing clip: {filename}")
            clips_missing += 1
            continue

        # Load the clip
        clip = AudioSegment.from_mp3(str(clip_path))

        # Determine silence gap based on context
        if seg.round_number != prev_round and prev_round > 0:
            # New round — longer pause
            podcast += silence_1500
        elif seg.pick_number != prev_pick and prev_pick > 0:
            # New pick — medium pause
            podcast += silence_800
        elif seg.segment_type == "reaction":
            # Reaction to same pick — short pause
            podcast += silence_400
        else:
            # Default
            podcast += silence_800

        podcast += clip
        clips_loaded += 1

        prev_pick = seg.pick_number
        prev_round = seg.round_number

    # Closing silence
    podcast += silence_1500

    # Export
    duration_secs = len(podcast) / 1000
    minutes = int(duration_secs // 60)
    seconds = int(duration_secs % 60)

    print(f"\n  📊 Podcast stats:")
    print(f"     Duration: {minutes}m {seconds}s")
    print(f"     Clips loaded: {clips_loaded}")
    print(f"     Clips missing: {clips_missing}")

    podcast.export(str(PODCAST_OUTPUT), format="mp3", bitrate="128k")
    print(f"  ✅ Podcast saved: {PODCAST_OUTPUT}")
    print(f"     File size: {PODCAST_OUTPUT.stat().st_size / (1024*1024):.1f} MB")


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _load_voice_config() -> dict:
    """Load voice_config.json, or fail with a helpful message."""
    if not VOICE_CONFIG_PATH.exists():
        print("  ❌ voice_config.json not found. Run --generate first.")
        sys.exit(1)

    with open(VOICE_CONFIG_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GDS AI Playoff Draft — Podcast Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--voices", action="store_true",
                        help="List all available ElevenLabs voices")
    parser.add_argument("--generate", action="store_true",
                        help="Generate custom voices from GM voice descriptions")
    parser.add_argument("--parse", action="store_true",
                        help="Parse draft log and show segments (dry run)")
    parser.add_argument("--test", type=str, metavar="GM",
                        help="Generate a test clip for one GM (e.g. 'Grok')")
    parser.add_argument("--render", action="store_true",
                        help="Render all TTS clips (cached)")
    parser.add_argument("--stitch", action="store_true",
                        help="Stitch clips into final podcast MP3")
    parser.add_argument("--all", action="store_true",
                        help="Run match → render → stitch")

    args = parser.parse_args()

    # Default: show help if no args
    if not any([args.voices, args.generate, args.parse, args.test,
                args.render, args.stitch, args.all]):
        parser.print_help()
        return

    print("\n" + "=" * 70)
    print("  GDS AI PLAYOFF DRAFT — PODCAST GENERATOR")
    print("=" * 70)

    if args.voices:
        list_voices()

    if args.generate:
        generate_voices()

    if args.parse:
        segments = parse_draft_log()
        print("\n  Segment preview (first 10):")
        for seg in segments[:10]:
            print(f"    [{seg.index:03d}] {seg.segment_type:<10} {seg.nickname:<25} "
                  f"\"{seg.text[:80]}...\"")

    if args.test:
        render_test_clip(args.test)

    if args.render:
        render_all_clips()

    if args.stitch:
        stitch_podcast()

    if args.all:
        print("\n  ▶ Stage 1: Generating voices...")
        generate_voices()
        print("\n  ▶ Stage 2: Parsing draft log...")
        segments = parse_draft_log()
        print("\n  ▶ Stage 3: Rendering clips...")
        render_all_clips(segments)
        print("\n  ▶ Stage 4: Stitching podcast...")
        stitch_podcast(segments)

    print("\n  🏁 Done.\n")


if __name__ == "__main__":
    main()
