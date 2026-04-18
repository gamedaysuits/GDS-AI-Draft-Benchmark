from __future__ import annotations

"""
GDS AI Playoff Draft — OpenRouter API Client

Handles all communication with the OpenRouter API.
Features: retry with exponential backoff, rate limiting,
token usage tracking, and response parsing.
"""

import json
import re
import time
import threading
import requests


class APIClient:
    """
    OpenRouter API client with retry logic and token tracking.
    Thread-safe for Phase 1 parallel calls.
    """

    def __init__(self, config: dict, api_key: str):
        self.base_url = config["api"]["base_url"]
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config["api"]["referer"],
            "X-Title": config["api"]["app_title"],
        }
        self.timeout = config["api"]["timeout"]
        self.max_retries = config["api"]["max_retries"]
        self.retry_delays = config["api"]["retry_delays"]

        # Rate limiting — max concurrent requests
        self._semaphore = threading.Semaphore(config["api"]["max_concurrent"])

        # Token usage tracking — aggregate and per-model
        self._usage_lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_calls = 0
        self.model_usage = {}  # {model_slug: {input, output, reasoning, calls}}

    def call_model(self, model: str, messages: list[dict],
                   max_tokens: int = 500, temperature: float = 0.7) -> dict:
        """
        Send a chat completion request to OpenRouter.

        Args:
            model: OpenRouter model slug
            messages: List of message dicts [{role, content}]
            max_tokens: Max response tokens
            temperature: Sampling temperature

        Returns:
            dict with keys: content, model, usage, raw_response
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        with self._semaphore:
            return self._call_with_retry(payload)

    def _call_with_retry(self, payload: dict) -> dict:
        """Execute request with exponential backoff retry."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                start = time.time()
                resp = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                elapsed = time.time() - start

                if resp.status_code == 200:
                    result = self._parse_response(resp.json(), elapsed)
                    self._track_usage(result)
                    return result

                elif resp.status_code == 429:
                    # Rate limited — wait extra
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)] * 2
                    print(f"    ⏳ Rate limited, waiting {delay}s...")
                    time.sleep(delay)
                    last_error = f"HTTP 429 (rate limited)"

                elif resp.status_code >= 500:
                    # Server error — retry with backoff
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    print(f"    ⏳ Server error ({resp.status_code}), retrying in {delay}s...")
                    time.sleep(delay)
                    last_error = f"HTTP {resp.status_code}"

                else:
                    # Client error — don't retry
                    error_text = resp.text[:200]
                    raise APIError(f"HTTP {resp.status_code}: {error_text}")

            except requests.Timeout:
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                print(f"    ⏳ Timeout after {self.timeout}s, retrying in {delay}s...")
                time.sleep(delay)
                last_error = "Timeout"

            except APIError:
                raise  # Don't retry client errors

            except Exception as e:
                delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                print(f"    ⏳ Error: {str(e)[:100]}, retrying in {delay}s...")
                time.sleep(delay)
                last_error = str(e)

        raise APIError(f"Failed after {self.max_retries} attempts. Last error: {last_error}")

    def _parse_response(self, data: dict, elapsed: float) -> dict:
        """Parse OpenRouter response into a clean dict."""
        content = ""
        if "choices" in data and len(data["choices"]) > 0:
            msg = data["choices"][0].get("message", {})
            content = msg.get("content", "")

            # Some models include reasoning in a separate field
            # Check for thinking/reasoning content
            if not content and msg.get("reasoning_content"):
                content = msg["reasoning_content"]

        usage = data.get("usage", {})

        return {
            "content": content,
            "model": data.get("model", "unknown"),
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
            },
            "latency": round(elapsed, 2),
            "raw_response": data,
        }

    def _track_usage(self, result: dict):
        """Thread-safe token usage accumulation — aggregate and per-model."""
        model = result.get("model", "unknown")
        inp = result["usage"]["input_tokens"]
        out = result["usage"]["output_tokens"]
        reasoning = result["usage"].get("reasoning_tokens", 0)

        with self._usage_lock:
            # Aggregate totals
            self.total_input_tokens += inp
            self.total_output_tokens += out
            self.total_reasoning_tokens += reasoning
            self.total_calls += 1

            # Per-model tracking
            if model not in self.model_usage:
                self.model_usage[model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "calls": 0,
                }
            self.model_usage[model]["input_tokens"] += inp
            self.model_usage[model]["output_tokens"] += out
            self.model_usage[model]["reasoning_tokens"] += reasoning
            self.model_usage[model]["calls"] += 1

    def get_usage_summary(self) -> dict:
        """Return aggregate and per-model usage stats."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "per_model": dict(self.model_usage),
        }


class APIError(Exception):
    """Custom exception for API errors."""
    pass


# ═══════════════════════════════════════════════════════════════════
# Response Parsing Utilities
# ═══════════════════════════════════════════════════════════════════

def extract_json(text: str) -> dict | None:
    """
    Extract JSON from a model response. Tries multiple strategies:
    1. Direct json.loads on the full text
    2. Look for ```json ... ``` code fences
    3. Find the first { ... } block (greedy)

    Returns parsed dict or None.
    """
    if not text:
        return None

    # Strategy 1: direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3: find first { ... } block (greedy, handles nested)
    brace_start = text.find('{')
    if brace_start >= 0:
        # Find matching closing brace
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except (json.JSONDecodeError, TypeError):
                        pass
                    break

    return None


def extract_pick_from_text(text: str) -> dict:
    """
    Last-resort extraction when JSON parsing fails.
    Tries to find player name and chirp from freeform text.
    """
    result = {"pick": "", "position": "", "chirp": ""}

    # Look for quoted player name patterns
    name_match = re.search(r'"pick"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if name_match:
        result["pick"] = name_match.group(1)

    pos_match = re.search(r'"position"\s*:\s*"([FDG])"', text, re.IGNORECASE)
    if pos_match:
        result["position"] = pos_match.group(1).upper()

    chirp_match = re.search(r'"chirp"\s*:\s*"([^"]*)"', text, re.IGNORECASE)
    if chirp_match:
        result["chirp"] = chirp_match.group(1)

    return result
