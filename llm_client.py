"""
llm_client.py
─────────────
Multi-key Groq client with round-robin key rotation.
Each API key is used in turn so no single key gets rate-limited.

SETUP
-----
1. Get free Groq API keys from https://console.groq.com/keys
2. Add them to a .env file in the same directory:
       GROQ_API_KEY_1=gsk_...
       GROQ_API_KEY_2=gsk_...
       GROQ_API_KEY_3=gsk_...
3. That's it — call_llm() and call_llm_binary() handle the rest.

LIVE GROQ MODELS (verified April 26 2026)
------------------------------------------
  "llama-3.3-70b-versatile"   128K ctx, 6K TPM/min — critics
  "openai/gpt-oss-120b"       128K ctx, 6K TPM/min — optimizer (strongest)
  "llama-3.1-8b-instant"      128K ctx, 6K TPM/min — dedup/repair (tiny calls)

RATE LIMIT STRATEGY
--------------------
  - Round-robin across N keys → N × 6K = effective TPM budget
  - Random stagger delay (0.1–0.6s) before each call prevents burst collisions
  - Exponential backoff with jitter on 429
  - Sequential critic fallback if concurrent calls get rate-limited
  - 413 (too large) → truncate to 1500 chars and retry
  - 400 with "decommissioned" → fail fast with clear error message
"""

import asyncio
import json
import os
import random
import time
from itertools import cycle
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# API KEYS — loaded from .env
# ─────────────────────────────────────────────────────────────
# Load all keys from .env — naming pattern: GROQ_API_KEY_{1|2|3}_{A-G}
# 3 keys × 7 accounts = 21 keys → 21 × 6K TPM = 126K effective TPM
GROQ_API_KEYS: list[str] = []
for _i in range(1, 4):
    for _ac in ["A", "B", "C", "D", "E", "F", "G","H","I","J","K"]:
        _key = os.getenv(f"GROQ_API_KEY_{_i}_{_ac}")
        if _key:
            GROQ_API_KEYS.append(_key)

if not GROQ_API_KEYS:
    raise ValueError(
        "No GROQ API keys found.\n"
        "Expected .env entries like: GROQ_API_KEY_1_A, GROQ_API_KEY_2_B ...\n"
        "Get free keys at: https://console.groq.com/keys"
    )

print(f"[llm_client] Loaded {len(GROQ_API_KEYS)} API keys "
      f"({len(GROQ_API_KEYS) * 6_000:,} effective TPM)")

# ─────────────────────────────────────────────────────────────
# MODEL ASSIGNMENTS
# Verified live from Groq Playground, April 26 2026.
# Check https://console.groq.com/docs/deprecations if you get a 400.
# ─────────────────────────────────────────────────────────────
MODEL_CRITIC    = "llama-3.3-70b-versatile"   # critics  — structured JSON output
MODEL_OPTIMIZER = "openai/gpt-oss-120b"       # optimizer — strongest code generation
MODEL_REPAIR    = "llama-3.1-8b-instant"      # dedup/repair — binary calls only

GROQ_API_BASE = "https://api.groq.com/openai/v1/chat/completions"


class QuotaExhaustedException(Exception):
    """
    Raised when ALL API keys have exhausted their daily token quota.
    Caught by run_experiment() to stop gracefully and save progress.
    Distinct from a regular RuntimeError so callers can handle it specifically.
    """
    pass

# ─────────────────────────────────────────────────────────────
# Key rotation state
# ─────────────────────────────────────────────────────────────
_key_cycle              = cycle(GROQ_API_KEYS)
_key_lock               = asyncio.Lock()
_call_count:      int   = 0
_tokens_prompt:   int   = 0   # cumulative prompt tokens this session
_tokens_output:   int   = 0   # cumulative completion tokens this session
_tokens_total:    int   = 0   # cumulative total tokens this session
_consecutive_429: int   = 0   # tracks back-to-back 429s across all keys
_bad_keys:        set   = set() # keys that returned 401 — skipped automatically


async def _next_key() -> str:
    """Return the next API key in round-robin order, skipping known bad keys."""
    async with _key_lock:
        for _ in range(len(GROQ_API_KEYS) + 1):
            key = next(_key_cycle)
            if key not in _bad_keys:
                return key
        raise RuntimeError(
            "All API keys are invalid (401). "
            "Run python validate_keys.py to identify bad keys."
        )


# ─────────────────────────────────────────────────────────────
# Core async call
# ─────────────────────────────────────────────────────────────
async def _raw_call(
    system_prompt: str,
    user_content:  str,
    model:         str,
    temperature:   float = 0.0,
    max_tokens:    int   = 2048,
    retries:       int   = 5,
) -> str:
    """
    Single async POST to Groq with key rotation and resilient retry logic.

    Retry behaviour:
      429 (rate limit)    → exponential backoff with jitter, rotate key
      413 (too large)     → truncate user_content to 1500 chars, retry
      400 (decommissioned)→ fail immediately with clear error message
      timeout             → retry with backoff
      other 4xx/5xx       → retry up to retries times
    """
    global _call_count

    for attempt in range(retries):
        api_key = await _next_key()

        # Stagger delay — prevents concurrent calls from all firing at the
        # same millisecond and colliding on the same rate-limit window.
        await asyncio.sleep(random.uniform(0.1, 0.6))

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       model,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(
                    GROQ_API_BASE, headers=headers, json=payload
                )

                if resp.status_code == 200:
                    global _tokens_prompt, _tokens_output, _tokens_total, _consecutive_429
                    _consecutive_429 = 0   # reset on success
                    _call_count += 1
                    data = resp.json()
                    # accumulate token usage from Groq response
                    usage = data.get("usage", {})
                    _tokens_prompt += usage.get("prompt_tokens", 0)
                    _tokens_output += usage.get("completion_tokens", 0)
                    _tokens_total  += usage.get("total_tokens", 0)
                    return data["choices"][0]["message"]["content"]

                elif resp.status_code == 429:
                    _consecutive_429 += 1

                    # if every key has failed repeatedly, daily quota is exhausted
                    # threshold = keys × retries (e.g. 21 keys × 5 = 105 consecutive 429s)
                    quota_threshold = len(GROQ_API_KEYS) * retries
                    if _consecutive_429 >= quota_threshold:
                        print(f"\n[llm_client] ⚠️  DAILY QUOTA EXHAUSTED")
                        print(f"  All {len(GROQ_API_KEYS)} keys have failed "
                              f"{_consecutive_429} times consecutively.")
                        print(f"  Stopping gracefully. Resume tomorrow with --resume")
                        raise QuotaExhaustedException(
                            f"Daily token quota exhausted across all "
                            f"{len(GROQ_API_KEYS)} API keys."
                        )

                    # Exponential backoff with jitter — spreads retries in time
                    wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                    print(f"[llm_client] Rate limited. Waiting {wait:.1f}s "
                          f"(attempt {attempt+1}/{retries})")
                    await asyncio.sleep(wait)
                    continue

                elif resp.status_code == 413:
                    # Request too large — truncate and retry
                    print(f"[llm_client] 413 too large ({len(user_content)} chars). "
                          f"Truncating to 1500 chars and retrying...")
                    user_content = user_content[:1500] + "\n[...truncated...]"
                    await asyncio.sleep(0.5)
                    continue

                elif resp.status_code == 401:
                    # Invalid key — mark it and skip to next key on retry
                    print(f"[llm_client] ✗ Invalid key detected — skipping it. "
                          f"Run python validate_keys.py to find which .env entry is bad.")
                    _bad_keys.add(api_key)
                    break  # break inner try, continue outer for-loop for next attempt

                elif resp.status_code == 400 and "decommissioned" in resp.text:
                    # Model no longer exists — fail immediately, no point retrying
                    raise RuntimeError(
                        f"\n[llm_client] Model '{model}' has been decommissioned.\n"
                        f"Update MODEL_CRITIC / MODEL_OPTIMIZER in llm_client.py.\n"
                        f"Check https://console.groq.com/docs/models for current list.\n"
                    )

                else:
                    print(f"[llm_client] HTTP {resp.status_code}: {resp.text[:200]}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1 + attempt)
                        continue
                    raise RuntimeError(
                        f"Groq API error {resp.status_code}: {resp.text[:200]}"
                    )

            except httpx.TimeoutException:
                if attempt < retries - 1:
                    wait = 2 + attempt
                    print(f"[llm_client] Timeout. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                raise

    raise RuntimeError(f"All {retries} attempts failed for Groq API call.")


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

async def call_llm(
    system_prompt: str,
    inputs:        dict,
    model:         str   = MODEL_CRITIC,
    temperature:   float = 0.0,
) -> str:
    """
    General-purpose async LLM call.

    Args:
        system_prompt: System/role instruction for the model.
        inputs:        Dict of values injected into the user message.
                       Each value is serialised and concatenated as
                       ## Key\\nvalue sections.
        model:         Groq model string. Defaults to MODEL_CRITIC.
        temperature:   0.0 for deterministic output (recommended for eval).

    Returns:
        Raw string response from the model.

    Example:
        raw = await call_llm(
            CRITIC_PROMPTS["correctness"],
            {"problem": prob, "code": code, "test_results": failures}
        )
    """
    user_parts = []
    for key, val in inputs.items():
        if isinstance(val, (dict, list)):
            val = json.dumps(val, indent=2)
        user_parts.append(f"## {key.replace('_', ' ').title()}\n{val}")
    user_content = "\n\n".join(user_parts)

    return await _raw_call(
        system_prompt = system_prompt,
        user_content  = user_content,
        model         = model,
        temperature   = temperature,
    )


async def call_llm_optimizer(
    prompt_text: str,
    temperature: float = 0.0,
) -> str:
    """
    Dedicated call for the TGD optimizer step.
    Uses MODEL_OPTIMIZER and accepts a pre-formatted prompt string
    (built by build_optimizer_prompt) rather than a dict.

    Args:
        prompt_text: Fully formatted optimizer prompt string.
        temperature: 0.0 recommended for reproducibility.

    Returns:
        Raw code string (LLM returns only code per prompt instructions).
    """
    return await _raw_call(
        system_prompt = "You are an expert Python programmer.",
        user_content  = prompt_text,
        model         = MODEL_OPTIMIZER,
        temperature   = temperature,
        max_tokens    = 3000,
    )


async def call_llm_binary(
    question: str,
    context:  str,
    model:    str = MODEL_REPAIR,
) -> str:
    """
    Cheap binary yes/no/uncertain call — used for issue deduplication.
    Uses MODEL_REPAIR (smallest/fastest) since the call is trivial.

    Args:
        question: The yes/no question to ask.
        context:  Supporting context for the decision.

    Returns:
        One of: "yes", "no", "uncertain"
    """
    system = (
        "You are a precise classifier. "
        "Answer ONLY with one word: yes, no, or uncertain. "
        "No explanation, no punctuation, nothing else."
    )
    user = f"{context}\n\nQuestion: {question}"

    raw = await _raw_call(
        system_prompt = system,
        user_content  = user,
        model         = model,
        temperature   = 0.0,
        max_tokens    = 10,
    )
    answer = raw.strip().lower().split()[0] if raw.strip() else "uncertain"
    if answer not in ("yes", "no", "uncertain"):
        answer = "uncertain"
    return answer


async def repair_json(raw: str) -> list[dict]:
    """
    Fallback: ask MODEL_REPAIR to fix malformed JSON from a critic.
    Called only when parse_critic_output's json.loads fails.

    Args:
        raw: The malformed string returned by a critic.

    Returns:
        Parsed list of issue dicts, or empty list on total failure.
    """
    system = (
        "You fix malformed JSON. "
        "Return ONLY valid JSON matching this schema, nothing else:\n"
        '{"issues": [{"description": str, "region": str, '
        '"severity": "high"|"medium"|"low", "fix_hint": str}]}'
    )
    user = f"Fix this malformed JSON:\n{raw}"

    try:
        fixed = await _raw_call(
            system_prompt = system,
            user_content  = user,
            model         = MODEL_REPAIR,
            temperature   = 0.0,
            max_tokens    = 1000,
        )
        clean = fixed.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(clean)["issues"]
    except Exception as e:
        print(f"[llm_client] repair_json failed: {e}")
        return []


def get_call_count() -> int:
    """Returns total number of successful LLM API calls made this session."""
    return _call_count


def get_token_usage() -> dict:
    """Returns cumulative token usage since last reset."""
    return {
        "prompt_tokens":     _tokens_prompt,
        "completion_tokens": _tokens_output,
        "total_tokens":      _tokens_total,
    }


def snapshot_and_reset_tokens() -> dict:
    """
    Returns current token usage snapshot then resets counters to zero.
    Call this after each problem to get per-problem token counts.

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens for the problem.
    """
    global _call_count, _tokens_prompt, _tokens_output, _tokens_total
    snapshot = {
        "llm_calls":          _call_count,
        "prompt_tokens":      _tokens_prompt,
        "completion_tokens":  _tokens_output,
        "total_tokens":       _tokens_total,
    }
    _call_count    = 0
    _tokens_prompt = 0
    _tokens_output = 0
    _tokens_total  = 0
    return snapshot


def reset_call_count() -> None:
    """Reset all counters (call at the start of each experiment run)."""
    global _call_count, _tokens_prompt, _tokens_output, _tokens_total, _consecutive_429
    _call_count      = 0
    _tokens_prompt   = 0
    _tokens_output   = 0
    _tokens_total    = 0
    _consecutive_429 = 0