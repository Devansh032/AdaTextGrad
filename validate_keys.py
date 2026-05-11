"""
validate_keys.py
────────────────
Tests every Groq API key and prints which ones are valid/invalid.
Run this before starting an experiment to catch bad keys early.

Usage:
    python validate_keys.py
"""

import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_BASE = "https://api.groq.com/openai/v1/chat/completions"
TEST_MODEL    = "llama-3.1-8b-instant"   # cheapest model for validation

# Load all keys with their names so we can report which .env entry is bad
named_keys: list[tuple[str, str]] = []
for i in range(1, 4):
    for ac in ["A", "B", "C", "D", "E", "F", "G","H","I","J","K"]:
        env_name = f"GROQ_API_KEY_{i}_{ac}"
        key      = os.getenv(env_name)
        if key:
            named_keys.append((env_name, key))

print(f"Found {len(named_keys)} keys in .env. Testing each one...\n")


async def test_key(env_name: str, key: str) -> tuple[str, bool, str]:
    """
    Send a minimal test request with this key.
    Returns (env_name, is_valid, reason).
    """
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":      TEST_MODEL,
        "max_tokens": 5,
        "messages":   [{"role": "user", "content": "Hi"}],
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(GROQ_API_BASE, headers=headers, json=payload)

            if resp.status_code == 200:
                return (env_name, True, "OK")
            elif resp.status_code == 401:
                return (env_name, False, "INVALID KEY — 401 Unauthorized")
            elif resp.status_code == 429:
                return (env_name, True, "OK (rate limited but key is valid)")
            elif resp.status_code == 400 and "decommissioned" in resp.text:
                return (env_name, True, "OK (key valid, model decommissioned)")
            else:
                return (env_name, False, f"HTTP {resp.status_code}: {resp.text[:100]}")

        except httpx.TimeoutException:
            return (env_name, False, "TIMEOUT — could not reach Groq API")
        except Exception as e:
            return (env_name, False, f"ERROR: {e}")


async def main():
    # test all keys concurrently
    tasks   = [test_key(name, key) for name, key in named_keys]
    results = await asyncio.gather(*tasks)

    valid   = [(n, r) for n, ok, r in results if ok]
    invalid = [(n, r) for n, ok, r in results if not ok]

    print("─" * 50)
    print(f"  VALID keys:   {len(valid)}/{len(named_keys)}")
    print(f"  INVALID keys: {len(invalid)}/{len(named_keys)}")
    print("─" * 50)

    if valid:
        print("\n  ✓ Valid:")
        for name, reason in valid:
            print(f"    {name:<25} {reason}")

    if invalid:
        print("\n  ✗ Invalid (fix these in your .env):")
        for name, reason in invalid:
            print(f"    {name:<25} {reason}")
        print(f"\n  Remove or replace the invalid keys in your .env file.")
        print(f"  Get new keys at: https://console.groq.com/keys")
    else:
        print("\n  All keys are valid. Ready to run experiments.")

    # show final usable key count
    usable = len(valid)
    print(f"\n  Effective TPM with {usable} keys: ~{usable * 6_000:,} tokens/min")


asyncio.run(main())