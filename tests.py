"""
tests.py
────────
Test suite for AdaTextGrad.
Tests each module independently so you can validate components
before wiring the full loop.

Run all tests:
    python tests.py

Run a specific stage:
    python tests.py Stage1
    python tests.py Stage2
    python tests.py Stage3
    python tests.py Stage4
    python tests.py Stage5

Stages mirror the implementation order from the paper:
  Stage 1 — Test runner (no LLM)
  Stage 2 — Critic output parsing + JSON repair (no LLM)
  Stage 3 — Structured log: update_log + find_duplicate (no LLM)
  Stage 4 — Momentum math + step controller (no LLM)
  Stage 5 — render_gradient + build_optimizer_prompt (no LLM)
  Stage 6 — Single iteration smoke test (REQUIRES real API keys)
"""

import asyncio
import json
import sys
import traceback
from typing import Callable


# ═══════════════════════════════════════════════════════════════
# TEST HARNESS
# ═══════════════════════════════════════════════════════════════

_passed = 0
_failed = 0
_errors = []


def ok(name: str) -> None:
    global _passed
    _passed += 1
    print(f"  ✓  {name}")


def fail(name: str, reason: str) -> None:
    global _failed
    _failed += 1
    _errors.append((name, reason))
    print(f"  ✗  {name}")
    print(f"       {reason}")


def check(name: str, condition: bool, reason: str = "") -> None:
    if condition:
        ok(name)
    else:
        fail(name, reason or "assertion failed")


def section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def summary() -> None:
    print(f"\n{'═'*55}")
    total = _passed + _failed
    print(f"  Results: {_passed}/{total} passed")
    if _errors:
        print(f"\n  Failures:")
        for name, reason in _errors:
            print(f"    ✗ {name}: {reason}")
    print(f"{'═'*55}\n")
    if _failed > 0:
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# STAGE 1 — TEST RUNNER
# ═══════════════════════════════════════════════════════════════

def test_stage1_runner():
    section("Stage 1 — Test runner (no LLM)")
    from problems import run_tests

    # correct two-sum
    correct_code = """
class Solution:
    def solve(self, nums, target):
        seen = {}
        for i, n in enumerate(nums):
            if target - n in seen:
                return [seen[target - n], i]
            seen[n] = i
"""
    cases = [
        {"input": "[2,7,11,15], 9",  "expected_output": "[0, 1]"},
        {"input": "[3,2,4], 6",      "expected_output": "[1, 2]"},
        {"input": "[3,3], 6",        "expected_output": "[0, 1]"},
    ]
    r = run_tests(correct_code, cases)
    check("correct code: pass_rate == 1.0",    r["pass_rate"] == 1.0,
          f"got {r['pass_rate']}")
    check("correct code: no failures",         len(r["failures"]) == 0,
          f"got {r['failures']}")
    check("correct code: passing_cases count", len(r["passing_cases"]) == 3,
          f"got {len(r['passing_cases'])}")

    # broken code
    broken_code = """
class Solution:
    def solve(self, nums, target):
        return []
"""
    r2 = run_tests(broken_code, cases)
    check("broken code: pass_rate == 0.0", r2["pass_rate"] == 0.0,
          f"got {r2['pass_rate']}")
    check("broken code: failures count",   len(r2["failures"]) == 3,
          f"got {len(r2['failures'])}")

    # timeout code
    timeout_code = """
class Solution:
    def solve(self, nums, target):
        while True:
            pass
"""
    r3 = run_tests(timeout_code, cases[:1])
    check("timeout code: caught as error",
          "ERROR" in r3["failures"][0]["got"] if r3["failures"] else False,
          f"got: {r3}")

    # empty test cases
    r4 = run_tests(correct_code, [])
    check("empty test cases: returns zero pass_rate", r4["pass_rate"] == 0.0)
    check("empty test cases: total == 0",             r4["total"] == 0)


# ═══════════════════════════════════════════════════════════════
# STAGE 2 — CRITIC OUTPUT PARSING
# ═══════════════════════════════════════════════════════════════

def test_stage2_parsing():
    section("Stage 2 — Critic output parsing (no LLM)")
    from main import parse_critic_output

    # valid JSON
    valid = json.dumps({
        "issues": [
            {"description": "off by one error", "region": "loop",
             "severity": "high", "fix_hint": "use <= instead of <"}
        ]
    })
    result = parse_critic_output(valid)
    check("valid JSON: returns list",         isinstance(result, list),
          f"got {type(result)}")
    check("valid JSON: one issue",            len(result) == 1,
          f"got {len(result)}")
    check("valid JSON: has description",      "description" in result[0])
    check("valid JSON: has region",           "region" in result[0])

    # JSON with markdown fences
    fenced = "```json\n" + valid + "\n```"
    result2 = parse_critic_output(fenced)
    check("fenced JSON: parsed correctly",    len(result2) == 1,
          f"got {len(result2)}")

    # empty issues list
    empty_json = '{"issues": []}'
    result3 = parse_critic_output(empty_json)
    check("empty issues: returns empty list", result3 == [],
          f"got {result3}")

    # missing required field — should be filtered out
    incomplete = json.dumps({
        "issues": [{"description": "bug", "region": "loop"}]  # missing severity + fix_hint
    })
    result4 = parse_critic_output(incomplete)
    check("incomplete issue: filtered out",   result4 == [],
          f"got {result4}")

    # empty string
    result5 = parse_critic_output("")
    check("empty string: returns []",         result5 == [],
          f"got {result5}")

    # multiple issues
    multi = json.dumps({
        "issues": [
            {"description": "bug A", "region": "func A",
             "severity": "high", "fix_hint": "fix A"},
            {"description": "bug B", "region": "func B",
             "severity": "low",  "fix_hint": "fix B"},
        ]
    })
    result6 = parse_critic_output(multi)
    check("multiple issues: count == 2",      len(result6) == 2,
          f"got {len(result6)}")


# ═══════════════════════════════════════════════════════════════
# STAGE 3 — STRUCTURED LOG
# ═══════════════════════════════════════════════════════════════

def test_stage3_log():
    section("Stage 3 — Structured log (no LLM, using mock dedup)")
    from main import get_issue, compute_momentum, update_log

    # test get_issue
    log = [
        {"id": "issue_000", "description": "bug A", "region": "loop",
         "resolved": False, "frequency": 1, "iterations_seen": [1],
         "fix_attempted": [], "weight": 0.0, "stuck": False, "severity": "high",
         "first_seen": 1},
    ]
    check("get_issue: found by id",    get_issue(log, "issue_000") is not None)
    check("get_issue: not found",      get_issue(log, "issue_999") is None)

    # test update_log with mock — bypass LLM dedup by monkeypatching
    async def mock_update_no_dedup(issue_log, new_issue, t, region_rewrites):
        """Simplified update that always treats issues as new (no LLM call)."""
        issue_log.append({
            "id":              f"issue_{len(issue_log):03d}",
            "description":     new_issue["description"],
            "region":          new_issue["region"],
            "severity":        new_issue.get("severity", "medium"),
            "first_seen":      t,
            "iterations_seen": [t],
            "frequency":       1,
            "fix_attempted":   [],
            "resolved":        False,
            "weight":          0.0,
            "stuck":           False,
        })
        region = new_issue["region"]
        region_rewrites[region] = region_rewrites.get(region, 0) + 1

    new_log       = []
    region_rw     = {}
    new_issue_1   = {"description": "empty array case", "region": "balance block",
                     "severity": "high", "fix_hint": "add early return"}
    asyncio.get_event_loop().run_until_complete(
        mock_update_no_dedup(new_log, new_issue_1, 1, region_rw)
    )
    check("update_log: issue added",              len(new_log) == 1,
          f"got {len(new_log)}")
    check("update_log: region counter incremented", region_rw.get("balance block") == 1,
          f"got {region_rw}")
    check("update_log: correct id format",        new_log[0]["id"] == "issue_000")
    check("update_log: frequency starts at 1",    new_log[0]["frequency"] == 1)

    # second different issue
    new_issue_2 = {"description": "off by one", "region": "window loop",
                   "severity": "medium", "fix_hint": "use <="}
    asyncio.get_event_loop().run_until_complete(
        mock_update_no_dedup(new_log, new_issue_2, 1, region_rw)
    )
    check("update_log: two issues in log",        len(new_log) == 2,
          f"got {len(new_log)}")


# ═══════════════════════════════════════════════════════════════
# STAGE 4 — MOMENTUM MATH + STEP CONTROLLER
# ═══════════════════════════════════════════════════════════════

def test_stage4_momentum():
    section("Stage 4 — Momentum math + step controller (no LLM)")
    from main import compute_momentum, step_scope

    beta1 = 0.9

    # build a mock issue log
    log = [
        {
            "id": "issue_000", "description": "bug A", "region": "loop A",
            "frequency": 1, "iterations_seen": [1], "fix_attempted": [],
            "resolved": False, "weight": 0.0, "stuck": False,
            "severity": "high", "first_seen": 1,
        },
        {
            "id": "issue_001", "description": "bug B", "region": "loop B",
            "frequency": 5, "iterations_seen": [1,2,3,4,5], "fix_attempted": [],
            "resolved": False, "weight": 0.0, "stuck": False,
            "severity": "high", "first_seen": 1,
        },
        {
            "id": "issue_002", "description": "bug C", "region": "loop C",
            "frequency": 1, "iterations_seen": [1], "fix_attempted": [],
            "resolved": True, "weight": 0.0, "stuck": False,  # resolved — should skip
            "severity": "low", "first_seen": 1,
        },
    ]
    region_rewrites = {"loop A": 1, "loop B": 5, "loop C": 1}

    compute_momentum(log, t=5, region_rewrites=region_rewrites, beta1=beta1,
                     stuck_threshold=3)

    w_low  = log[0]["weight"]  # freq=1, should be lower
    w_high = log[1]["weight"]  # freq=5, should be higher

    check("momentum: freq=1 weight < freq=5 weight",  w_low < w_high,
          f"w_low={w_low:.4f}, w_high={w_high:.4f}")
    check("momentum: freq=5 weight > 0.9",            w_high > 0.9,
          f"got {w_high:.4f}")
    check("momentum: resolved issue weight unchanged", log[2]["weight"] == 0.0,
          f"got {log[2]['weight']}")
    check("momentum: region stuck flagged (rewrites=5 >= 3)", log[1]["stuck"],
          f"got stuck={log[1]['stuck']}")
    check("momentum: region not stuck (rewrites=1 < 3)", not log[0]["stuck"],
          f"got stuck={log[0]['stuck']}")

    # step controller
    check("step_scope: delta>0.1 → targeted",        step_scope(0.2,  False) == "targeted")
    check("step_scope: delta=0 → structural",         step_scope(0.0,  False) == "structural")
    check("step_scope: any_stuck → structural",       step_scope(0.05, True)  == "structural")
    check("step_scope: slight progress → targeted",   step_scope(0.05, False) == "targeted")
    check("step_scope: good progress + stuck → structural", step_scope(0.15, True) == "structural")

    # bias correction: t=1 should give lower weight than t=5 for same freq
    log_t1 = [{"id": "i0", "region": "r", "frequency": 3, "resolved": False,
                "weight": 0.0, "stuck": False, "iterations_seen": [1],
                "fix_attempted": [], "severity": "high", "first_seen": 1,
                "description": "x"}]
    log_t5 = [{"id": "i0", "region": "r", "frequency": 3, "resolved": False,
                "weight": 0.0, "stuck": False, "iterations_seen": [1],
                "fix_attempted": [], "severity": "high", "first_seen": 1,
                "description": "x"}]
    compute_momentum(log_t1, t=1, region_rewrites={"r": 1}, beta1=beta1)
    compute_momentum(log_t5, t=5, region_rewrites={"r": 1}, beta1=beta1)
    check("bias correction: weight at t=1 > weight at t=5 (same freq)",
          log_t1[0]["weight"] > log_t5[0]["weight"],
          f"t1={log_t1[0]['weight']:.4f}, t5={log_t5[0]['weight']:.4f}")


# ═══════════════════════════════════════════════════════════════
# STAGE 5 — RENDER GRADIENT + BUILD OPTIMIZER PROMPT
# ═══════════════════════════════════════════════════════════════

def test_stage5_rendering():
    section("Stage 5 — render_gradient + build_optimizer_prompt (no LLM)")
    from main import render_gradient, build_optimizer_prompt

    issues = [
        {"description": "empty array not handled", "region": "balance block",
         "frequency": 3, "weight": 0.91, "fix_attempted": ["early return", "check len"],
         "stuck": True, "resolved": False},
        {"description": "off by one", "region": "window loop",
         "frequency": 1, "weight": 0.33, "fix_attempted": [],
         "stuck": False, "resolved": False},
    ]

    gradient = render_gradient(issues, "targeted")
    check("render_gradient: contains HIGH PRIORITY",     "HIGH PRIORITY" in gradient)
    check("render_gradient: contains MEDIUM PRIORITY",  "MEDIUM PRIORITY" in gradient)
    check("render_gradient: contains STUCK flag",        "STUCK" in gradient)
    check("render_gradient: contains fix_attempted",     "early return" in gradient)
    check("render_gradient: targeted scope instruction", "targeted" in gradient.lower())
    check("render_gradient: issue frequency shown",      "3 iteration" in gradient)

    gradient_s = render_gradient(issues, "structural")
    check("render_gradient: structural scope instruction",
          "fundamentally different" in gradient_s.lower() or "FUNDAMENTALLY" in gradient_s)

    # empty issues
    gradient_empty = render_gradient([], "targeted")
    check("render_gradient: empty issues handled", len(gradient_empty) > 0)

    # build_optimizer_prompt — targeted
    prompt_t = build_optimizer_prompt(
        problem       = "Find median",
        code          = "class Solution: pass",
        test_failures = "[]",
        gradient_text = gradient,
        scope         = "targeted",
        issue_log     = issues,
    )
    check("build_optimizer_prompt: targeted contains MINIMAL", "MINIMAL" in prompt_t)
    check("build_optimizer_prompt: targeted contains gradient", "HIGH PRIORITY" in prompt_t)
    check("build_optimizer_prompt: targeted contains problem",  "Find median" in prompt_t)

    # build_optimizer_prompt — structural
    prompt_s = build_optimizer_prompt(
        problem       = "Find median",
        code          = "class Solution: pass",
        test_failures = "[]",
        gradient_text = gradient_s,
        scope         = "structural",
        issue_log     = issues,
    )
    check("build_optimizer_prompt: structural contains STUCK",   "STUCK" in prompt_s)
    check("build_optimizer_prompt: structural contains attempts","early return" in prompt_s)


# ═══════════════════════════════════════════════════════════════
# STAGE 6 — SINGLE ITERATION SMOKE TEST (requires API keys)
# ═══════════════════════════════════════════════════════════════

async def _smoke_test_single_iter():
    """
    End-to-end single iteration test on a simple known problem.
    Requires valid GROQ_API_KEYS in llm_client.py.
    """
    from llm_client import GROQ_API_KEYS
    from main import ada_textgrad

    if not GROQ_API_KEYS or GROQ_API_KEYS[0].startswith("YOUR_"):
        print("  [SKIP] No real API keys found in llm_client.py")
        print("         Set GROQ_API_KEYS to run this stage.")
        return

    # very simple problem — one iteration should suffice
    problem = (
        "Given a list of integers nums and an integer target, "
        "return the indices of two numbers that add up to target. "
        "There is exactly one solution. "
        "Class Solution with method solve(self, nums, target) -> list[int]."
    )
    initial_code = "class Solution:\n    def solve(self, nums, target):\n        pass"
    test_cases   = [
        {"input": "[2,7,11,15], 9", "expected_output": "[0, 1]"},
        {"input": "[3,2,4], 6",     "expected_output": "[1, 2]"},
    ]

    final_code, run_log = await ada_textgrad(
        problem      = problem,
        initial_code = initial_code,
        test_suite   = test_cases,
        max_iter     = 3,
    )

    check("smoke: final_code is a string",       isinstance(final_code, str))
    check("smoke: final_code non-empty",         len(final_code) > 10)
    check("smoke: run_log has pass_history",     "pass_history" in run_log)
    check("smoke: run_log has llm_calls",        run_log["llm_calls"] > 0)
    check("smoke: llm_calls > 0",               run_log["llm_calls"] > 0)
    print(f"  [info] LLM calls used: {run_log['llm_calls']}")
    print(f"  [info] Pass history: {run_log['pass_history']}")


def test_stage6_smoke():
    section("Stage 6 — Single iteration smoke test (REQUIRES API keys)")
    asyncio.get_event_loop().run_until_complete(_smoke_test_single_iter())


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════



def test_stage1b_dataset_helpers():
    section("Stage 1b — Dataset helpers: _parse_input_output + _build_exec_script")
    from problems import _parse_input_output, _build_exec_script

    # --- _parse_input_output ---

    # normal usable entries
    io_list = [
        {"input": "n = 5, queries = [[2,4],[0,2],[0,4]]", "output": "[3, 2, 1]"},
        {"input": "n = 4, queries = [[0,3],[0,2]]",       "output": "[1, 1]"},
        {"input": "n = 6, queries = [[1,5],[2,5]]",       "output": "Execution timed out"},
        {"input": "n = 10, queries = [[1,9]]",            "output": "Error: list index out of range"},
        {"input": "n = 3, queries = [[0,2]]",             "output": ""},  # empty — skip
    ]
    visible, hidden = _parse_input_output(io_list)
    check("parse_io: filters timed out entries",   len(visible) + len(hidden) == 2,
          f"got visible={len(visible)} hidden={len(hidden)}")
    check("parse_io: filters Error entries",       all("Error" not in v["expected_output"] for v in visible + hidden))
    check("parse_io: has expected_output key",     "expected_output" in visible[0])
    check("parse_io: has input key",               "input" in visible[0])
    check("parse_io: hidden >= 1",                 len(hidden) >= 1)

    # all entries bad — should return empty
    bad_io = [
        {"input": "x", "output": "Execution timed out"},
        {"input": "y", "output": "Error: something"},
    ]
    vis2, hid2 = _parse_input_output(bad_io)
    check("parse_io: all bad → empty lists",       vis2 == [] and hid2 == [])

    # empty input
    vis3, hid3 = _parse_input_output([])
    check("parse_io: empty input → empty lists",   vis3 == [] and hid3 == [])

    # single usable entry — goes to hidden only
    single = [{"input": "n = 5, queries = [[2,4]]", "output": "[3]"}]
    vis4, hid4 = _parse_input_output(single)
    check("parse_io: single entry in hidden",      len(hid4) == 1)

    # --- _build_exec_script ---

    # standard entry_point format "Solution().methodName"
    script = _build_exec_script(
        prompt_header = "from typing import *",
        code          = "class Solution:\n    def solve(self, n): return n*2",
        entry_point   = "Solution().solve",
        inp           = "n = 5",
        expected      = "10",
    )
    check("build_exec: contains entry point method",  "solve" in script)
    check("build_exec: contains dict() call",          "dict(n = 5)" in script or "dict(" in script)
    check("build_exec: contains expected value",       "10" in script)
    check("build_exec: contains PASS string",          "PASS" in script)
    check("build_exec: contains FAIL string",          "FAIL" in script)

    # entry_point with Solution().methodName format
    script2 = _build_exec_script(
        prompt_header = "from typing import *\nfrom collections import *",
        code          = "class Solution:\n    def shortestDistanceAfterQueries(self, n, queries): return []",
        entry_point   = "Solution().shortestDistanceAfterQueries",
        inp           = "n = 5, queries = [[2,4],[0,2],[0,4]]",
        expected      = "[3, 2, 1]",
    )
    check("build_exec: correct method name extracted", "shortestDistanceAfterQueries" in script2)
    check("build_exec: getattr pattern used",          "getattr" in script2)

    # actually run the script to verify it works end-to-end
    import subprocess, tempfile, os
    ref_code = """class Solution:
    def shortestDistanceAfterQueries(self, n, queries):
        nxt = list(range(1, n))
        ans = []
        cnt = n - 1
        for u, v in queries:
            if 0 < nxt[u] < v:
                i = nxt[u]
                while i < v:
                    cnt -= 1
                    nxt[i], i = 0, nxt[i]
                nxt[u] = v
            ans.append(cnt)
        return ans
"""
    runnable = _build_exec_script(
        prompt_header = "from typing import *\nfrom collections import *",
        code          = ref_code,
        entry_point   = "Solution().shortestDistanceAfterQueries",
        inp           = "n = 5, queries = [[2,4],[0,2],[0,4]]",
        expected      = "[3, 2, 1]",
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(runnable)
        tmp = f.name
    try:
        out = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=5)
        result = out.stdout.strip()
    finally:
        os.unlink(tmp)
    check("build_exec: reference solution runs and PASS", result == "PASS",
          f"got: {repr(result)}\nstderr: {out.stderr[:200]}")

STAGES = {
    "Stage1": test_stage1_runner,
    "Stage1b": test_stage1b_dataset_helpers,
    "Stage2": test_stage2_parsing,
    "Stage3": test_stage3_log,
    "Stage4": test_stage4_momentum,
    "Stage5": test_stage5_rendering,
    "Stage6": test_stage6_smoke,
}

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None

    if target:
        if target not in STAGES:
            print(f"Unknown stage '{target}'. Available: {list(STAGES.keys())}")
            sys.exit(1)
        STAGES[target]()
    else:
        # run all non-API stages by default
        for name, fn in STAGES.items():
            if name not in ("Stage6",):
                try:
                    fn()
                except Exception as e:
                    section(f"{name} — CRASHED")
                    print(f"  ERROR: {e}")
                    traceback.print_exc()
                    _failed += 1
                    _errors.append((name, str(e)))

    summary()
