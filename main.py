"""
main.py
───────
AdaTextGrad core system.

Contains:
  - CRITIC_PROMPTS                  critic system prompts
  - OPTIMIZER_PROMPT_TARGETED       targeted edit prompt
  - OPTIMIZER_PROMPT_STRUCTURAL     structural rewrite prompt
  - parse_critic_output()           parse + validate critic JSON
  - render_gradient()               build gradient text from issue log
  - build_optimizer_prompt()        select + format optimizer prompt
  - find_duplicate()                LLM-assisted issue deduplication
  - get_issue()                     helper to fetch issue from log by id
  - update_log()                    add/merge new issue into structured log
  - compute_momentum()              adam-style weight computation (pure math)
  - step_scope()                    adaptive step controller (pure logic)
  - concurrent_critics()            fire 3 critics in parallel
  - ada_textgrad()                  full optimization loop
  - vanilla_textgrad()              baseline for comparison
"""

import asyncio
import json
from typing import Optional

from llm_client import (
    call_llm,
    call_llm_optimizer,
    call_llm_binary,
    repair_json,
    MODEL_OPTIMIZER,
)

# ═══════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════

CRITIC_PROMPTS: dict[str, str] = {
    # ── FIX A: Added LeetCode constraint guardrails to correctness critic ──
    "correctness": """\
You are a code correctness critic. Analyze the code and test failures.
Your ONLY job is to identify logical bugs that cause wrong answers.

IMPORTANT — Do NOT report issues about:
- Empty input handling (LeetCode constraints guarantee non-empty inputs)
- Negative numbers (constraints specify valid ranges)
- Overflow (Python handles arbitrary precision)
- Duplicate values (unless duplicates are the actual bug)
Focus ONLY on algorithmic correctness bugs visible in the test failures.

Respond ONLY in this exact JSON format, no preamble, no explanation:
{
  "issues": [
    {
      "description": "one sentence describing the exact bug",
      "region": "short name of the code region (e.g. 'sliding window loop', 'base case check')",
      "severity": "high",
      "fix_hint": "one concrete sentence on how to fix it"
    }
  ]
}
If no correctness issues found, return: {"issues": []}""",

    # ── FIX A: Added "visible in test failures" constraint to edge case critic ──
    "edge_cases": """\
You are an edge case critic. Your ONLY job is to identify missing boundary conditions
that are VISIBLE IN THE TEST FAILURES provided.

Do NOT report generic edge cases like empty inputs, negative numbers, or overflow
unless the test failures explicitly show those failing.
Focus ONLY on boundary conditions that the failing tests actually expose.

Respond ONLY in this exact JSON format, no preamble, no explanation:
{
  "issues": [
    {
      "description": "one sentence describing the missing edge case",
      "region": "short name of the code region",
      "severity": "high" | "medium" | "low",
      "fix_hint": "one concrete sentence on how to handle this edge case"
    }
  ]
}
If no edge case issues found, return: {"issues": []}""",

    "complexity": """\
You are an algorithmic complexity critic. Your ONLY job is to identify approaches
that will cause TLE (time limit exceeded) on large inputs.
Focus on: O(n²) where O(n log n) exists, unnecessary nested loops, exponential recursion without memoization.

Respond ONLY in this exact JSON format, no preamble, no explanation:
{
  "issues": [
    {
      "description": "one sentence describing the complexity problem",
      "region": "short name of the code region",
      "severity": "high" | "medium" | "low",
      "fix_hint": "one concrete sentence suggesting a better algorithm"
    }
  ]
}
If no complexity issues found, return: {"issues": []}""",
}

VANILLA_EVALUATOR_PROMPT = """\
You are a code reviewer. Analyze the code and test failures.
Identify the most important bugs and suggest fixes.
Be specific and concrete."""

OPTIMIZER_PROMPT_TARGETED = """\
You are a precise code editor. Your job is to make MINIMAL fixes to the code.

## Problem
{problem}

## Current Code
{current_code}

## Test Failures
{test_failures}

## Prioritized Issues (fix in this order)
{gradient}

## Instructions
- Fix ONLY the flagged regions listed above
- Do NOT rewrite working parts of the code
- Do NOT change the overall algorithm structure
- Address the highest-weight issues first
- If a fix was already attempted (listed under "Tried:"), do not try the same approach again

Return ONLY the complete corrected Python code. No explanation. No markdown fences."""

OPTIMIZER_PROMPT_STRUCTURAL = """\
You are an algorithm designer. The current approach is stuck — patching it is not working.
You must design a completely different solution from scratch.

## Problem
{problem}

## Current Code (STUCK — do not patch this, replace it entirely)
{current_code}

## Test Failures
{test_failures}

## Issues That Resisted All Previous Fixes
{gradient}

## Fixes Already Attempted (do NOT repeat these)
{attempted_fixes_summary}

## Instructions
- Discard the current algorithm entirely
- Think about the problem from scratch before writing any code
- Design a new approach that structurally avoids the listed failure modes
- Do NOT reuse the same core logic

Return ONLY the complete new Python code. No explanation. No markdown fences."""

VANILLA_OPTIMIZER_PROMPT = """\
You are an expert Python programmer. Fix the code based on the feedback provided.

## Problem
{problem}

## Current Code
{current_code}

## Feedback
{feedback}

## Test Failures
{test_failures}

Return ONLY the complete corrected Python code. No explanation. No markdown fences."""


# ═══════════════════════════════════════════════════════════════
# PARSING + RENDERING
# ═══════════════════════════════════════════════════════════════

def parse_critic_output(raw: str) -> list[dict]:
    """
    Parse JSON output from a critic LLM call.
    Strips markdown fences if present. Falls back to LLM repair on failure.

    Args:
        raw: Raw string response from the critic.

    Returns:
        List of issue dicts with keys: description, region, severity, fix_hint.
    """
    if not raw or not raw.strip():
        return []

    # strip markdown fences if model added them despite instructions
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        # remove first and last fence lines
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    clean = clean.strip()

    try:
        parsed = json.loads(clean)
        issues = parsed.get("issues", [])
        # validate each issue has required fields
        validated = []
        for issue in issues:
            if all(k in issue for k in ("description", "region", "severity", "fix_hint")):
                validated.append(issue)
        return validated
    except json.JSONDecodeError:
        # async repair — run synchronously for simplicity in the calling context
        print(f"[parse_critic_output] JSON parse failed, attempting repair...")
        return asyncio.get_event_loop().run_until_complete(repair_json(raw))


def render_gradient(issues: list[dict], scope: str) -> str:
    """
    Convert the prioritized issue log into the textual gradient string
    that gets passed to the optimizer prompt.

    Args:
        issues: List of issue dicts sorted by weight descending.
        scope:  "targeted" or "structural" — appended as scope instruction.

    Returns:
        Formatted gradient string.
    """
    if not issues:
        return "No specific issues identified. Review the test failures carefully.\n"

    lines = []
    for issue in issues:
        persistence = f"(seen in {issue['frequency']} iteration(s))"
        attempts    = (f"Tried: {', '.join(issue['fix_attempted'])}."
                       if issue["fix_attempted"] else "")
        stuck_flag  = " [STUCK — change approach]" if issue.get("stuck") else ""

        if issue["weight"] > 0.7:
            priority = "HIGH PRIORITY"
        elif issue["weight"] > 0.2:
            priority = "MEDIUM PRIORITY"
        else:
            priority = "LOW PRIORITY"

        line = (
            f"[{priority}]{stuck_flag} {issue['description']} "
            f"{persistence}. {attempts} Region: {issue['region']}."
        )
        lines.append(line)

    scope_instruction = (
        "\nMake ONLY targeted fixes to the flagged regions above."
        if scope == "targeted"
        else "\nThe current approach is stuck. Propose a FUNDAMENTALLY DIFFERENT algorithm."
    )

    return "\n".join(lines) + scope_instruction


def build_optimizer_prompt(
    problem:       str,
    code:          str,
    test_failures: str,
    gradient_text: str,
    scope:         str,
    issue_log:     list[dict],
) -> str:
    """
    Build the final optimizer prompt based on the step scope.

    Args:
        problem:       Problem description string.
        code:          Current code string.
        test_failures: Stringified test failure details.
        gradient_text: Output of render_gradient().
        scope:         "targeted" or "structural".
        issue_log:     Full issue log (used for attempt history in structural mode).

    Returns:
        Fully formatted prompt string ready for call_llm_optimizer().
    """
    if scope == "structural":
        attempted = [
            f"- {i['description']}: tried {', '.join(i['fix_attempted'])}"
            for i in issue_log
            if i["fix_attempted"] and not i["resolved"]
        ]
        attempted_summary = "\n".join(attempted) if attempted else "none recorded"

        return OPTIMIZER_PROMPT_STRUCTURAL.format(
            problem                 = problem,
            current_code            = code,
            test_failures           = test_failures,
            gradient                = gradient_text,
            attempted_fixes_summary = attempted_summary,
        )
    else:
        return OPTIMIZER_PROMPT_TARGETED.format(
            problem       = problem,
            current_code  = code,
            test_failures = test_failures,
            gradient      = gradient_text,
        )


# ═══════════════════════════════════════════════════════════════
# STRUCTURED LOG HELPERS
# ═══════════════════════════════════════════════════════════════

def get_issue(issue_log: list[dict], issue_id: str) -> Optional[dict]:
    """
    Retrieve an issue from the log by its id.

    Args:
        issue_log: The structured issue log list.
        issue_id:  The id string (e.g. "issue_001").

    Returns:
        The issue dict if found, else None.
    """
    for issue in issue_log:
        if issue["id"] == issue_id:
            return issue
    return None


async def find_duplicate(
    new_issue: dict,
    issue_log: list[dict],
) -> Optional[str]:
    """
    Check whether new_issue describes the same bug as an existing log entry.
    Uses a cheap binary LLM call, filtered first to same-region candidates
    to reduce false positives and minimise API calls.

    Args:
        new_issue: Newly parsed issue dict from a critic.
        issue_log: Current structured log.

    Returns:
        The id of the matching existing issue, or None if new.
    """
    if not issue_log:
        return None

    # filter to same-region candidates first (no LLM needed)
    candidates = [
        i for i in issue_log
        if i["region"].lower() == new_issue["region"].lower()
        and not i["resolved"]
    ]

    if not candidates:
        return None

    for existing in candidates:
        answer = await call_llm_binary(
            question=(
                f'Is "{new_issue["description"]}" describing the same underlying bug as '
                f'"{existing["description"]}"?'
            ),
            context=(
                f"Both are in region: {existing['region']}. "
                f"Existing: {existing['description']}. "
                f"New: {new_issue['description']}."
            ),
        )
        if answer == "yes":
            return existing["id"]

    return None


async def update_log(
    issue_log:       list[dict],
    new_issue:       dict,
    t:               int,
    region_rewrites: dict,
) -> None:
    """
    Add a new issue to the log, or merge with existing if duplicate.
    Also increments the region rewrite counter.

    Args:
        issue_log:       Structured issue log (mutated in place).
        new_issue:       New issue dict from a critic.
        t:               Current iteration number.
        region_rewrites: Region rewrite counter dict (mutated in place).
    """
    matched_id = await find_duplicate(new_issue, issue_log)

    if matched_id:
        entry = get_issue(issue_log, matched_id)
        if t not in entry["iterations_seen"]:
            entry["iterations_seen"].append(t)
            entry["frequency"] += 1
        # append fix hint if it's a new suggestion
        hint = new_issue.get("fix_hint", "")
        if hint and hint not in entry["fix_attempted"]:
            entry["fix_attempted"].append(hint)
    else:
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

    # update variance tracker
    region = new_issue["region"]
    region_rewrites[region] = region_rewrites.get(region, 0) + 1


def compute_momentum(
    issue_log:       list[dict],
    t:               int,
    region_rewrites: dict,
    beta1:           float = 0.9,
    stuck_threshold: int   = 3,
) -> None:
    """
    Compute adam-style momentum weights for all active issues.
    Mutates issue_log in place — sets 'weight' and 'stuck' fields.

    Args:
        issue_log:       Structured issue log.
        t:               Current iteration (for bias correction).
        region_rewrites: Region rewrite counter dict.
        beta1:           Momentum decay (mirrors Adam β₁). Default 0.9.
        stuck_threshold: Rewrites before a region is flagged "stuck". Default 3.
    """
    bias_correction = 1 - (beta1 ** t)

    for issue in issue_log:
        if issue["resolved"]:
            continue

        # 1st moment: frequency-weighted raw weight
        raw_weight     = 1 - (beta1 ** issue["frequency"])

        # bias-corrected weight (prevents early iterations over-weighting)
        issue["weight"] = raw_weight / bias_correction

        # variance proxy: flag region if rewritten too many times
        rewrite_count  = region_rewrites.get(issue["region"], 0)
        issue["stuck"] = rewrite_count >= stuck_threshold


# ── FIX B: Require 2 consecutive zero-delta iters before structural rewrite ──
def step_scope(delta: float, any_stuck: bool,
               consecutive_zero: int = 0) -> str:
    """
    Adaptive step controller — determines edit scope for this iteration.
    Mirrors the adaptive learning rate concept from Adam.

    Args:
        delta:            Change in pass rate vs previous iteration.
        any_stuck:        True if any active issue has been flagged as stuck.
        consecutive_zero: Number of consecutive iterations with 0% pass rate.
                          Require 2+ before triggering structural rewrite,
                          so first real attempt gets a chance to improve.

    Returns:
        "targeted"   — make minimal focused edits
        "structural" — discard and redesign the algorithm
    """
    if any_stuck and consecutive_zero >= 2:
        return "structural"     # stuck AND no progress for 2+ iters
    elif delta > 0.1:
        return "targeted"       # good progress → keep going, small edits
    elif delta == 0.0 and consecutive_zero >= 2:
        return "structural"     # completely stuck for 2+ iters → redesign
    else:
        return "targeted"       # give it another targeted attempt first


# ═══════════════════════════════════════════════════════════════
# PARALLEL CRITICS
# ═══════════════════════════════════════════════════════════════

async def concurrent_critics(critic_inputs: dict) -> tuple[str, str, str]:
    """
    Fire all three critics — concurrently first, falls back to sequential
    with inter-call delays if rate limits are hit.

    Args:
        critic_inputs: Dict with keys 'problem', 'code', 'test_results'.

    Returns:
        Tuple of (raw_correctness, raw_edge_cases, raw_complexity) strings.
    """
    import asyncio as _asyncio

    try:
        # attempt concurrent — fastest path
        raw_A, raw_B, raw_C = await _asyncio.gather(
            call_llm(CRITIC_PROMPTS["correctness"], critic_inputs),
            call_llm(CRITIC_PROMPTS["edge_cases"],  critic_inputs),
            call_llm(CRITIC_PROMPTS["complexity"],  critic_inputs),
        )
        return raw_A, raw_B, raw_C

    except RuntimeError as e:
        if "429" not in str(e) and "413" not in str(e):
            raise   # re-raise non-rate-limit errors immediately

        # rate limited — fall back to sequential with delay between calls
        print("  [critics] concurrent hit rate limit, switching to sequential...")
        results = []
        for key in ("correctness", "edge_cases", "complexity"):
            await _asyncio.sleep(2.0)   # 2s gap between sequential calls
            try:
                raw = await call_llm(CRITIC_PROMPTS[key], critic_inputs)
            except Exception as inner_e:
                print(f"  [critics] {key} critic failed: {inner_e}")
                raw = '{"issues": []}'   # empty — fallback handles this
            results.append(raw)

        return results[0], results[1], results[2]


# ═══════════════════════════════════════════════════════════════
# MAIN OPTIMIZATION LOOPS
# ═══════════════════════════════════════════════════════════════

async def ada_textgrad(
    problem:      str,
    initial_code: str,
    test_suite:   list[dict],
    max_iter:     int   = 5,
    beta1:        float = 0.9,
    beta2:        float = 0.999,
    problem_dict: Optional[dict] = None,
) -> tuple[str, dict]:
    """
    AdaTextGrad: Adam-inspired iterative code optimization via textual gradients.

    Args:
        problem:      Problem description string.
        initial_code: Starting code (usually a stub/skeleton).
        test_suite:   List of test case dicts with 'input' and 'expected_output'.
        max_iter:     Maximum optimization iterations. Default 5.
        beta1:        Momentum decay factor. Default 0.9.
        beta2:        Variance decay factor (reserved for future use). Default 0.999.
        problem_dict: Full problem dict from load_leetcode_hard(). Provides
                      prompt_header and entry_point for subprocess execution.
                      Pass None only in unit tests.

    Returns:
        Tuple of (final_code, run_log) where run_log contains full audit trail.
    """
    # import here to avoid circular imports
    from problems import run_tests

    # ── initialise state ────────────────────────────────────────
    code             = initial_code
    issue_log        = []
    region_rewrites  = {}
    pass_history     = []
    structural_count = 0
    llm_calls        = 0
    consecutive_zero: int = 0   # Fix B: consecutive iters with 0% pass rate

    for t in range(1, max_iter + 1):

        # ═══════════════════════════════════════════════════════
        # TIER 1 — FORWARD PASS  (0 LLM calls)
        # ═══════════════════════════════════════════════════════
        test_results = run_tests(code, test_suite, problem=problem_dict)
        pass_rate    = test_results["pass_rate"]
        pass_history.append(pass_rate)

        # compute delta BEFORE the print so it's always defined on iter 1 too
        delta         = (pass_history[-1] - pass_history[-2]) if t > 1 else 0.0
        # Fix B: track consecutive zero pass rate iterations
        if pass_rate == 0.0:
            consecutive_zero += 1
        else:
            consecutive_zero = 0

        active_before = sum(1 for i in issue_log if not i["resolved"])
        print(f"  [iter {t}] pass rate: {pass_rate:.0%}  "
              f"active issues: {active_before}  delta: {delta:+.0%}")

        if pass_rate == 1.0:
            print(f"  ✓ Solved in {t} iteration(s)!")
            break

        # ═══════════════════════════════════════════════════════
        # TIER 2 — PARALLEL CRITICS  (3 LLM calls, concurrent)
        # ═══════════════════════════════════════════════════════
        # use slim_header context for critics — full prompt_header is too large
        # for free-tier Groq token limits
        # only send top 3 failures to critics — more is rarely useful and costs tokens
        top_failures   = test_results["failures"][:3]
        slim_header    = (problem_dict or {}).get("slim_header", "")
        critic_inputs  = {
            "problem":      problem[:2000],   # truncate long descriptions
            "code":         code[:2500],      # truncate very long solutions
            "test_results": json.dumps(top_failures, indent=2),
            "imports":      slim_header,
        }

        raw_A, raw_B, raw_C = await concurrent_critics(critic_inputs)
        llm_calls += 3

        all_new = (
            parse_critic_output(raw_A)
            + parse_critic_output(raw_B)
            + parse_critic_output(raw_C)
        )

        # fallback: if all critics failed (e.g. all 413 errors), inject a generic issue
        # so the optimizer still gets called with something useful
        if not all_new and test_results["failures"]:
            first_fail = test_results["failures"][0]
            all_new = [{
                "description": f"code fails on input: {str(first_fail.get('input',''))[:100]}",
                "region":      "main solution logic",
                "severity":    "high",
                "fix_hint":    f"expected {str(first_fail.get('expected',''))[:80]}, got {str(first_fail.get('got',''))[:80]}",
            }]
            print(f"  [warn] all critics empty — using fallback from test failure. Check API keys/rate limits.")

        # ═══════════════════════════════════════════════════════
        # TIER 3A — UPDATE STRUCTURED LOG  (0–N cheap binary calls)
        # ═══════════════════════════════════════════════════════
        for new_issue in all_new:
            await update_log(issue_log, new_issue, t, region_rewrites)
            llm_calls += 1  # approximate: one dedup call per issue

        # ═══════════════════════════════════════════════════════
        # TIER 3B — MOMENTUM + VARIANCE  (0 LLM calls, pure math)
        # ═══════════════════════════════════════════════════════
        compute_momentum(issue_log, t, region_rewrites, beta1=beta1)

        # ═══════════════════════════════════════════════════════
        # TIER 3C — ADAPTIVE STEP CONTROLLER  (0 LLM calls, pure logic)
        # ═══════════════════════════════════════════════════════
        any_stuck = any(i["stuck"] for i in issue_log if not i["resolved"])
        # Fix B: pass consecutive_zero so structural rewrite needs 2+ stuck iters
        scope     = step_scope(delta, any_stuck, consecutive_zero)

        if scope == "structural":
            structural_count += 1

        # ═══════════════════════════════════════════════════════
        # TIER 4 — GRADIENT AGGREGATOR  (0 LLM calls, pure logic)
        # ═══════════════════════════════════════════════════════
        active      = [i for i in issue_log if not i["resolved"]]
        prioritized  = sorted(active, key=lambda x: x["weight"], reverse=True)
        # Fix C: cap at 10 issues max — more than 10 is gradient noise not signal
        MAX_ISSUES   = 10
        if len(prioritized) > MAX_ISSUES:
            # keep top-weighted; mark excess as resolved to prevent re-adding
            for excess in prioritized[MAX_ISSUES:]:
                excess["resolved"] = True
            prioritized = prioritized[:MAX_ISSUES]
        gradient_text = render_gradient(prioritized, scope)

        # ═══════════════════════════════════════════════════════
        # TIER 5 — ADAPTIVE TGD OPTIMIZER  (1 LLM call)
        # ═══════════════════════════════════════════════════════
        test_failures_str = json.dumps(test_results["failures"][:5], indent=2)
        optimizer_prompt  = build_optimizer_prompt(
            problem, code, test_failures_str, gradient_text, scope, issue_log
        )
        code = await call_llm_optimizer(optimizer_prompt)
        llm_calls += 1

        # strip accidental markdown fences from code output
        if code.strip().startswith("```"):
            lines = code.strip().split("\n")
            code  = "\n".join(
                lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            )

        # ═══════════════════════════════════════════════════════
        # POST-ITERATION — MARK RESOLVED ISSUES  (0 LLM calls)
        # ═══════════════════════════════════════════════════════
        new_results   = run_tests(code, test_suite, problem=problem_dict)
        new_pass_rate = new_results["pass_rate"]

        # Fix 1: structural rewrite = entire algorithm replaced.
        # old issues are irrelevant to the new code — clear the log
        # so stale bugs don't corrupt the next gradient.
        if scope == "structural":
            issue_log.clear()
            region_rewrites.clear()
            print(f"  [structural] issue log reset — old issues cleared")

        # Fix 2: pass rate improved — mark all issues resolved.
        # They will reappear next iter if still real bugs.
        # This replaces the broken region-name string match which
        # compared issue["region"] against test INPUT strings (never matched).
        elif new_pass_rate > pass_history[-1]:
            for issue in issue_log:
                issue["resolved"] = True
            print(f"  [resolved] {pass_history[-1]:.0%}→{new_pass_rate:.0%} "
                  f"— issues cleared, will rebuild next iter")

    run_log = {
        "iterations_used":    t,
        "llm_calls":          llm_calls,
        "structural_rewrites": structural_count,
        "pass_history":       pass_history,
        "issue_log":          issue_log,
    }

    return code, run_log


async def vanilla_textgrad(
    problem:      str,
    initial_code: str,
    test_suite:   list[dict],
    max_iter:     int  = 5,
    problem_dict: Optional[dict] = None,
) -> tuple[str, dict]:
    """
    Vanilla TextGrad baseline — single evaluator, no memory, no adaptation.
    Used as the comparison baseline in experiments.

    Args:
        problem:      Problem description string.
        initial_code: Starting code stub.
        test_suite:   List of test case dicts.
        max_iter:     Maximum iterations. Default 5 (matches AdaTextGrad).
        problem_dict: Full problem dict for test execution context.

    Returns:
        Tuple of (final_code, run_log).
    """
    from problems import run_tests

    code         = initial_code
    pass_history = []
    llm_calls    = 0

    for t in range(1, max_iter + 1):

        test_results = run_tests(code, test_suite, problem=problem_dict)
        pass_rate    = test_results["pass_rate"]
        pass_history.append(pass_rate)

        print(f"  [iter {t}] pass rate: {pass_rate:.0%}")

        if pass_rate == 1.0:
            print(f"  ✓ Solved in {t} iteration(s)!")
            break

        # single evaluator — no momentum, no critics, no memory
        feedback = await call_llm(
            VANILLA_EVALUATOR_PROMPT,
            {
                "problem":      problem[:2000],
                "code":         code[:3000],
                "test_results": json.dumps(test_results["failures"][:3], indent=2),
            },
        )
        llm_calls += 1

        # cap inputs to same budget as AdaTextGrad critics
        # prevents prompt from growing unboundedly across iterations
        optimizer_prompt = VANILLA_OPTIMIZER_PROMPT.format(
            problem       = problem[:2000],
            current_code  = code[:3000],
            feedback      = feedback[:1500],
            test_failures = json.dumps(test_results["failures"][:3], indent=2),
        )
        code = await call_llm_optimizer(optimizer_prompt)
        llm_calls += 1

        if code.strip().startswith("```"):
            lines = code.strip().split("\n")
            code  = "\n".join(
                lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            )

    run_log = {
        "iterations_used": t,
        "llm_calls":       llm_calls,
        "pass_history":    pass_history,
        "issue_log":       [],
    }

    return code, run_log