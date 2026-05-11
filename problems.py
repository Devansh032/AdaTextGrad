"""
problems.py
───────────
Dataset loading, test execution, experiment orchestration, and metrics.

Contains:
  - _parse_input_output()       filter + reformat dataset I/O pairs
  - load_leetcode_hard()        load problems from local JSON
  - load_from_huggingface()     download + format from HuggingFace (exact schema)
  - run_tests()                 execute code against test cases in subprocess
  - run_experiment()            orchestrate full AdaTextGrad / vanilla run
  - compute_metrics()           solve rate + efficiency metrics
  - print_comparison()          pretty-print comparison table

DATASET
───────
Source: https://huggingface.co/datasets/newfacade/LeetCodeDataset  (split="test")
228 problems total, filter to difficulty=="Hard" gives ~39 usable Hard problems.

Exact dataset fields used here:
    task_id           str   slug  e.g. "shortest-distance-after-road-addition-queries-ii"
    question_id       int         e.g. 3244
    difficulty        str         "Easy" | "Medium" | "Hard"
    tags              list        e.g. ["Greedy", "Graph", "Array", "Ordered Set"]
    problem_description str       full problem statement
    starter_code      str         class Solution stub
    prompt            str         boilerplate imports + starter_code (run header)
    completion        str         reference solution
    entry_point       str         e.g. "Solution().shortestDistanceAfterQueries"
    test              str         check(candidate) function as Python source
    input_output      list        [{input: str, output: str}, ...]
    query             str         full prompt string (unused)
    response          str         empty in test split

input_output notes:
  - input  is a kwarg string e.g. "n = 5, queries = [[2,4],[0,2],[0,4]]"
  - output is the expected result string e.g. "[3, 2, 1]"
  - some output values are "Execution timed out" or "Error: ..." -- skip these
"""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# DATASET HELPERS
# ═══════════════════════════════════════════════════════════════


# Essential imports that cover 99% of LeetCode problems
# Used when sending context to LLM critics (keeps token count low)
_SLIM_IMPORTS = """from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque, Counter, OrderedDict
from itertools import combinations, permutations, product, accumulate
from heapq import heappush, heappop, heapify
from bisect import bisect_left, bisect_right, insort
from functools import lru_cache, reduce
from math import inf, gcd, lcm, sqrt, log2, comb
import math, sys, string

inf = float('inf')
"""


def make_slim_header() -> str:
    """Return minimal imports string for sending to LLM critics."""
    return _SLIM_IMPORTS

def _parse_input_output(io_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Filter and reformat the dataset's input_output list.

    Skips entries whose output is:
      - "Execution timed out"   (broken reference case in dataset)
      - starts with "Error"     (dataset-side execution error)

    Our output format per test case:
        {"input": "n = 5, queries = [[2,4],[0,2],[0,4]]",
         "expected_output": "[3, 2, 1]"}

    Returns:
        (visible_tests, hidden_tests) split 80/20 on usable entries.
    """
    usable = []
    for io in (io_list or []):
        out = str(io.get("output") or "")
        if out.startswith("Execution timed out") or out.startswith("Error"):
            continue
        if not out.strip():
            continue
        usable.append({
            "input":           str(io["input"]),
            "expected_output": out,
        })

    if not usable:
        return [], []

    split   = max(1, int(len(usable) * 0.8))
    visible = usable[:split]
    hidden  = usable[split:] or usable[:1]   # guarantee at least 1 hidden test

    return visible, hidden


def _build_exec_script(
    prompt_header: str,
    code:          str,
    entry_point:   str,
    inp:           str,
    expected:      str,
) -> str:
    """
    Build a self-contained Python script that:
      1. Runs the boilerplate imports from prompt_header
      2. Defines the Solution class (from `code`)
      3. Calls entry_point with the kwarg string `inp`
      4. Compares result to expected and prints PASS / FAIL / ERROR

    The dataset input format is kwargs, not positional args:
        inp = "n = 5, queries = [[2,4],[0,2],[0,4]]"
    We call: Solution().methodName(n=5, queries=[[2,4],[0,2],[0,4]])
    by doing: eval("dict(n=5, queries=[[2,4],[0,2],[0,4]])")

    entry_point format: "Solution().methodName"
    We split on "(). " to get instance + method.
    """
    if "()." in entry_point:
        cls_part, method_name = entry_point.split("().", 1)
        instance_expr = cls_part + "()"
    else:
        instance_expr = "Solution()"
        method_name   = entry_point

    script = f'''\
{prompt_header}

{code}

try:
    _kwargs   = dict({inp})
    _instance = {instance_expr}
    _result   = getattr(_instance, "{method_name}")(**_kwargs)
    _expected = {expected}
    if _result == _expected:
        print("PASS")
    else:
        print(f"FAIL: got {{repr(_result)}}, expected {{repr(_expected)}}")
except Exception as _e:
    import traceback
    print(f"ERROR: {{_e}}")
'''
    return script


# ═══════════════════════════════════════════════════════════════
# DATASET LOADER
# ═══════════════════════════════════════════════════════════════

def load_leetcode_hard(path: str = "leetcode_hard_39.json") -> list[dict]:
    """
    Load problems from a local JSON file saved by load_from_huggingface().

    Args:
        path: Path to JSON file. Default "leetcode_hard_39.json".

    Returns:
        List of problem dicts.

    Raises:
        FileNotFoundError with instructions if file missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\nDataset file not found: '{path}'\n\n"
            "Generate it by running once:\n\n"
            "    from problems import load_from_huggingface\n"
            "    load_from_huggingface()   # saves leetcode_hard_39.json\n\n"
            "Requires: pip install datasets\n"
            "Source:   https://huggingface.co/datasets/newfacade/LeetCodeDataset\n"
        )

    with open(path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print(f"[problems] Loaded {len(problems)} problems from '{path}'")
    return problems


def load_from_huggingface(
    n:          int = 39,
    difficulty: str = "Hard",
    save_path:  str = "leetcode_hard_39.json",
) -> list[dict]:
    """
    Download and format problems from HuggingFace newfacade/LeetCodeDataset.
    Requires: pip install datasets

    Dataset schema (verified against actual dataset):
        task_id             str   problem slug
        question_id         int   numeric ID
        difficulty          str   "Easy" | "Medium" | "Hard"
        tags                list  topic tags
        problem_description str   full problem statement
        starter_code        str   class Solution stub
        prompt              str   boilerplate imports + starter (use as exec header)
        completion          str   reference solution
        entry_point         str   e.g. "Solution().shortestDistanceAfterQueries"
        test                str   check(candidate) Python source
        input_output        list  [{input: str, output: str}]
        query               str   full prompt (unused)
        response            str   empty in test split

    Args:
        n:          Number of Hard problems to collect. Default 39.
        difficulty: Difficulty filter. Default "Hard".
        save_path:  Where to save formatted JSON for local runs.

    Returns:
        List of problem dicts.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    print("[problems] Downloading LeetCodeDataset from HuggingFace...")
    ds   = load_dataset("newfacade/LeetCodeDataset", split="test")
    hard = ds.filter(lambda x: x.get("difficulty") == difficulty)
    print(f"[problems] Found {len(hard)} {difficulty} problems in dataset.")

    problems = []
    skipped  = 0

    for i, item in enumerate(hard):
        if len(problems) >= n:
            break

        # parse I/O pairs — filters out "Execution timed out" and "Error" entries
        io_list                     = item.get("input_output") or []
        visible_tests, hidden_tests = _parse_input_output(io_list)

        # skip problems with no usable test cases at all
        if not visible_tests or not hidden_tests:
            skipped += 1
            continue

        problems.append({
            # identifiers
            "problem_id":  str(item.get("question_id") or i + 1),
            "title":       item.get("task_id") or f"problem_{i+1}",
            "tags":        item.get("tags") or [],

            # problem content
            "description": item.get("problem_description") or "",

            # execution context
            # prompt_header = full dataset prompt (used ONLY in subprocess execution)
            # slim_header   = minimal imports (used when sending to LLM critics to save tokens)
            "prompt_header":      item.get("prompt") or "",
            "slim_header":        make_slim_header(),
            "starter_code":       item.get("starter_code") or "",

            # how to invoke the solution — e.g. "Solution().shortestDistanceAfterQueries"
            "entry_point":        item.get("entry_point") or "Solution().solve",

            # reference solution — for sanity checking, not used in optimization
            "reference_solution": item.get("completion") or "",

            # test strings (check() function source) — kept for reference
            "test_fn_source":     item.get("test") or "",

            # structured test cases derived from input_output field
            "test_cases":   visible_tests,   # used during optimization iterations
            "hidden_tests": hidden_tests,    # held-out, used only for final eval
        })

    if skipped:
        print(f"[problems] Skipped {skipped} problems (no usable I/O pairs after filtering).")
    print(f"[problems] Collected {len(problems)} problems.")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)
    print(f"[problems] Saved to '{save_path}'")

    return problems


# ═══════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════

def run_tests(
    code:       str,
    test_cases: list[dict],
    problem:    Optional[dict] = None,
) -> dict:
    """
    Execute code against test cases in isolated subprocesses.

    Handles the dataset's kwarg-style input format:
        "n = 5, queries = [[2,4],[0,2],[0,4]]"
    and correct entry_point dispatch:
        "Solution().shortestDistanceAfterQueries"

    If `problem` is None (unit test mode), uses a simple fallback that
    calls Solution().solve() with positional args — only for tests.py.

    Args:
        code:       Python code string defining class Solution.
        test_cases: List of {input, expected_output} dicts.
        problem:    Full problem dict. Provides prompt_header + entry_point.
                    Pass None only in isolated unit tests.

    Returns:
        Dict:
            pass_rate     float  0.0-1.0
            failures      list   [{input, expected, got}]
            passing_cases list   [input strings that passed]
            total         int
    """
    if not test_cases:
        return {"pass_rate": 0.0, "failures": [], "passing_cases": [], "total": 0}

    # execution context from problem dict
    if problem:
        prompt_header = problem.get("prompt_header", "")
        entry_point   = problem.get("entry_point", "Solution().solve")
    else:
        # unit test fallback — simple imports, positional solve()
        prompt_header = "from typing import *\nfrom collections import *\nfrom heapq import *"
        entry_point   = "Solution().solve"

    results = {
        "pass_rate":      0.0,
        "failures":       [],
        "passing_cases":  [],
        "total":          len(test_cases),
    }

    passed = 0

    for tc in test_cases:
        inp      = tc.get("input", "")
        expected = tc.get("expected_output", "")

        if problem:
            # real dataset mode: kwargs call via _build_exec_script
            script = _build_exec_script(
                prompt_header = prompt_header,
                code          = code,
                entry_point   = entry_point,
                inp           = inp,
                expected      = expected,
            )
        else:
            # unit test fallback: positional call Solution().solve(inp)
            script = f"""\
{prompt_header}

{code}

try:
    result   = Solution().solve({inp})
    expected = {expected}
    if result == expected:
        print("PASS")
    else:
        print(f"FAIL: got {{repr(result)}}, expected {{repr(expected)}}")
except Exception as e:
    print(f"ERROR: {{e}}")
"""

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(script)
                tmp_path = f.name

            proc = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = proc.stdout.strip()
            if not output and proc.stderr.strip():
                output = f"ERROR: {proc.stderr.strip()[:300]}"

        except subprocess.TimeoutExpired:
            output = "ERROR: timeout (TLE)"
        except Exception as e:
            output = f"ERROR: subprocess failed: {e}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if output == "PASS":
            passed += 1
            results["passing_cases"].append(inp)
        else:
            results["failures"].append({
                "input":    inp,
                "expected": expected,
                "got":      output,
            })

    results["pass_rate"] = passed / len(test_cases)
    return results


def verify_reference_solution(problem: dict) -> dict:
    """
    Run the dataset reference solution against visible test cases.
    Use this to confirm the test runner works correctly before optimizing.

    Args:
        problem: Problem dict from load_leetcode_hard().

    Returns:
        run_tests result dict.
    """
    ref_code = problem.get("reference_solution", "")
    if not ref_code:
        return {"error": "no reference_solution in problem dict"}

    result = run_tests(ref_code, problem["test_cases"], problem=problem)
    status = "PASS" if result["pass_rate"] == 1.0 else "PARTIAL"
    print(f"  [{status}] reference: {result['pass_rate']:.0%} "
          f"({len(result['passing_cases'])}/{result['total']} visible tests)")
    if result["failures"]:
        print(f"  First failure: {result['failures'][0]}")
    return result


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════

async def run_experiment(
    method:           str   = "ada_textgrad",
    max_iter:         int   = 5,
    problems_path:    str   = "leetcode_hard_39.json",
    results_dir:      str   = "results",
    limit:            Optional[int] = None,
    verify_refs:      bool  = False,
    start_from:       int   = 0,
    cooldown_seconds: float = 8.0,
) -> list[dict]:
    """
    Run AdaTextGrad or vanilla TextGrad on the full problem set.
    Saves results incrementally to JSONL so no progress is lost on API failures.

    Args:
        method:           "ada_textgrad" or "vanilla_textgrad".
        max_iter:         Max iterations per problem. Default 5.
        problems_path:    Path to problems JSON file.
        results_dir:      Directory for result JSONL files.
        limit:            Only run first N problems (quick testing). Default: all.
        verify_refs:      Run reference solutions first to sanity-check test runner.
        start_from:       Skip first N problems (resume after a crash). Default 0.
        cooldown_seconds: Sleep between problems to avoid hitting daily token limit.
                          Default 8s — reduce if you have more API keys.

    Returns:
        List of result dicts.
    """
    from main import ada_textgrad, vanilla_textgrad
    from llm_client import (snapshot_and_reset_tokens, reset_call_count,
                             QuotaExhaustedException)
    reset_call_count()   # start fresh counters for this experiment

    problems = load_leetcode_hard(problems_path)
    if start_from > 0:
        print(f"[run_experiment] Resuming from problem index {start_from} "
              f"(skipping first {start_from} problems)")
        problems = problems[start_from:]
    if limit:
        problems = problems[:limit]

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(results_dir) / f"{method}.jsonl"

    results = []
    total_problems = len(problems)

    for prob_idx, prob in enumerate(problems):
        print(f"\n[{prob['problem_id']}] {prob['title']}")
        if prob.get("tags"):
            print(f"  tags: {', '.join(prob['tags'])}")

        if verify_refs:
            verify_reference_solution(prob)

        start = time.time()

        try:
            if method == "ada_textgrad":
                final_code, run_log = await ada_textgrad(
                    problem      = prob["description"],
                    initial_code = prob["starter_code"],
                    test_suite   = prob["test_cases"],
                    max_iter     = max_iter,
                    problem_dict = prob,
                )
            else:
                final_code, run_log = await vanilla_textgrad(
                    problem      = prob["description"],
                    initial_code = prob["starter_code"],
                    test_suite   = prob["test_cases"],
                    max_iter     = max_iter,
                    problem_dict = prob,
                )

            # snapshot token usage for this problem before hidden eval
            token_usage  = snapshot_and_reset_tokens()
            hidden_result = run_tests(final_code, prob["hidden_tests"], problem=prob)

            record = {
                "problem_id":          prob["problem_id"],
                "title":               prob["title"],
                "tags":                prob.get("tags", []),
                "method":              method,
                "solved":              hidden_result["pass_rate"] == 1.0,
                "final_pass_rate":     hidden_result["pass_rate"],
                "iterations_used":     run_log["iterations_used"],
                "llm_calls":           token_usage["llm_calls"],
                "prompt_tokens":       token_usage["prompt_tokens"],
                "completion_tokens":   token_usage["completion_tokens"],
                "total_tokens":        token_usage["total_tokens"],
                "structural_rewrites": run_log.get("structural_rewrites", 0),
                "time_seconds":        round(time.time() - start, 2),
                "pass_history":        run_log["pass_history"],
                "issue_log":           run_log.get("issue_log", []),
                "error":               None,
            }

        except QuotaExhaustedException as e:
            # daily quota exhausted — stop cleanly, do NOT write a failed record
            # so --resume picks up from this exact problem tomorrow
            completed = prob_idx + (start_from or 0)
            print(f"\n{'='*55}")
            print(f"  QUOTA EXHAUSTED — stopping after {completed} problems")
            print(f"  Progress saved to {output_path}")
            print(f"  Resume tomorrow with:")
            print(f"    python run.py --method {method} --max_iter {max_iter} "
                  f"--cooldown {cooldown_seconds:.0f} --resume")
            print(f"{'='*55}\n")
            break   # exit problem loop, return what we have so far

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            token_usage = snapshot_and_reset_tokens()
            record = {
                "problem_id":          prob["problem_id"],
                "title":               prob["title"],
                "tags":                prob.get("tags", []),
                "method":              method,
                "solved":              False,
                "final_pass_rate":     0.0,
                "iterations_used":     0,
                "llm_calls":           token_usage["llm_calls"],
                "prompt_tokens":       token_usage["prompt_tokens"],
                "completion_tokens":   token_usage["completion_tokens"],
                "total_tokens":        token_usage["total_tokens"],
                "structural_rewrites": 0,
                "time_seconds":        round(time.time() - start, 2),
                "pass_history":        [],
                "issue_log":           [],
                "error":               str(e),
            }

        results.append(record)
        status = "✓ SOLVED" if record["solved"] else "✗ unsolved"
        print(f"  {status} | hidden pass: {record['final_pass_rate']:.0%} | "
              f"iters: {record['iterations_used']} | calls: {record['llm_calls']} | "
              f"tokens: {record['total_tokens']:,}")

        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # inter-problem cooldown — prevents burning daily token quota too fast
        if cooldown_seconds > 0 and prob_idx < total_problems - 1:
            remaining = total_problems - prob_idx - 1
            print(f"  [cooldown] waiting {cooldown_seconds:.0f}s "
                  f"({remaining} problems remaining)...")
            await asyncio.sleep(cooldown_seconds)

    print(f"\n[run_experiment] Done. Results saved to {output_path}")
    return results


def load_results(path: str) -> list[dict]:
    """Load a JSONL results file back into a list of dicts."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def count_early_resolutions(issue_log: list[dict], iterations_used: int) -> int:
    """Count issues resolved before the final iteration."""
    return sum(
        1 for i in issue_log
        if i.get("resolved")
        and max(i.get("iterations_seen", [0])) < iterations_used
    )


def _avg(values: list) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    return sum(filtered) / len(filtered) if filtered else None


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute all metrics from a results list.

    Args:
        results: List of result dicts from run_experiment().

    Returns:
        Dict of metric -> value.
    """
    if not results:
        return {}

    solved   = [r for r in results if r["solved"]]
    unsolved = [r for r in results if not r["solved"]]

    return {
        # primary metric — matches TextGrad paper
        "solve_rate": len(solved) / len(results),

        # efficiency
        "avg_iterations_to_solve": _avg(
            [r["iterations_used"] for r in solved]
        ),
        "avg_llm_calls_per_problem": _avg(
            [r["llm_calls"] for r in results]
        ),
        "avg_time_seconds": _avg(
            [r["time_seconds"] for r in results]
        ),

        # AdaTextGrad-specific
        "structural_rewrite_trigger_rate": _avg([
            r["structural_rewrites"] / max(r["iterations_used"], 1)
            for r in results
            if r.get("structural_rewrites") is not None
        ]),
        "avg_issues_resolved_early": _avg([
            count_early_resolutions(r["issue_log"], r["iterations_used"])
            for r in results
        ]),

        # breakdown
        "solved_count":   len(solved),
        "unsolved_count": len(unsolved),
        "total":          len(results),
    }


def print_comparison(ada_results: list[dict], tg_results: list[dict]) -> None:
    """Print a comparison table between AdaTextGrad and vanilla TextGrad."""
    ada = compute_metrics(ada_results)
    tg  = compute_metrics(tg_results)

    def fmt(val, fmt_str=".1%"):
        if val is None:
            return "  N/A  "
        return format(val, fmt_str).rjust(8)

    print("""
┌─────────────────────────────────────┬──────────────┬──────────────┐
│ Metric                              │  TextGrad    │ AdaTextGrad  │
├─────────────────────────────────────┼──────────────┼──────────────┤
│ Solve rate                          │{tg_sr}      │{ada_sr}      │
│ Avg iterations to solve             │{tg_it}      │{ada_it}      │
│ Avg LLM calls per problem           │{tg_lc}      │{ada_lc}      │
│ Avg time per problem (s)            │{tg_tm}      │{ada_tm}      │
│ Structural rewrite trigger rate     │   N/A        │{ada_str}      │
│ Problems solved                     │{tg_sc}      │{ada_sc}      │
└─────────────────────────────────────┴──────────────┴──────────────┘""".format(
        tg_sr  = fmt(tg.get("solve_rate")),
        ada_sr = fmt(ada.get("solve_rate")),
        tg_it  = fmt(tg.get("avg_iterations_to_solve"),  ".2f"),
        ada_it = fmt(ada.get("avg_iterations_to_solve"), ".2f"),
        tg_lc  = fmt(tg.get("avg_llm_calls_per_problem"),  ".1f"),
        ada_lc = fmt(ada.get("avg_llm_calls_per_problem"), ".1f"),
        tg_tm  = fmt(tg.get("avg_time_seconds"),  ".1f"),
        ada_tm = fmt(ada.get("avg_time_seconds"), ".1f"),
        ada_str= fmt(ada.get("structural_rewrite_trigger_rate")),
        tg_sc  = f"  {tg.get('solved_count', 0)}/{tg.get('total', 0)}   ".rjust(8),
        ada_sc = f"  {ada.get('solved_count', 0)}/{ada.get('total', 0)}   ".rjust(8),
    ))