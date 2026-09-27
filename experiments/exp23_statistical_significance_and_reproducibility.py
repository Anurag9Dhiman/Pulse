"""Experiment 23: Statistical Significance and Reproducibility.

Cross-cutting by design (per the validation plan), not a new scenario of its
own. Rather than assert that this suite is reproducible and well-reported,
this script actually re-runs code and diffs results:

1. Exact reproducibility: a sample of fully deterministic experiments
   (Experiments 1 and 8 - seeded benchmark generation, zero wall-clock
   dependency in their reported fields) must produce bit-identical results
   across two independent calls to run(). Experiments 20 and 21 are also
   checked, with wall-clock latency fields excluded from the comparison
   (real timing legitimately varies run to run; nothing else should).

2. Seeded-RNG reproducibility: Experiment 19 is the one script in this suite
   that draws from random.Random(SEED). Same seed must reproduce identical
   results; a different seed must still preserve the outcomes that are
   deterministic by that experiment's own design (capability_restricted's 0%
   success, since the one capability its planner uses is unregistered
   regardless of any random draw elsewhere).

3. Statistical reporting audit: a recursive scan of every already-produced
   experiments/results/*.json file for dicts matching
   par.evaluation.metrics.SummaryStats's shape (n/mean/median/stdev/
   ci95_half_width) - the shared format this suite standardized on - counted
   directly from what's on disk, not from memory of what each script does.

4. Provenance: the exact Python version, platform, git commit, and key
   package versions in effect for this run, so any result in this suite can
   be reproduced exactly.
"""
from __future__ import annotations

import importlib
import json
import platform
import random
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Running this script directly (python experiments/exp23_....py) puts only
# experiments/ itself on sys.path, not its parent - so "import experiments.*"
# (needed to call sibling experiment scripts' run() functions) fails unless
# the project root is added explicitly. No other script in this suite needs
# this, since none of them import each other.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
DETERMINISTIC_NO_LATENCY = ["exp01_baseline_safety_interception", "exp08_safety_kernel_ablation"]
DETERMINISTIC_WITH_LATENCY = ["exp20_scaling_evaluation", "exp21_long_horizon_evaluation"]
STAT_SUMMARY_KEYS = {"n", "mean", "median", "stdev", "ci95_half_width"}


def _strip_latency(obj):
    """Removes wall-clock-timing fields before an exact equality check -
    real latency legitimately varies run to run even when everything else
    about a deterministic experiment is bit-identical."""
    if isinstance(obj, dict):
        return {k: _strip_latency(v) for k, v in obj.items() if "latency" not in k.lower()}
    if isinstance(obj, list):
        return [_strip_latency(v) for v in obj]
    return obj


def _check_exact_reproducibility(module_name: str, strip_latency: bool) -> dict:
    module = importlib.import_module(f"experiments.{module_name}")
    run_a, run_b = module.run(), module.run()
    if strip_latency:
        run_a, run_b = _strip_latency(run_a), _strip_latency(run_b)
    return {
        "experiment": module_name,
        "identical_on_rerun": run_a == run_b,
        "latency_fields_excluded_from_comparison": strip_latency,
    }


def _check_seed_reproducibility() -> dict:
    module = importlib.import_module("experiments.exp19_task_diverse_benchmark")
    original_seed = module.SEED
    try:
        same_seed_a = module.run()
        same_seed_b = module.run()
        module.SEED = 999
        different_seed = module.run()
    finally:
        module.SEED = original_seed

    def _capability_restricted_rate(result: dict) -> float:
        return result["results_by_family"]["capability_restricted"]["task_success_rate"]

    # exp19's own aggregated per-family fields are deliberately invariant to
    # the *exact* value random.uniform draws for the "reaching" family - any
    # x in its safe range (-0.3,-0.05) ALLOWs and succeeds the same way, that
    # invariance is the point of the family's design. Comparing aggregates
    # wouldn't detect a seed change even though the seed genuinely changes
    # what's drawn, so this checks the draw directly at its source instead.
    draw_a, _ = module._build_trial("reaching", random.Random(original_seed))
    draw_b, _ = module._build_trial("reaching", random.Random(999))
    move_a = draw_a._steps[0][1]
    move_b = draw_b._steps[0][1]

    return {
        "same_seed_identical": same_seed_a == same_seed_b,
        "different_seed_preserves_capability_restricted_invariant": (
            _capability_restricted_rate(same_seed_a) == 0.0 and _capability_restricted_rate(different_seed) == 0.0
        ),
        "different_seed_changes_underlying_random_draw": move_a != move_b,
        "note": (
            "exp19's own aggregated per-family fields (n_actions_proposed, "
            "safety_outcomes, final success) are deliberately invariant to "
            "the exact value drawn within a family's safe range, so a direct "
            "check against the random draw itself is used here instead of "
            "comparing run() output, which wouldn't detect a seed change."
        ),
    }


def _count_stat_summaries(obj) -> int:
    count = 0
    if isinstance(obj, dict):
        if STAT_SUMMARY_KEYS.issubset(obj.keys()):
            count += 1
        for value in obj.values():
            count += _count_stat_summaries(value)
    elif isinstance(obj, list):
        for value in obj:
            count += _count_stat_summaries(value)
    return count


def _audit_statistical_reporting() -> dict:
    per_file = {}
    for path in sorted(RESULTS_DIR.glob("exp*.json")):
        if path.stem == "exp23_statistical_significance_and_reproducibility":
            continue
        per_file[path.stem] = _count_stat_summaries(json.loads(path.read_text()))
    return per_file


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except Exception as exc:  # pragma: no cover - environment-dependent
        return f"unavailable: {exc}"


def _package_versions() -> dict:
    packages = ["pydantic", "pyyaml", "python-dotenv", "pytest", "anthropic", "google-genai"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def _provenance() -> dict:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
    }


def run() -> dict:
    exact_repro = [_check_exact_reproducibility(name, strip_latency=False) for name in DETERMINISTIC_NO_LATENCY] + [
        _check_exact_reproducibility(name, strip_latency=True) for name in DETERMINISTIC_WITH_LATENCY
    ]
    seed_repro = _check_seed_reproducibility()
    stat_audit = _audit_statistical_reporting()

    return {
        "experiment": "23_statistical_significance_and_reproducibility",
        "exact_reproducibility_checks": exact_repro,
        "all_exact_reproducibility_checks_passed": all(c["identical_on_rerun"] for c in exact_repro),
        "seeded_experiment_reproducibility": seed_repro,
        "statistical_reporting_audit": {
            "stat_summary_dicts_per_result_file": stat_audit,
            "files_with_zero_stat_summary_dicts": [name for name, n in stat_audit.items() if n == 0],
            "note": (
                "A 'stat summary dict' matches par.evaluation.metrics."
                "SummaryStats's shape (n/mean/median/stdev/ci95_half_width) - "
                "the shared reporting format this suite standardized on. A "
                "file with 0 is not necessarily non-compliant: Experiments "
                "16-18 are explicitly-documented methodological/not-run "
                "entries with no numeric trial data to summarize (see their "
                "own 'status' fields), and some experiments report "
                "classification metrics (confusion matrices, precision/"
                "recall) for which mean/stdev is the wrong format, not a "
                "missing one."
            ),
        },
        "provenance_for_exact_reproduction": _provenance(),
        "note": (
            "This experiment verifies, by actually re-running code rather "
            "than asserting it, that (a) deterministic experiments in this "
            "suite reproduce bit-identical results run-to-run once wall-clock "
            "latency fields are excluded, (b) the one seeded-RNG experiment "
            "(19) reproduces identically under a fixed seed and preserves its "
            "by-design invariants under a different one, and (c) which result "
            "files already carry the shared statistical reporting format. It "
            "also persists the exact software/environment provenance needed "
            "to reproduce any run in this suite."
        ),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = RESULTS_DIR / "exp23_statistical_significance_and_reproducibility.json"
    out_path.write_text(json.dumps(result, indent=2))
