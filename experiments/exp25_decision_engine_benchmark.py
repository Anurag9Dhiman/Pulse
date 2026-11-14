"""Experiment 25: Decision Engine Benchmark (PULSE v2, Milestone 1).

Does the local RuleDecisionEngine added in Milestone 1 (src/par/core/decision.py)
actually displace any reasoning from the Planner, agree with the Planner on
the goals it does claim, and confirm the "local floor" latency this
milestone exists to establish before any Jev integration is attempted?

Reasoning displacement is the PULSE v2 deck's own headline metric:
R_d = 1 - N_S2 / N_total (the fraction of decisions that did NOT need the
slow/LLM-equivalent path). We measure R_d directly here rather than assert
it, over a deliberately mixed goal set spanning all 6 builtin skills, not
just the two RuleDecisionEngine is designed to resolve - an inflated R_d
from a cherry-picked goal set would misrepresent what this milestone
actually achieves.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from par.core.decision import RuleDecisionEngine
from par.core.observation import Observation
from par.core.planner import RuleBasedPlanner
from par.core.skill import SkillRegistry
from par.evaluation.metrics import latency_stats, rate
from par.skills import builtin_skills

# Deliberately mixed: goals the engine is designed to resolve (stop/detect,
# zero-parameter), goals it must defer on (the other 4 skills, all of which
# require parameters), and edge cases (ambiguous, no match) - see
# RuleDecisionEngine's own docstring for why each category should behave
# as labeled.
GOALS = [
    # zero-parameter: RuleDecisionEngine should confidently resolve these
    ("stop the robot", "stop"),
    ("stop now", "stop"),
    ("please stop", "stop"),
    ("detect objects nearby", "detect"),
    ("detect what is around", "detect"),
    ("run detect", "detect"),
    # parameterized: RuleDecisionEngine must defer (confidence 0.0)
    ("pick the red_object", None),
    ("pick up blue_container", None),
    ("place the object in blue_container", None),
    ("place item at target", None),
    ("move to the target position", None),
    ("move forward", None),
    ("inspect the red_object", None),
    ("inspect blue_container closely", None),
    # ambiguous: two zero-parameter capability names both present
    ("detect then stop", None),
    ("stop and detect", None),
    # no capability keyword present at all
    ("do something unspecified", None),
    ("complete the overall task", None),
]


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    for skill in builtin_skills():
        registry.register(skill)
    return registry


def _obs() -> Observation:
    return Observation(observation_id="bench", timestamp=datetime.now(timezone.utc), source="mock")


def run() -> dict:
    registry = _registry()
    capabilities = registry.capabilities()
    engine = RuleDecisionEngine()
    planner = RuleBasedPlanner()
    obs = _obs()

    per_goal = []
    latencies = []
    for goal, expected_skill_if_confident in GOALS:
        started = time.perf_counter()
        decision = engine.decide(goal, obs, capabilities)
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        confident = decision.skill_name is not None
        agrees_with_planner = None
        if confident:
            # The real correctness question for a System-1 engine: when it
            # claims a goal, does it agree with what System 2 (here,
            # RuleBasedPlanner standing in for the LLM planner) would have
            # said anyway? Disagreement would mean displacing reasoning
            # changes the outcome, not just who computed it.
            planner_skill, _ = planner.propose(goal, obs, capabilities)
            agrees_with_planner = planner_skill == decision.skill_name

        per_goal.append(
            {
                "goal": goal,
                "expected_confident_skill": expected_skill_if_confident,
                "decision_confident": confident,
                "decision_skill_name": decision.skill_name,
                "decision_confidence": decision.confidence,
                "matches_expectation": confident == (expected_skill_if_confident is not None)
                and (decision.skill_name == expected_skill_if_confident if confident else True),
                "agrees_with_planner": agrees_with_planner,
                "latency_seconds": elapsed,
            }
        )

    n_total = len(GOALS)
    n_confident = sum(1 for g in per_goal if g["decision_confident"])
    confident_rows = [g for g in per_goal if g["decision_confident"]]

    return {
        "experiment": "25_decision_engine_benchmark",
        "engine": "RuleDecisionEngine",
        "n_goals": n_total,
        "goal_mix": "6 zero-parameter (should resolve), 8 parameterized (must defer), "
        "2 ambiguous (must defer), 2 no-match (must defer)",
        "reasoning_displacement_rate": rate(n_confident, n_total),
        "n_confident_resolved_locally": n_confident,
        "n_deferred_to_planner": n_total - n_confident,
        "all_confident_decisions_match_expected_skill": all(
            g["decision_skill_name"] == g["expected_confident_skill"] for g in confident_rows
        ),
        "all_confident_decisions_agree_with_planner": all(g["agrees_with_planner"] for g in confident_rows),
        "all_goals_matched_expectation": all(g["matches_expectation"] for g in per_goal),
        "latency_stats_seconds": latency_stats(latencies),
        "per_goal": per_goal,
        "note": (
            "reasoning_displacement_rate here is specific to this deliberately "
            "mixed 18-goal set, not a claim about real-task distributions - "
            "RuleDecisionEngine only ever resolves the subset of goals that "
            "are zero-parameter and unambiguous, by design (see its own "
            "docstring). latency_stats_seconds is the real measured local "
            "floor this milestone exists to establish, to compare a future "
            "JevDecisionEngine against - not a simulated or estimated number."
        ),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp25_decision_engine_benchmark.json"
    out_path.write_text(json.dumps(result, indent=2))
