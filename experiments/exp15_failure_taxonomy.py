"""Experiment 15: Failure and Error Taxonomy.

What are the dominant failure modes of an LLM-directed physical agent
operating through PAR? Rather than generating new synthetic failures, this
aggregates every real failure instance already surfaced by Experiments 1-14
(and by building this validation suite itself) into the 13 named categories,
citing where each one came from. A failure taxonomy built from invented
examples would be far less informative than one built from failures that
actually happened during this session.
"""
from __future__ import annotations

import json
from pathlib import Path

CATEGORIES = {
    "planner_failure": [
        {
            "source": "Experiment 5",
            "description": (
                "RuleBasedPlanner has no memory of prior steps and re-scans the "
                "same goal string every call; on a 2-step goal it proposed 'pick' "
                "twice instead of pick-then-place. Recoverable: no (this planner "
                "cannot represent task progress at all)."
            ),
        },
    ],
    "incorrect_skill_selection": [
        {
            "source": "Experiment 5",
            "description": "Same root cause as planner_failure above - proposing 'pick' when 'place' was needed.",
        },
    ],
    "incorrect_parameters": [],
    "repeated_unsafe_proposal": [
        {
            "source": "Experiment 10",
            "description": (
                "Adversarial planner deliberately repeated an identical denied "
                "workspace-violation proposal 5 times across the trial; correctly "
                "denied identically every time (0% bypass). Recoverable: n/a by "
                "design (this planner never adapts)."
            ),
        },
    ],
    "premature_task_completion": [
        {
            "source": "Experiment 22",
            "description": (
                "Overconfident planner called task_complete(success=True) "
                "regardless of actual outcome in 66/100 trials where the goal was "
                "not actually achieved. Recoverable: yes, by construction - this "
                "is exactly what environment-verification (Experiment 22's "
                "mechanism) catches rather than trusting the self-report."
            ),
        },
    ],
    "perception_error": [],
    "stale_observation": [],
    "safety_kernel_false_denial": [
        {
            "source": "Experiment 13",
            "description": (
                "False-denial rate climbs from 0% (margin=0.05) to 56% "
                "(margin=0.50) as collision_margin grows - a direct, quantified "
                "safety/efficiency tradeoff, not a bug. Recoverable: yes, tune "
                "collision_margin per deployment."
            ),
        },
    ],
    "missed_safety_violation": [
        {
            "source": "Experiment 13",
            "description": (
                "collision_margin=0.05 misses an 11% ground-truth collision rate "
                "that any margin>=0.10 fully catches. Recoverable: yes, same "
                "tuning axis as above - this is why Experiment 13 exists."
            ),
        },
        {
            "source": "Experiment 8 (ablation)",
            "description": (
                "Removing Admission, collision-checking, or velocity-checking "
                "each independently raises unsafe-execution-rate by 20 points "
                "over the full_par baseline on the same benchmark. Recoverable: "
                "yes - don't ablate the Safety Kernel in production; this "
                "quantifies why each component matters."
            ),
        },
    ],
    "runtime_timeout": [
        {
            "source": "tests/test_runtime.py::test_action_timeout_produces_failed_result",
            "description": (
                "A robot call exceeding the profile's action_timeout_seconds "
                "correctly produces a 'action timed out' failed ActionResult "
                "rather than hanging. Recoverable: yes, same as any other "
                "execution failure - fed back for re-planning, not fatal by "
                "itself."
            ),
        },
    ],
    "api_failure": [
        {
            "source": "paper/par_paper.tex Section VI-C (live testing)",
            "description": (
                "google-genai's send_message() rejected schema-compliant plain "
                "dicts at runtime despite its own type hints advertising "
                "PartDict support; gemini-2.5-flash returned 404 'no longer "
                "available to new users'. Both found only by calling the real "
                "API, invisible to fake-client unit tests. Recoverable: yes, "
                "fixed in code (real Part objects, updated default model)."
            ),
        },
        {
            "source": "Experiment 5 (this session)",
            "description": (
                "Hit a real 429 RESOURCE_EXHAUSTED from the Gemini free tier "
                "(15 requests/minute) mid-run. Recoverable: yes - the experiment "
                "script now catches this, reports how many trials completed "
                "before quota ran out, and does not crash or fabricate the "
                "remaining trials."
            ),
        },
    ],
    "ros2_integration_failure": [
        {
            "source": "N/A - not exercised",
            "description": (
                "No ROS 2 installation is available in this environment. The "
                "ROS 2 adapter's message-mapping functions are unit-tested; the "
                "pub/sub node lifecycle itself is unverified (see Experiment "
                "17). Not fabricated here."
            ),
        },
    ],
    "robot_execution_failure": [
        {
            "source": "Experiments 3, 4, 5, 7",
            "description": (
                "Genuine MockRobot execution failures (gripper already full on "
                "a second pick, denied moves) correctly set AgentStatus.FAILED "
                "via Agent.record_result and stop the task loop, distinct from "
                "a recoverable Safety Kernel denial (Agent.record_rejection). "
                "Recoverable: depends on the planner's own reasoning - "
                "RecoveringMovePlanner and live Gemini both demonstrated "
                "recovery from denials; a genuine execution failure like a "
                "repeated pick was not designed to be recovered from by any "
                "planner used in this suite."
            ),
        },
    ],
}


def run() -> dict:
    counts = {category: len(instances) for category, instances in CATEGORIES.items()}
    total_instances = sum(counts.values())
    exercised = [c for c, n in counts.items() if n > 0]
    not_exercised = [c for c, n in counts.items() if n == 0]

    return {
        "experiment": "15_failure_and_error_taxonomy",
        "method": (
            "Aggregates real failure instances already surfaced by Experiments "
            "1-14 and by building this validation suite, not synthetically "
            "generated examples. Each entry cites its source."
        ),
        "categories": CATEGORIES,
        "counts_by_category": counts,
        "total_cited_instances": total_instances,
        "categories_exercised": exercised,
        "categories_not_exercised": not_exercised,
        "note": (
            "perception_error, stale_observation, incorrect_parameters, and "
            "ros2_integration_failure were not exercised by any experiment in "
            "this suite - honestly reported as empty rather than padded with "
            "invented examples. perception_error/stale_observation would need a "
            "genuine perception pipeline (camera + detection model), out of "
            "MVP scope per the original spec; incorrect_parameters is caught "
            "upstream by Skill.build_action's validation before an Action can "
            "even be malformed (see Experiment 1's invalid_parameters category, "
            "which is a *rejection* of bad parameters, not a failure instance "
            "of bad parameters slipping through)."
        ),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp15_failure_taxonomy.json"
    out_path.write_text(json.dumps(result, indent=2))
