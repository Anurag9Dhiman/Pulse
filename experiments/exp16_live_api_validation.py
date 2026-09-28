"""Experiment 16: Live API Validation.

Do mocked planner clients accurately represent behavior observed when PAR
interacts with real LLM provider APIs?

This session already generated direct, real evidence for two of the three
testing levels the question asks about - documented here formally rather
than re-spending live API quota to re-manufacture discrepancies that
already happened organically. "Repeated scheduled live validation" (level
3) is a genuinely different thing (requires running over days/weeks, not a
single session) - its design is specified here, not faked as a result.
"""
from __future__ import annotations

import json
from pathlib import Path

TESTING_LEVELS = {
    "1_unit_tests_fake_clients": {
        "description": "tests/test_llm_planner.py, tests/test_gemini_planner.py - hand-written fake clients reproducing each provider's documented response shape.",
        "coverage": "Tool-schema construction, response parsing, task_complete handling, conversation-history feedback loop.",
        "limitation": "Can only catch discrepancies the fake client was built to reproduce - by definition cannot catch a real API behaving in a way the fake client's author didn't anticipate.",
    },
    "2_integration_tests_live_api": {
        "description": "Direct live calls to the real Gemini API, across this session and the paper-writing session that preceded it.",
        "coverage": "Real function calling, real multi-step task execution, real re-planning after a Safety Kernel denial (paper Section VI-B), Experiment 5's 3 live trials.",
        "discrepancies_found": [
            {
                "what": "google-genai's send_message() rejects plain dicts at runtime with a strict type check, despite its own type hints advertising PartDict as an accepted alternative to Part objects.",
                "found_by": "Live call during paper-writing session - GeminiPlanner initially built message parts as plain dicts (matching the documented type union), which worked for the config dict but failed at runtime for message parts specifically.",
                "fake_client_status": "Did NOT catch this - the fake client's type checking only enforces what the test author encoded, not what the real SDK enforces internally.",
                "fixed": True,
            },
            {
                "what": "gemini-2.5-flash returned 404 NOT_FOUND: 'This model is no longer available to new users', discovered only at call time.",
                "found_by": "Live call during paper-writing session.",
                "fake_client_status": "Cannot be caught by any fake client in principle - model availability is a vendor-side fact with no local representation.",
                "fixed": True,
            },
            {
                "what": "429 RESOURCE_EXHAUSTED from the free tier's 15 requests/minute limit on gemini-3.1-flash-lite, mid-experiment.",
                "found_by": "Experiment 5 (this session), running 3 live trials.",
                "fake_client_status": "Cannot be caught by any fake client in principle - rate limits are a live-account-state fact.",
                "fixed": True,
            },
        ],
    },
    "3_repeated_scheduled_live_validation": {
        "description": "Running the same live-API smoke test on a recurring schedule (e.g. daily) to catch model deprecations, SDK behavior changes, and quota changes as they happen, rather than only at development time.",
        "status": "NOT implemented in this session - this is a genuinely different kind of thing (an operational practice over time, not a one-shot experiment) and would need a scheduling mechanism outside a single validation run.",
        "design_for_future_implementation": (
            "A minimal version: run examples/gemini_loop.py (or an equivalent smoke "
            "test) via a scheduled job (the `schedule` skill available in this "
            "environment, or an external cron), asserting only that a basic "
            "pick+place task completes and that par doctor reports the LLM "
            "planner ready. A failure - model 404, new schema rejection, quota "
            "exhaustion pattern change - would be the actionable signal, not the "
            "full experiment suite re-run daily."
        ),
    },
}


def run() -> dict:
    n_discrepancies = len(TESTING_LEVELS["2_integration_tests_live_api"]["discrepancies_found"])
    n_uncatchable_in_principle = sum(
        1
        for d in TESTING_LEVELS["2_integration_tests_live_api"]["discrepancies_found"]
        if "in principle" in d["fake_client_status"]
    )
    return {
        "experiment": "16_live_api_validation",
        "testing_levels": TESTING_LEVELS,
        "summary": {
            "n_real_discrepancies_found": n_discrepancies,
            "n_uncatchable_by_any_fake_client_in_principle": n_uncatchable_in_principle,
            "n_all_fixed": sum(
                1 for d in TESTING_LEVELS["2_integration_tests_live_api"]["discrepancies_found"] if d["fixed"]
            ),
        },
        "conclusion": (
            "Mocked/fake-client testing did not, and structurally could not, "
            "catch 2 of the 3 real discrepancies found (model availability and "
            "rate limits are facts about the live account/vendor state with no "
            "local representation to mock correctly). Even the one discrepancy "
            "that was in principle representable (the strict dict-rejection) was "
            "missed because the fake client was built to match documentation "
            "that turned out not to match the real implementation. This is the "
            "central finding, not a one-off bug report: any embodied-agent "
            "system treating a third-party model API as an interchangeable "
            "backend needs periodic live validation as a standing practice, not "
            "a one-time integration test."
        ),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
    out_path = Path(__file__).parent / "results" / "exp16_live_api_validation.json"
    out_path.write_text(json.dumps(result, indent=2))
