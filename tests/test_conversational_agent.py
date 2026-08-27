"""Main evaluation test suite for conversational AI agent.

Runs 3 evaluation scenarios using DeepEval's LLM-as-a-judge approach:
1. Policy Inquiry - accuracy and completeness
2. Complaint Resolution - empathy, resolution quality, completeness
3. Account Service Request - process adherence, task completion

Usage:
    pytest tests/test_conversational_agent.py -v
    deepeval test run tests/test_conversational_agent.py
    python tests/test_conversational_agent.py  (standalone)
"""

import json
from pathlib import Path
from deepeval import assert_test
from deepeval.test_case import ConversationalTestCase, Turn, LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval import evaluate

from metrics.custom_criteria import (
    get_accuracy_metric,
    get_empathy_metric,
    get_resolution_metric,
    get_process_adherence_metric,
    get_task_completion_metric,
    get_completeness_metric
)
from config import JUDGE_MODEL, THRESHOLD_STANDARD


# =============================================================================
# DATA LOADING
# =============================================================================

def load_scenario(scenario_id: str) -> ConversationalTestCase:
    """Load a specific scenario by ID and return as ConversationalTestCase."""
    data_path = Path(__file__).parent.parent / "sample_data" / "scenarios.json"
    with open(data_path, "r") as f:
        scenarios = json.load(f)["scenarios"]
    
    scenario = next(s for s in scenarios if s["id"] == scenario_id)
    
    turns = []
    for turn_data in scenario["turns"]:
        turn_kwargs = {"role": turn_data["role"], "content": turn_data["content"]}
        if "retrieval_context" in turn_data:
            turn_kwargs["retrieval_context"] = turn_data["retrieval_context"]
        turns.append(Turn(**turn_kwargs))
    
    return ConversationalTestCase(
        scenario=scenario["scenario"],
        expected_outcome=scenario["expected_outcome"],
        chatbot_role=scenario["chatbot_role"],
        turns=turns
    )


# =============================================================================
# SCENARIO 1: POLICY INQUIRY
# =============================================================================

def test_policy_inquiry():
    """Evaluate: Does the agent provide accurate, complete policy information?
    
    Metrics:
    - Policy Accuracy (ConversationalGEval): Factual correctness vs. source docs
    - Conversation Completeness: All user questions addressed
    
    Threshold: 0.7 (standard)
    """
    test_case = load_scenario("policy-inquiry-001")
    
    metrics = [
        get_accuracy_metric(),
        get_completeness_metric()
    ]
    
    # Run evaluation
    for metric in metrics:
        assert_test(test_case, [metric])


# =============================================================================
# SCENARIO 2: COMPLAINT RESOLUTION
# =============================================================================

def test_complaint_resolution():
    """Evaluate: Does the agent handle complaints with empathy and resolution?
    
    Metrics:
    - Empathy & De-escalation (ConversationalGEval): Emotional intelligence
    - Resolution Quality (ConversationalGEval): Concrete problem-solving
    - Conversation Completeness: Full issue resolution
    
    Threshold: 0.6-0.75 (lenient for empathy, standard for resolution)
    """
    test_case = load_scenario("complaint-resolution-001")
    
    metrics = [
        get_empathy_metric(),
        get_resolution_metric(),
        get_completeness_metric(threshold=0.75)
    ]
    
    for metric in metrics:
        assert_test(test_case, [metric])


# =============================================================================
# SCENARIO 3: ACCOUNT SERVICE REQUEST
# =============================================================================

def test_account_service_request():
    """Evaluate: Does the agent process requests with proper verification?
    
    Metrics:
    - Process Adherence (ConversationalGEval): Security protocol followed
    - Task Completion (ConversationalGEval): Change fully executed & confirmed
    - Conversation Completeness: All steps completed
    
    Threshold: 0.8 (strict - security-sensitive scenario)
    """
    test_case = load_scenario("account-service-001")
    
    metrics = [
        get_process_adherence_metric(),
        get_task_completion_metric(),
        get_completeness_metric(threshold=0.8)
    ]
    
    for metric in metrics:
        assert_test(test_case, [metric])


# =============================================================================
# BONUS: PER-TURN QUALITY CHECK (GEval on individual responses)
# =============================================================================

def test_individual_turn_quality():
    """Evaluate individual agent responses for quality.
    
    Uses GEval (single-turn) to score each assistant response independently.
    This catches issues like:
    - Overly verbose responses
    - Poor formatting/structure
    - Unclear or ambiguous language
    
    Threshold: 0.6 (lenient - some responses may be brief by design)
    """
    data_path = Path(__file__).parent.parent / "sample_data" / "scenarios.json"
    with open(data_path, "r") as f:
        scenarios = json.load(f)["scenarios"]
    
    # Quality metric for individual turns
    turn_quality = GEval(
        name="Response Clarity & Helpfulness",
        criteria=(
            "Evaluate whether this individual agent response is clear, well-structured, "
            "and helpful. Score higher for: (1) clear and scannable formatting, "
            "(2) actionable information, (3) appropriate length (not too verbose, not too terse), "
            "(4) professional yet warm tone. Score lower for: (1) wall of text with no structure, "
            "(2) vague or non-committal language, (3) overly robotic tone."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.6,
        model=JUDGE_MODEL
    )
    
    # Extract all assistant turns across all scenarios
    test_cases = []
    for scenario in scenarios:
        for turn in scenario["turns"]:
            if turn["role"] == "assistant":
                test_cases.append(
                    LLMTestCase(
                        input=f"[Scenario: {scenario['name']}]",
                        actual_output=turn["content"]
                    )
                )
    
    # Evaluate all turns
    results = evaluate(test_cases=test_cases, metrics=[turn_quality])
    
    # Assert all pass
    failed = [r for r in results.test_results if not r.success]
    assert len(failed) == 0, (
        f"{len(failed)}/{len(test_cases)} individual responses failed quality check. "
        f"Review the DeepEval output above for details."
    )


# =============================================================================
# STANDALONE EXECUTION (without pytest)
# =============================================================================

if __name__ == "__main__":
    """Run all evaluations and print a summary report."""
    
    print("=" * 70)
    print("DeepEval: Conversational AI Agent Evaluation")
    print(f"Judge Model: {JUDGE_MODEL}")
    print("=" * 70)
    
    scenarios = {
        "Policy Inquiry": {
            "case": load_scenario("policy-inquiry-001"),
            "metrics": [get_accuracy_metric(), get_completeness_metric()]
        },
        "Complaint Resolution": {
            "case": load_scenario("complaint-resolution-001"),
            "metrics": [get_empathy_metric(), get_resolution_metric(), get_completeness_metric(threshold=0.75)]
        },
        "Account Service Request": {
            "case": load_scenario("account-service-001"),
            "metrics": [get_process_adherence_metric(), get_task_completion_metric(), get_completeness_metric(threshold=0.8)]
        }
    }
    
    all_results = []
    
    for name, config in scenarios.items():
        print(f"\n{'─' * 70}")
        print(f"Scenario: {name}")
        print(f"{'─' * 70}")
        
        results = evaluate(
            test_cases=[config["case"]],
            metrics=config["metrics"]
        )
        
        scenario_pass = True
        for test_result in results.test_results:
            for metric_result in test_result.metrics_data:
                status = "✅ PASS" if metric_result.success else "❌ FAIL"
                print(f"  {metric_result.name}: {metric_result.score:.2f} "
                      f"(threshold: {metric_result.threshold}) {status}")
                if metric_result.reason:
                    print(f"    Reason: {metric_result.reason[:120]}...")
                if not metric_result.success:
                    scenario_pass = False
        
        all_results.append({"name": name, "passed": scenario_pass})
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    passed = sum(1 for r in all_results if r["passed"])
    total = len(all_results)
    for r in all_results:
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} {r['name']}")
    print(f"\nOverall: {passed}/{total} scenarios passed")
    print(f"{'=' * 70}")
