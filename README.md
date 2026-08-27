<!-- agent:brain -->
<!-- Commit convention: Conventional Commits with agent scope -->
<!-- Format: type(agent:brain): description -->
<!-- Tags: semantic versioning (v0.1.0, v1.0.0) -->
<!-- Branch: main (PRs for all changes) -->

# DeepEval: LLM-as-a-Judge for Conversational AI Agents

> Enterprise-grade evaluation framework using [DeepEval](https://deepeval.com) to validate conversational AI agent outputs with LLM-as-a-judge scoring.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Evaluation Scenarios](#evaluation-scenarios)
4. [Setup & Installation](#setup--installation)
5. [Walkthrough: How It Works](#walkthrough-how-it-works)
6. [Input/Output JSON Structure](#inputoutput-json-structure)
7. [Scoring Mechanism](#scoring-mechanism)
8. [Running Evaluations](#running-evaluations)
9. [Interpreting Results](#interpreting-results)
10. [Extending with New Scenarios](#extending-with-new-scenarios)

---

## Overview

This repository demonstrates how to evaluate a **conversational AI agent** using DeepEval's LLM-as-a-judge approach. Instead of human reviewers manually scoring agent conversations, we use a powerful LLM (GPT-4o, Claude, etc.) as an automated judge that scores agent outputs against defined criteria.

### Why LLM-as-a-Judge?

| Traditional QA | LLM-as-a-Judge |
|---|---|
| Manual review of conversations | Automated, scalable evaluation |
| Subjective, inconsistent scoring | Reproducible criteria-based scoring |
| Expensive at scale | Cost-effective batch evaluation |
| Slow feedback loops | CI/CD-integrated, instant feedback |

### What This Repo Covers

- **3 real-world evaluation scenarios** for a customer service AI agent
- **Multi-turn conversation evaluation** using `ConversationalTestCase`
- **Multiple scoring dimensions**: completeness, empathy, accuracy, task completion
- **Custom G-Eval criteria** tailored to conversational AI
- **Sample data** with realistic agent interactions
- **Threshold-based pass/fail** with detailed score reasoning

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DeepEval Pipeline                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │ Sample Data   │───▶│ Test Cases    │───▶│  Metrics  │  │
│  │ (JSON)        │    │ (Multi-Turn)  │    │ (LLM Judge│  │
│  └──────────────┘    └──────────────┘    └─────┬─────┘  │
│                                                  │       │
│                                          ┌───────▼─────┐ │
│                                          │   Scores    │ │
│                                          │ + Reasoning │ │
│                                          └─────────────┘ │
└─────────────────────────────────────────────────────────┘

Metrics Used:
├── ConversationalGEval (custom criteria per scenario)
├── ConversationCompletenessMetric (end-to-end resolution)
└── GEval (per-turn quality checks)
```

---

## Evaluation Scenarios

### Scenario 1: Policy Inquiry
**Goal**: Validate the agent provides accurate, complete policy information.

| Parameter | Value |
|---|---|
| Turns | 4 (user asks, agent responds, follow-up, clarification) |
| Criteria | Accuracy, completeness, no hallucination |
| Threshold | 0.7 |
| Judge Model | GPT-4o |

**What we check**: Does the agent cite correct policy details? Does it avoid making up information? Does it handle follow-up questions consistently?

### Scenario 2: Complaint Resolution
**Goal**: Validate the agent handles complaints with empathy and resolution.

| Parameter | Value |
|---|---|
| Turns | 6 (complaint, acknowledgment, investigation, resolution, confirmation) |
| Criteria | Empathy, de-escalation, resolution offered, follow-through |
| Threshold | 0.75 |
| Judge Model | GPT-4o |

**What we check**: Does the agent acknowledge frustration? Does it offer concrete resolution? Does it maintain professional tone throughout?

### Scenario 3: Account Service Request
**Goal**: Validate the agent correctly processes service requests with identity verification.

| Parameter | Value |
|---|---|
| Turns | 5 (request, verification, confirmation, execution, summary) |
| Criteria | Process adherence, identity verification, task completion, accuracy |
| Threshold | 0.8 |
| Judge Model | GPT-4o |

**What we check**: Does the agent verify identity before making changes? Does it confirm details before executing? Does it summarize what was done?

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- OpenAI API key (for the LLM judge)

### Install

```bash
# Clone the repo
git clone https://github.com/swaroop-raj/deepeval-test-samples.git
cd deepeval-test-samples

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="your-key-here"
```

---

## Walkthrough: How It Works

### Step 1: Define Your Conversation Data

Conversations are stored in `sample_data/scenarios.json`. Each scenario has:
- A **scenario description** (what's being tested)
- An **expected outcome** (what success looks like)
- A **chatbot role** (the agent's persona)
- **Turns** (the actual conversation exchanges)

```python
# Each turn in the conversation
{
    "role": "user" | "assistant",
    "content": "The actual message text",
    "retrieval_context": ["Optional: grounding docs the agent used"]
}
```

### Step 2: Build ConversationalTestCase Objects

DeepEval uses `ConversationalTestCase` to represent multi-turn interactions:

```python
from deepeval.test_case import ConversationalTestCase, Turn

test_case = ConversationalTestCase(
    scenario="User inquires about claim filing deadline",
    expected_outcome="Agent provides accurate deadline (30 days) with supporting policy reference",
    chatbot_role="Insurance customer service agent",
    turns=[
        Turn(role="user", content="What's the deadline to file a claim?"),
        Turn(
            role="assistant",
            content="You have 30 days from the date of incident to file a claim...",
            retrieval_context=["Policy Section 4.2: Claims must be filed within 30 calendar days..."]
        ),
        # ... more turns
    ]
)
```

### Step 3: Configure Metrics (LLM-as-a-Judge)

We use three types of metrics:

#### ConversationalGEval (Custom Criteria)
```python
from deepeval.metrics import ConversationalGEval

accuracy_metric = ConversationalGEval(
    name="Policy Accuracy",
    criteria="Determine whether the AI agent provides factually correct policy information that aligns with the retrieval context provided. The agent should not hallucinate or fabricate policy details.",
    evaluation_params=[
        ConversationalGEvalParam.TURNS,
        ConversationalGEvalParam.RETRIEVAL_CONTEXT,
        ConversationalGEvalParam.EXPECTED_OUTCOME
    ],
    threshold=0.7,
    model="gpt-4o"
)
```

#### ConversationCompletenessMetric (Built-in)
```python
from deepeval.metrics import ConversationCompletenessMetric

completeness = ConversationCompletenessMetric(
    threshold=0.7,
    model="gpt-4o"
)
```

#### GEval for Per-Turn Quality
```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

turn_quality = GEval(
    name="Response Quality",
    criteria="Evaluate whether the response is helpful, clear, and professionally worded.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6,
    model="gpt-4o"
)
```

### Step 4: Run Evaluation

```python
from deepeval import evaluate

results = evaluate(
    test_cases=[test_case],
    metrics=[accuracy_metric, completeness]
)
```

### Step 5: Interpret Scores

Each metric returns:
- **score** (0.0 to 1.0): The LLM judge's rating
- **reason**: Natural language explanation of why that score was given
- **success** (bool): Whether score >= threshold

```json
{
    "metric": "Policy Accuracy",
    "score": 0.85,
    "threshold": 0.7,
    "success": true,
    "reason": "The agent correctly cited the 30-day filing deadline and referenced Section 4.2. However, it did not mention the exception for force majeure events."
}
```

---

## Input/Output JSON Structure

### Input Format (`sample_data/scenarios.json`)

```json
{
    "scenarios": [
        {
            "id": "policy-inquiry-001",
            "name": "Claim Filing Deadline Inquiry",
            "scenario": "User asks about insurance claim filing deadlines and follow-up procedures",
            "expected_outcome": "Agent provides accurate 30-day deadline, references policy section, and explains the follow-up process",
            "chatbot_role": "Insurance customer service AI agent",
            "turns": [
                {
                    "role": "user",
                    "content": "Hi, I had a car accident last week. What's the deadline to file a claim?"
                },
                {
                    "role": "assistant",
                    "content": "I'm sorry to hear about your accident...",
                    "retrieval_context": ["Policy Section 4.2: ..."]
                }
            ],
            "eval_config": {
                "metrics": ["accuracy", "completeness"],
                "threshold": 0.7,
                "model": "gpt-4o"
            }
        }
    ]
}
```

### Output Format (Evaluation Results)

```json
{
    "evaluation_run": {
        "timestamp": "2026-08-27T14:30:00Z",
        "model_judge": "gpt-4o",
        "total_scenarios": 3,
        "passed": 2,
        "failed": 1
    },
    "results": [
        {
            "scenario_id": "policy-inquiry-001",
            "metrics": [
                {
                    "name": "Policy Accuracy",
                    "score": 0.85,
                    "threshold": 0.7,
                    "success": true,
                    "reason": "Agent correctly cited..."
                },
                {
                    "name": "Conversation Completeness",
                    "score": 0.72,
                    "threshold": 0.7,
                    "success": true,
                    "reason": "All user questions addressed..."
                }
            ],
            "overall_pass": true
        }
    ]
}
```

---

## Scoring Mechanism

### How LLM-as-a-Judge Scores

| Metric Type | Scoring Method | Range | What It Measures |
|---|---|---|---|
| ConversationalGEval | Chain-of-thought + criteria rubric | 0.0 - 1.0 | Custom criteria across full conversation |
| ConversationCompleteness | Turn-by-turn need satisfaction | 0.0 - 1.0 | Whether all user needs were met |
| GEval (per-turn) | Single-output criteria evaluation | 0.0 - 1.0 | Individual response quality |

### Scoring Parameters Checked

| Parameter | Description | Used In |
|---|---|---|
| `turns` | Full conversation history | All conversational metrics |
| `retrieval_context` | Source documents agent referenced | Accuracy, hallucination checks |
| `expected_outcome` | What a good resolution looks like | Completeness, task completion |
| `scenario` | Context of the interaction | All metrics (provides framing) |
| `chatbot_role` | Agent persona/constraints | Role adherence checks |

### Threshold Strategy

```python
# Conservative: production-critical scenarios
THRESHOLD_STRICT = 0.8   # Account changes, financial ops

# Standard: general quality bar
THRESHOLD_STANDARD = 0.7  # Policy info, general inquiries

# Lenient: subjective/creative responses
THRESHOLD_LENIENT = 0.6   # Empathy scoring, tone evaluation
```

A scenario **passes** only if ALL configured metrics meet their respective thresholds.

---

## Running Evaluations

### Run All Scenarios
```bash
python -m pytest tests/ -v
```

### Run Specific Scenario
```bash
python -m pytest tests/test_conversational_agent.py::test_policy_inquiry -v
```

### Run with DeepEval CLI
```bash
deepeval test run tests/test_conversational_agent.py
```

### Run as Standalone Script
```bash
python tests/test_conversational_agent.py
```

---

## Interpreting Results

### Console Output
```
✅ Scenario: Policy Inquiry
   Policy Accuracy: 0.85 (threshold: 0.7) PASS
   Conversation Completeness: 0.72 (threshold: 0.7) PASS

✅ Scenario: Complaint Resolution  
   Empathy & De-escalation: 0.90 (threshold: 0.75) PASS
   Resolution Quality: 0.78 (threshold: 0.75) PASS
   Conversation Completeness: 0.80 (threshold: 0.75) PASS

❌ Scenario: Account Service Request
   Process Adherence: 0.65 (threshold: 0.8) FAIL
   Reason: "Agent did not verify customer identity before proceeding with the address change."
   Task Completion: 0.88 (threshold: 0.8) PASS

Overall: 2/3 scenarios passed
```

### What a Failure Tells You

When a metric fails, the **reason** field is your debugging tool. It tells you exactly what the LLM judge found lacking, which maps directly to what you need to fix in your agent's prompts or retrieval pipeline.

---

## Extending with New Scenarios

### Adding a New Scenario

1. **Add conversation data** to `sample_data/scenarios.json`
2. **Define criteria** in `metrics/custom_criteria.py`
3. **Add test function** in `tests/test_conversational_agent.py`

```python
# In metrics/custom_criteria.py
NEW_SCENARIO_CRITERIA = {
    "name": "Technical Accuracy",
    "criteria": "Evaluate whether the agent provides technically correct information...",
    "threshold": 0.75
}
```

### Swapping the Judge Model

```python
# In config.py
JUDGE_MODEL = "gpt-4o"          # Default
JUDGE_MODEL = "claude-3-opus"    # Alternative
JUDGE_MODEL = "gpt-4o-mini"      # Cost-effective for dev
```

---

## Project Structure

```
deepeval-test-samples/
├── README.md                          # This walkthrough
├── requirements.txt                   # Python dependencies
├── config.py                          # Global evaluation config
├── sample_data/
│   └── scenarios.json                 # All test scenarios (input/output)
├── metrics/
│   ├── __init__.py
│   └── custom_criteria.py            # Custom G-Eval criteria definitions
└── tests/
    ├── __init__.py
    ├── conftest.py                    # Pytest fixtures & data loading
    └── test_conversational_agent.py   # Main evaluation test suite
```

---

## References

- [DeepEval Documentation](https://deepeval.com/docs)
- [ConversationalTestCase Reference](https://deepeval.com/docs/evaluation-multiturn-test-cases)
- [ConversationalGEval Metric](https://deepeval.com/docs/metrics-conversational-g-eval)
- [LLM-as-a-Judge Techniques (2026)](https://deepeval.com/blog/llm-as-a-judge)
- [G-Eval Paper](https://arxiv.org/abs/2303.16634)

---

## License

MIT
