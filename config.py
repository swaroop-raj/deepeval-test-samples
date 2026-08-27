"""Global configuration for DeepEval conversational agent evaluation."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Judge Model Configuration ---
JUDGE_MODEL = os.getenv("DEEPEVAL_JUDGE_MODEL", "gpt-4o")

# --- Threshold Levels ---
THRESHOLD_STRICT = 0.8      # For high-stakes scenarios (account changes, financial ops)
THRESHOLD_STANDARD = 0.7    # For general quality (policy info, inquiries)
THRESHOLD_LENIENT = 0.6     # For subjective criteria (empathy, tone)

# --- Scenario Configuration ---
SCENARIO_CONFIG = {
    "policy_inquiry": {
        "metrics": ["accuracy", "completeness"],
        "threshold": THRESHOLD_STANDARD,
        "description": "Validates agent provides accurate, complete policy information"
    },
    "complaint_resolution": {
        "metrics": ["empathy", "resolution", "completeness"],
        "threshold": THRESHOLD_STANDARD,
        "description": "Validates agent handles complaints with empathy and resolution"
    },
    "account_service": {
        "metrics": ["process_adherence", "task_completion", "completeness"],
        "threshold": THRESHOLD_STRICT,
        "description": "Validates agent processes service requests with proper verification"
    }
}

# --- Output Configuration ---
OUTPUT_DIR = os.getenv("DEEPEVAL_OUTPUT_DIR", "./results")
VERBOSE = os.getenv("DEEPEVAL_VERBOSE", "true").lower() == "true"
