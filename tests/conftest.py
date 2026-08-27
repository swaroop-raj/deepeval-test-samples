"""Pytest fixtures for loading scenario data and configuring evaluations."""

import json
import os
import pytest
from pathlib import Path
from deepeval.test_case import ConversationalTestCase, Turn


DATA_PATH = Path(__file__).parent.parent / "sample_data" / "scenarios.json"


def load_scenarios() -> dict:
    """Load all scenarios from the JSON data file."""
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def build_test_case(scenario_data: dict) -> ConversationalTestCase:
    """Convert a scenario JSON object into a DeepEval ConversationalTestCase."""
    turns = []
    for turn_data in scenario_data["turns"]:
        turn_kwargs = {
            "role": turn_data["role"],
            "content": turn_data["content"]
        }
        if "retrieval_context" in turn_data:
            turn_kwargs["retrieval_context"] = turn_data["retrieval_context"]
        turns.append(Turn(**turn_kwargs))

    return ConversationalTestCase(
        scenario=scenario_data["scenario"],
        expected_outcome=scenario_data["expected_outcome"],
        chatbot_role=scenario_data["chatbot_role"],
        turns=turns
    )


@pytest.fixture(scope="session")
def all_scenarios():
    """Fixture: all scenario data as raw dicts."""
    return load_scenarios()["scenarios"]


@pytest.fixture(scope="session")
def policy_inquiry_case(all_scenarios):
    """Fixture: ConversationalTestCase for policy inquiry scenario."""
    scenario = next(s for s in all_scenarios if s["id"] == "policy-inquiry-001")
    return build_test_case(scenario)


@pytest.fixture(scope="session")
def complaint_resolution_case(all_scenarios):
    """Fixture: ConversationalTestCase for complaint resolution scenario."""
    scenario = next(s for s in all_scenarios if s["id"] == "complaint-resolution-001")
    return build_test_case(scenario)


@pytest.fixture(scope="session")
def account_service_case(all_scenarios):
    """Fixture: ConversationalTestCase for account service request scenario."""
    scenario = next(s for s in all_scenarios if s["id"] == "account-service-001")
    return build_test_case(scenario)
