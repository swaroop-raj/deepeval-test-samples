"""Custom G-Eval criteria definitions for conversational AI agent evaluation.

Each metric uses LLM-as-a-judge with specific criteria tailored to
different aspects of conversational agent quality.
"""

from deepeval.metrics import ConversationalGEval, ConversationCompletenessMetric
from deepeval.metrics import ConversationalGEvalParam
from config import JUDGE_MODEL, THRESHOLD_STANDARD, THRESHOLD_STRICT, THRESHOLD_LENIENT


# =============================================================================
# SCENARIO 1: Policy Inquiry Metrics
# =============================================================================

def get_accuracy_metric(threshold: float = THRESHOLD_STANDARD) -> ConversationalGEval:
    """Measures factual accuracy of policy information provided by the agent.
    
    Checks:
    - Information aligns with retrieval_context (source documents)
    - No hallucinated policy details
    - Correct dates, numbers, and procedural steps
    - Proper citations/references to policy sections
    """
    return ConversationalGEval(
        name="Policy Accuracy",
        criteria=(
            "Evaluate whether the AI agent provides factually correct policy "
            "information that precisely aligns with the retrieval context provided. "
            "Score higher when the agent: (1) correctly states policy details like "
            "deadlines, requirements, and procedures, (2) references specific policy "
            "sections or documents, (3) does not fabricate or hallucinate information "
            "not present in the retrieval context. Score lower when the agent: "
            "(1) provides incorrect numbers, dates, or procedures, (2) makes up "
            "policy details not grounded in the context, (3) contradicts the source material."
        ),
        evaluation_params=[
            ConversationalGEvalParam.TURNS,
            ConversationalGEvalParam.RETRIEVAL_CONTEXT,
            ConversationalGEvalParam.EXPECTED_OUTCOME
        ],
        threshold=threshold,
        model=JUDGE_MODEL
    )


# =============================================================================
# SCENARIO 2: Complaint Resolution Metrics
# =============================================================================

def get_empathy_metric(threshold: float = THRESHOLD_LENIENT) -> ConversationalGEval:
    """Measures empathetic communication and de-escalation skills.
    
    Checks:
    - Acknowledges customer's emotional state
    - Uses empathetic language without being dismissive
    - De-escalates tension through validation
    - Maintains professional tone even when customer is aggressive
    - Avoids robotic or scripted-sounding responses
    """
    return ConversationalGEval(
        name="Empathy & De-escalation",
        criteria=(
            "Evaluate the AI agent's ability to demonstrate empathy and de-escalate "
            "customer frustration throughout the conversation. Score higher when the "
            "agent: (1) explicitly acknowledges the customer's feelings and frustration, "
            "(2) validates their concern as legitimate before jumping to solutions, "
            "(3) uses warm, human-sounding language rather than corporate jargon, "
            "(4) adapts tone based on the customer's emotional state across turns, "
            "(5) takes responsibility where appropriate. Score lower when the agent: "
            "(1) ignores emotional cues, (2) responds with generic scripted phrases, "
            "(3) becomes defensive or dismissive, (4) fails to match urgency to the situation."
        ),
        evaluation_params=[
            ConversationalGEvalParam.TURNS,
            ConversationalGEvalParam.EXPECTED_OUTCOME
        ],
        threshold=threshold,
        model=JUDGE_MODEL
    )


def get_resolution_metric(threshold: float = THRESHOLD_STANDARD) -> ConversationalGEval:
    """Measures quality and completeness of problem resolution offered.
    
    Checks:
    - Concrete resolution offered (not just acknowledgment)
    - Clear next steps and timeline communicated
    - Reference/case number provided for tracking
    - Proactive offers (goodwill gestures, escalation paths)
    - Resolution aligns with company policy
    """
    return ConversationalGEval(
        name="Resolution Quality",
        criteria=(
            "Evaluate whether the AI agent provides a concrete, actionable resolution "
            "to the customer's complaint. Score higher when the agent: (1) offers a "
            "specific solution (refund, credit, fix) rather than vague promises, "
            "(2) provides clear timelines for resolution, (3) gives a reference or "
            "case number for tracking, (4) proactively offers additional goodwill or "
            "escalation paths, (5) confirms the resolution with the customer. Score "
            "lower when the agent: (1) only acknowledges the problem without solving it, "
            "(2) provides unclear or missing timelines, (3) fails to give trackable "
            "reference information, (4) does not check if the customer is satisfied."
        ),
        evaluation_params=[
            ConversationalGEvalParam.TURNS,
            ConversationalGEvalParam.RETRIEVAL_CONTEXT,
            ConversationalGEvalParam.EXPECTED_OUTCOME
        ],
        threshold=threshold,
        model=JUDGE_MODEL
    )


# =============================================================================
# SCENARIO 3: Account Service Request Metrics
# =============================================================================

def get_process_adherence_metric(threshold: float = THRESHOLD_STRICT) -> ConversationalGEval:
    """Measures adherence to security and operational processes.
    
    Checks:
    - Identity verification performed BEFORE making changes
    - Correct verification factors requested (policy # + DOB)
    - Current information confirmed before updating
    - Change not executed until verification passes
    - Security protocol followed in correct order
    """
    return ConversationalGEval(
        name="Process Adherence",
        criteria=(
            "Evaluate whether the AI agent follows required security and operational "
            "processes in the correct order. Score higher when the agent: (1) requests "
            "identity verification BEFORE accessing or modifying account details, "
            "(2) asks for the correct verification factors (policy number + date of birth), "
            "(3) confirms current information on file before making changes, "
            "(4) does not reveal sensitive account details before verification is complete, "
            "(5) follows the change process step by step. Score lower when the agent: "
            "(1) skips identity verification entirely, (2) reveals account details before "
            "verifying identity, (3) makes changes without confirmation, (4) processes "
            "the request in wrong order."
        ),
        evaluation_params=[
            ConversationalGEvalParam.TURNS,
            ConversationalGEvalParam.RETRIEVAL_CONTEXT,
            ConversationalGEvalParam.EXPECTED_OUTCOME
        ],
        threshold=threshold,
        model=JUDGE_MODEL
    )


def get_task_completion_metric(threshold: float = THRESHOLD_STRICT) -> ConversationalGEval:
    """Measures whether the requested service task was fully completed.
    
    Checks:
    - Requested change was actually executed
    - Confirmation number/reference provided
    - All affected items updated (multi-policy scenarios)
    - Summary of changes communicated to customer
    - Side effects explained (premium changes, etc.)
    """
    return ConversationalGEval(
        name="Task Completion",
        criteria=(
            "Evaluate whether the AI agent successfully completed the requested service "
            "task and communicated the results clearly. Score higher when the agent: "
            "(1) executes the requested change completely, (2) provides a confirmation "
            "number or reference, (3) updates all affected records (e.g., multiple policies), "
            "(4) summarizes exactly what was changed, (5) proactively informs about side "
            "effects or downstream impacts (premium changes, coverage adjustments). Score "
            "lower when the agent: (1) claims to make changes but doesn't confirm execution, "
            "(2) forgets to provide tracking/confirmation info, (3) misses affected items, "
            "(4) doesn't explain implications of the change."
        ),
        evaluation_params=[
            ConversationalGEvalParam.TURNS,
            ConversationalGEvalParam.RETRIEVAL_CONTEXT,
            ConversationalGEvalParam.EXPECTED_OUTCOME
        ],
        threshold=threshold,
        model=JUDGE_MODEL
    )


# =============================================================================
# SHARED METRIC: Conversation Completeness (Built-in)
# =============================================================================

def get_completeness_metric(threshold: float = THRESHOLD_STANDARD) -> ConversationCompletenessMetric:
    """Built-in DeepEval metric for overall conversation completeness.
    
    Measures whether all user needs expressed throughout the conversation
    were addressed by the end of the interaction.
    """
    return ConversationCompletenessMetric(
        threshold=threshold,
        model=JUDGE_MODEL
    )
