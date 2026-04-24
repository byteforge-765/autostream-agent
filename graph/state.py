from typing import TypedDict, Optional, List
from enum import Enum


class IntentType(str, Enum):
    GREETING = "greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    HIGH_INTENT = "high_intent"
    UNKNOWN = "unknown"


class LeadInfo(TypedDict, total=False):
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]


class AgentState(TypedDict):
    """
    Central state object passed through the LangGraph nodes.
    All conversation data lives here — this is how memory is maintained
    across multiple turns without an external database.
    """
    # Full conversation history (list of role+content dicts)
    messages: List[dict]

    # Current classified intent
    intent: IntentType

    # Collected lead details (filled progressively)
    lead_info: LeadInfo

    # Whether the lead has been successfully captured
    lead_captured: bool

    # Which field we're currently asking the user for
    collecting_field: Optional[str]

    # RAG context retrieved for the current query
    rag_context: str

    # The final response to send back to the user
    response: str
