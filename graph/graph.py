from langgraph.graph import StateGraph, END
from graph.state import AgentState, IntentType
from graph.nodes import classify_intent, retrieve_knowledge, generate_response, qualify_lead


def route_after_intent(state: AgentState) -> str:
    """
    Router: decides the next node based on classified intent.

    - greeting / product_inquiry / unknown → retrieve RAG context → generate response
    - high_intent OR actively collecting lead fields → qualify_lead
    - lead already captured → back to normal response (don't re-trigger qualification)
    """
    # Lead already captured — treat subsequent messages as normal inquiries
    if state.get("lead_captured"):
        return "retrieve_knowledge"

    # Mid-collection: user is answering our name/email/platform questions
    if state.get("collecting_field"):
        return "qualify_lead"

    intent = state.get("intent", IntentType.UNKNOWN)

    if intent == IntentType.HIGH_INTENT:
        return "qualify_lead"

    return "retrieve_knowledge"


def build_graph() -> StateGraph:
    """
    Constructs and compiles the full LangGraph agent graph.

    Flow:
        classify_intent
            ├── high_intent / collecting_field → qualify_lead → END
            └── greeting / inquiry / unknown   → retrieve_knowledge → generate_response → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("generate_response", generate_response)
    graph.add_node("qualify_lead", qualify_lead)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Conditional routing after intent classification
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "qualify_lead": "qualify_lead",
            "retrieve_knowledge": "retrieve_knowledge",
        }
    )

    # After retrieval → always generate response
    graph.add_edge("retrieve_knowledge", "generate_response")

    # Both terminal nodes end the graph
    graph.add_edge("generate_response", END)
    graph.add_edge("qualify_lead", END)

    return graph.compile()
