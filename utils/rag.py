import json
from pathlib import Path
from typing import Optional

# FIX BUG #6 — Cache knowledge base in memory after first load.
# No repeated disk reads on every user message.
_KB_CACHE: Optional[dict] = None


def load_knowledge_base() -> dict:
    """Load the AutoStream knowledge base from JSON (cached after first read)."""
    global _KB_CACHE
    if _KB_CACHE is None:
        kb_path = Path(__file__).parent.parent / "data" / "knowledge_base.json"
        with open(kb_path, "r", encoding="utf-8") as f:
            _KB_CACHE = json.load(f)
    return _KB_CACHE


def retrieve_context(query: str) -> str:
    """
    Keyword-based RAG retriever.
    Searches the knowledge base for relevant sections based on the user query.
    Returns a formatted string with matching context to inject into the LLM prompt.
    """
    kb = load_knowledge_base()
    query_lower = query.lower()
    context_parts: list[str] = []

    # Always include company overview
    company = kb["company"]
    context_parts.append(
        f"Company: {company['name']} — {company['tagline']}.\n"
        f"Description: {company['description']}"
    )

    # Pricing / plan keywords
    pricing_keywords = [
        "price", "cost", "plan", "basic", "pro", "monthly", "annual",
        "how much", "pricing", "subscription", "pay", "cheap", "afford",
        "upgrade", "features", "difference", "compare", "resolution",
        "4k", "720p", "unlimited", "captions", "sign up", "signup", "start", "begin", "get started", "try", "join",
    ]
    if any(kw in query_lower for kw in pricing_keywords):
        plans_text = "\n\nAvailable Plans:\n"
        for plan in kb["plans"]:
            features_list = "\n  - ".join(plan["features"])
            plans_text += (
                f"\n[{plan['name']} Plan]\n"
                f"  Price: ${plan['price_monthly']}/month "
                f"(or ${plan['price_annual']}/year)\n"
                f"  Best for: {plan['best_for']}\n"
                f"  Features:\n  - {features_list}\n"
            )
        context_parts.append(plans_text)

    # Policy keywords
    policy_keywords = [
        "refund", "support", "cancel", "storage", "trial", "free",
        "downgrade", "policy", "return", "money back", "24/7",
    ]
    if any(kw in query_lower for kw in policy_keywords):
        policies_text = "\n\nPolicies:\n"
        for policy in kb["policies"]:
            policies_text += f"- {policy['topic']}: {policy['detail']}\n"
        context_parts.append(policies_text)

    # FAQ keywords
    faq_keywords = [
        "can i", "what if", "how do", "do you", "team", "agency",
        "exceed", "limit", "try before",
    ]
    if any(kw in query_lower for kw in faq_keywords):
        faqs_text = "\n\nFrequently Asked Questions:\n"
        for faq in kb["faqs"]:
            faqs_text += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"
        context_parts.append(faqs_text)

    return "\n".join(context_parts)
