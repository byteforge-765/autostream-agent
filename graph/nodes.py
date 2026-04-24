import os
import re
from google import genai
from google.genai.errors import APIError, ServerError
from graph.state import AgentState, IntentType
from utils.rag import retrieve_context
from tools.lead_capture import mock_lead_capture

_client = None

FALLBACK_RESPONSE = (
    "I'm sorry, I didn't quite catch that. Could you rephrase? "
    "I'm here to help with AutoStream's plans, pricing, and getting you started."
)

# gemini-2.0-flash is the correct model ID for free-tier access via google-genai SDK
_MODEL_NAME = "gemini-3-flash-preview"


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _sanitize_input(text: str) -> str:
    """Strip injection chars and cap length at 500."""
    sanitized = re.sub(r"[`<>]", "", text)
    return sanitized[:500].strip()


def _call_gemini(prompt: str) -> str:
    """
    Call Gemini safely using the google-genai SDK.
    Uses APIError.code (int) from google.genai.errors for precise handling.
    """
    try:
        response = _get_client().models.generate_content(
            model=_MODEL_NAME,
            contents=prompt,
        )
        # Extract text from parts — works with thinking models (thought_signature bytes)
        text = ""
        try:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        except (IndexError, AttributeError):
            pass
        return text.strip() if text.strip() else FALLBACK_RESPONSE
    except APIError as e:
        if e.code == 429:
            raise
        if e.code in (400, 404):
            raise
        return FALLBACK_RESPONSE
    except (ValueError, AttributeError):
        return FALLBACK_RESPONSE


def _get_last_message(state: AgentState) -> str:
    """Safe messages[-1] accessor — returns '' on empty list."""
    messages = state.get("messages") or []
    return messages[-1].get("content", "") if messages else ""


# ─────────────────────────────────────────────
# NODE 1: Intent Classifier
# ─────────────────────────────────────────────
def classify_intent(state: AgentState) -> AgentState:
    raw_message = _get_last_message(state)
    if not raw_message:
        return {**state, "intent": IntentType.UNKNOWN}

    safe_message = _sanitize_input(raw_message)

    prompt = (
        "You are an intent classifier for AutoStream, a SaaS video editing platform.\n\n"
        "Classify the user message into EXACTLY one of these intents:\n"
        "- greeting        : casual hello, hi, how are you, general small talk\n"
        "- product_inquiry : asking about features, pricing, plans, refunds, support\n"
        "- high_intent     : clearly wants to sign up, buy, start a trial, try the product\n\n"
        f"User message: {safe_message}\n\n"
        "Reply with ONLY the intent label (one word, no punctuation). No explanation."
    )

    raw = _call_gemini(prompt).lower().strip()

    if "greeting" in raw:
        intent = IntentType.GREETING
    elif "high_intent" in raw or "high intent" in raw:
        intent = IntentType.HIGH_INTENT
    elif "product" in raw or "inquiry" in raw:
        intent = IntentType.PRODUCT_INQUIRY
    else:
        intent = IntentType.UNKNOWN

    return {**state, "intent": intent}


# ─────────────────────────────────────────────
# NODE 2: RAG Retriever
# ─────────────────────────────────────────────
def retrieve_knowledge(state: AgentState) -> AgentState:
    last_message = _get_last_message(state)
    context = retrieve_context(last_message) if last_message else ""
    return {**state, "rag_context": context}


# ─────────────────────────────────────────────
# NODE 3: Response Generator
# ─────────────────────────────────────────────
def generate_response(state: AgentState) -> AgentState:
    raw_message   = _get_last_message(state)
    safe_message  = _sanitize_input(raw_message)
    intent        = state.get("intent", IntentType.UNKNOWN)
    context       = state.get("rag_context", "")
    lead_captured = state.get("lead_captured", False)

    # FIX BUG-3: use msg.get("role", "user") to avoid KeyError on malformed messages
    history_text = ""
    messages = state.get("messages") or []
    for msg in messages[:-1][-6:]:
        role = "User" if msg.get("role", "user") == "user" else "AutoStream Agent"
        safe_content = _sanitize_input(msg.get("content", ""))
        history_text += f"{role}: {safe_content}\n"

    if intent == IntentType.GREETING:
        prompt = (
            "You are a friendly and knowledgeable sales agent for AutoStream, "
            "an AI-powered video editing SaaS for content creators.\n\n"
            f"The user just said: {safe_message}\n\n"
            "Respond warmly and naturally. Briefly mention you can help with "
            "AutoStream plans, pricing, features, and getting started. "
            "Keep it short (2-3 sentences). Sound human, not robotic."
        )
    else:
        lead_note = (
            "Note: This user has already been registered as a lead. "
            "Do NOT ask for their details again.\n\n"
            if lead_captured else ""
        )
        prompt = (
            "You are a knowledgeable and helpful sales agent for AutoStream, "
            "an AI-powered video editing SaaS for content creators.\n\n"
            f"{lead_note}"
            "Use ONLY the following product knowledge to answer accurately:\n\n"
            f"--- KNOWLEDGE BASE ---\n{context}\n--- END ---\n\n"
            f"Conversation so far:\n{history_text}\n"
            f"User: {safe_message}\n\n"
            "Instructions:\n"
            "- Answer accurately using ONLY the knowledge base above\n"
            "- Be conversational and helpful, not robotic\n"
            "- If the user seems interested and hasn't signed up, mention the 7-day free trial\n"
            "- Keep response concise (3-5 sentences max)\n"
            "- Do NOT make up any info not in the knowledge base"
        )

    response = _call_gemini(prompt)
    return {**state, "response": response}


# ─────────────────────────────────────────────
# NODE 4: Lead Qualifier
# ─────────────────────────────────────────────
def qualify_lead(state: AgentState) -> AgentState:
    lead_info        = dict(state.get("lead_info") or {})
    raw_message      = _get_last_message(state)
    collecting_field = state.get("collecting_field")

    if collecting_field == "name":
        safe_name = _sanitize_input(raw_message)
        safe_name = safe_name.split("\n")[0].strip().title()
        if safe_name:
            lead_info["name"] = safe_name
        collecting_field = None

    elif collecting_field == "email":
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", raw_message)
        if email_match:
            lead_info["email"] = email_match.group().lower()
            collecting_field = None
        else:
            return {
                **state,
                "lead_info": lead_info,
                "collecting_field": "email",
                "response": (
                    "Hmm, that doesn't look like a valid email address. "
                    "Could you re-enter it? For example: yourname@gmail.com"
                ),
            }

    elif collecting_field == "platform":
        platform_map = {
            r"\byoutube\b":   "YouTube",
            r"\byt\b":        "YouTube",
            r"\binstagram\b": "Instagram",
            r"\binsta\b":     "Instagram",
            r"\big\b":        "Instagram",
            r"\btiktok\b":    "TikTok",
            r"\btik\s?tok\b": "TikTok",
            r"\bfacebook\b":  "Facebook",
            r"\bfb\b":        "Facebook",
            r"\btwitter\b":   "Twitter/X",
            r"\btwitter/x\b": "Twitter/X",
            r"\bx\.com\b":    "Twitter/X",
        }
        msg_lower       = raw_message.lower()
        matched_platform = None
        for pattern, platform_name in platform_map.items():
            if re.search(pattern, msg_lower):
                matched_platform = platform_name
                break
        lead_info["platform"] = matched_platform if matched_platform else _sanitize_input(raw_message)
        collecting_field = None

    # Decide what to ask next
    if not lead_info.get("name"):
        collecting_field = "name"
        response = (
            "That's great to hear! I'd love to get you set up with AutoStream. "
            "To get started, could I get your full name?"
        )

    elif not lead_info.get("email"):
        collecting_field = "email"
        response = (
            f"Nice to meet you, {lead_info['name']}! "
            "What's the best email address to reach you at?"
        )

    elif not lead_info.get("platform"):
        collecting_field = "platform"
        response = (
            "Almost there! Which platform do you mainly create content for? "
            "(e.g. YouTube, Instagram, TikTok, Facebook)"
        )

    else:
        # FIX BUG-7: check return value of mock_lead_capture
        result = mock_lead_capture(
            name=lead_info["name"],
            email=lead_info["email"],
            platform=lead_info["platform"],
        )
        if result.get("success"):
            response = (
                f"You're all set, {lead_info['name']}!\n\n"
                f"We've saved your details and our team will reach out to "
                f"{lead_info['email']} shortly. In the meantime, you can start your "
                f"7-day free Pro trial at autostream.io — no credit card needed!\n\n"
                "Is there anything else I can help you with?"
            )
        else:
            response = (
                "I'm sorry, there was an issue saving your details. "
                "Please try again in a moment."
            )
        return {
            **state,
            "lead_info":        lead_info,
            "lead_captured":    result.get("success", False),
            "collecting_field": None,
            "response":         response,
        }

    return {
        **state,
        "lead_info":        lead_info,
        "collecting_field": collecting_field,
        "response":         response,
    }
