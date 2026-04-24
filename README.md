# AutoStream AI Agent

A conversational AI agent built for **AutoStream** — a fictional SaaS platform offering automated video editing tools for content creators. This agent handles product queries, detects high-intent users, and captures qualified leads using a structured agentic workflow.

Built with **LangGraph**, **Google Gemini 2.5 Flash Preview**, and a local RAG pipeline.

---

## Features

- **Intent Detection** — classifies each message as a greeting, product inquiry, or high-intent (ready to sign up)
- **RAG-Powered Answers** — retrieves accurate pricing and policy info from a local JSON knowledge base
- **Progressive Lead Capture** — collects name, email, and platform only when the user shows genuine intent
- **Multi-turn Memory** — retains full conversation history across 5–6 turns using LangGraph state
- **Mock Lead Tool** — fires `mock_lead_capture()` only after all three details are confirmed

---

## Project Structure

```
autostream-agent/
│
├── main.py                   # Entry point — starts the CLI chat loop
│
├── graph/
│   ├── state.py              # AgentState TypedDict (shared memory object)
│   ├── nodes.py              # All node functions (intent, RAG, response, lead)
│   └── graph.py              # LangGraph graph builder and router
│
├── utils/
│   └── rag.py                # Knowledge base loader and context retriever
│
├── tools/
│   └── lead_capture.py       # mock_lead_capture() tool
│
├── data/
│   └── knowledge_base.json   # AutoStream pricing, plans, policies, FAQs
│
├── .env.example              # Template for environment variables
├── .gitignore
└── requirements.txt
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Copy the example env file and add your Gemini API key:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

Get a free key at: https://aistudio.google.com/apikey

### 5. Run the agent

```bash
python main.py
```

The CLI chat interface will start. Type your messages and press Enter.

---

## Example Conversation

```
You: Hi there!
Agent: Hey! Welcome to AutoStream. I can help you with our plans, pricing,
       features, or getting started with a free trial. What would you like to know?

You: What's the difference between Basic and Pro?
Agent: Great question! The Basic plan is $29/month and gives you 10 videos/month
       at 720p. The Pro plan is $79/month with unlimited videos, 4K export,
       AI captions, and 24/7 live support. There's also a 7-day free Pro trial
       if you'd like to test it out first!

You: That sounds great, I want to try Pro for my YouTube channel.
Agent: That's great to hear! I'd love to get you set up. Could I get your full name?

You: Rahul Sharma
Agent: Nice to meet you, Rahul! What's the best email address to reach you at?

You: rahul@example.com
Agent: Almost there! Which platform do you mainly create content for?

You: YouTube
Agent: You're all set, Rahul! 🎉 We've saved your details and our team will
       reach out to rahul@example.com shortly...

==================================================
  LEAD CAPTURED SUCCESSFULLY
  Name     : Rahul Sharma
  Email    : rahul@example.com
  Platform : YouTube
==================================================
```

---

## Architecture Explanation

### Why LangGraph?

LangGraph was chosen over AutoGen because it gives explicit, transparent control over the agent's execution flow. Unlike AutoGen's multi-agent conversation model (where agents talk to each other), LangGraph treats the agent as a **directed graph of nodes** — each node is a discrete function with a clear responsibility.

This matters a lot for a lead capture workflow, where the sequence of actions must be predictable. We cannot afford the agent to fire the lead capture tool before collecting all three required fields. With LangGraph's conditional routing, this logic is expressed cleanly as a graph edge — not buried inside a prompt or a nested if-else chain inside an LLM call.

### How State Management Works

The entire conversation lives in a single `AgentState` TypedDict that flows through every node in the graph. At each turn:

1. The user's message is appended to `state["messages"]`
2. The graph is invoked with the current state
3. Each node reads from state and returns an updated copy
4. The updated state is saved and passed into the next turn

This means no external database or session store is needed. The state object itself is the memory — it holds the full conversation history, the classified intent, collected lead fields, and a flag for whether the lead has been captured. This approach keeps the system stateless from an infrastructure perspective while being fully stateful from the agent's perspective.

### RAG Pipeline

Instead of embedding vectors and a vector database (which would be overengineering for this use case), the RAG layer uses keyword-based context retrieval from a structured `knowledge_base.json`. When a user asks about pricing, the retriever pulls the relevant plan details. When they ask about refunds, it pulls the policy section. This context is then injected into the Gemini prompt, ensuring the agent never hallucinates product details.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the integration would use the **WhatsApp Business API** (via Meta's Cloud API) combined with a webhook server.

**How it would work:**

1. **Webhook Setup** — A FastAPI or Flask server is deployed (e.g. on Railway or Render). The `/webhook` endpoint is registered with Meta's WhatsApp Business API. Meta sends a `GET` request to verify the endpoint using a challenge token, which the server echoes back.

2. **Receiving Messages** — When a WhatsApp user sends a message, Meta POSTs a JSON payload to the webhook. The server extracts the sender's phone number and message text.

3. **Agent Invocation** — The server looks up or creates an `AgentState` for that phone number (stored in Redis or a simple dict), appends the new message, and invokes the LangGraph graph — exactly as `main.py` does in the CLI version.

4. **Sending Replies** — After the graph returns a response, the server calls the WhatsApp Cloud API's `messages` endpoint with the reply text, sending it back to the user's number.

5. **Session Persistence** — Since WhatsApp conversations can span hours, state would be serialized to Redis with a TTL (e.g. 24 hours) keyed by phone number. This preserves the multi-turn memory across disconnected sessions.

```python
# Simplified webhook handler (FastAPI example)
@app.post("/webhook")
async def receive_message(payload: dict):
    phone = payload["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    text  = payload["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

    state = redis.get(phone) or initial_state()
    state["messages"].append({"role": "user", "content": text})

    result = graph.invoke(state)
    redis.set(phone, result, ex=86400)

    send_whatsapp_message(phone, result["response"])
```

This approach requires no changes to the core agent logic — only the I/O layer changes from CLI to HTTP.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph 0.2 |
| LLM | Google Gemini 2.5 Flash Preview |
| Knowledge Base | Local JSON (RAG) |
| Language | Python 3.9+ |
| Environment | python-dotenv |

---

## License

MIT — free to use and modify.
