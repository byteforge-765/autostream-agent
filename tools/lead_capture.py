import json
from datetime import datetime
from pathlib import Path

# FIX BUG #5 — Use absolute path so leads.json always saves in the
# correct location regardless of which directory the user runs main.py from.
_DATA_DIR = Path(__file__).parent.parent / "data"
_LEADS_FILE = _DATA_DIR / "leads.json"


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock lead capture function.
    In production this would POST to a CRM API (e.g. HubSpot, Salesforce).
    Here it prints a confirmation banner and persists to a local JSON file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 50)
    print("  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 50)
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Time     : {timestamp}")
    print("=" * 50 + "\n")

    lead_record = {
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": timestamp,
        "source": "AutoStream AI Agent",
    }

    # Persist to local file (acts as a simple mock database)
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_LEADS_FILE, "r", encoding="utf-8") as f:
                leads = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            leads = []

        leads.append(lead_record)

        with open(_LEADS_FILE, "w", encoding="utf-8") as f:
            json.dump(leads, f, indent=2, ensure_ascii=False)

    except OSError as e:
        # Non-critical — log but do not crash the agent
        print(f"[Warning] Could not save lead to file: {e}")

    return {
        "success": True,
        "message": f"Lead captured successfully for {name}",
        "lead_id": f"AS-{int(datetime.now().timestamp())}",
        "data": lead_record,
    }
