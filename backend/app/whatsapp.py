import requests, asyncio
from fastapi import HTTPException
from app.config import WHATSAPP_VERIFY_TOKEN, WHATSAPP_TOKEN, WHATSAPP_PHONE_NUMBER_ID
from app.rag import ask_policies

GRAPH_URL = "https://graph.facebook.com/v20.0"

def verify(mode: str, challenge: str, token: str):
    # Meta sends as hub.mode/hub.challenge/hub.verify_token (FastAPI mapped names differ)
    # Accept both
    if token == WHATSAPP_VERIFY_TOKEN and mode == "subscribe":
        return int(challenge) if challenge.isdigit() else challenge
    raise HTTPException(status_code=403, detail="Verification failed")

async def send_text(to: str, text: str):
    if not (WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID):
        return
    url = f"{GRAPH_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }
    r = requests.post(url, headers=headers, json=data, timeout=15)
    try:
        r.raise_for_status()
    except Exception as e:
        # Log and ignore
        print("WhatsApp send error:", e, r.text)

def format_answer_for_wa(answer_dict: dict) -> str:
    # Keep it short on WA, add 1–2 citations
    ans = answer_dict.get("answer", "").strip()
    cits = answer_dict.get("citations", [])[:2]
    lines = [ans]
    if cits:
        lines.append("\nCitations:")
        for c in cits:
            frag = f"- {c.get('doc_id')} (page {c.get('page','?')})"
            lines.append(frag)
    return "\n".join(lines)[:1200]  # WA limit safety

async def receive(payload: dict):
    # Extract messages (Meta format)
    try:
        entries = payload.get("entry", [])
        for entry in entries:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                for m in messages:
                    from_ = m.get("from")
                    text = (m.get("text") or {}).get("body", "").strip()
                    if not text:
                        continue
                    # Ask RAG
                    data = ask_policies(text, k=5, filters=None)
                    msg = format_answer_for_wa(data)
                    await send_text(from_, msg)
    except Exception as e:
        print("WA webhook error:", e)
    return {"ok": True}
