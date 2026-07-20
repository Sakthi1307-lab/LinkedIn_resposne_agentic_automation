"""
Feed dummy engagement payloads through the agent LOCALLY and print its drafted
replies - no LinkedIn, no server, no secrets.

It drives the real FastAPI app in-process via TestClient, hitting the
POST /debug/preview-reply endpoint (which runs the actual persona + reply
pipeline and returns the draft WITHOUT posting anything).

Usage (PowerShell):
  python scripts/local_preview.py                      # uses data/dummy_payloads.json
  python scripts/local_preview.py --file my.json       # your own payloads
  python scripts/local_preview.py --text "Nice post!" --name Sam   # one-off

Edit data/dummy_payloads.json to add your own payloads. Each supports:
  text, type, name, tier, company, role_guess, confidence_score,
  is_vip, voice_note, actor.urn   (plus an optional "note" label)
"""

import argparse
import json
import os
import sys

# Force local-test config BEFORE importing the app so it never demands org
# secrets and uses the offline fallback reply when no OpenRouter key is set.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.environ["LOCAL_TEST_MODE"] = "true"
os.environ["DEBUG"] = "true"

from dotenv import dotenv_values as _dv  # noqa: E402

_ok = _dv(os.path.join(_ROOT, ".env")).get("OPENROUTER_API_KEY", "")
if not _ok or "your_" in _ok:
    os.environ["OPENROUTER_API_KEY"] = ""  # -> offline fallback reply

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402


def load_payloads(args) -> list:
    if args.text:
        return [{"text": args.text, "type": args.type, "name": args.name, "tier": args.tier}]
    path = args.file or os.path.join(_ROOT, "data", "dummy_payloads.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main_cli():
    parser = argparse.ArgumentParser(description="Preview agent replies for dummy payloads")
    parser.add_argument("--file", help="Path to a JSON array of payloads")
    parser.add_argument("--text", help="One-off engagement text (skips the file)")
    parser.add_argument("--name", default="there", help="Commenter name (with --text)")
    parser.add_argument("--tier", default="D_general", help="Persona tier (with --text)")
    parser.add_argument("--type", default="comment", help="Engagement type (with --text)")
    args = parser.parse_args()

    payloads = load_payloads(args)
    using_llm = bool(os.environ.get("OPENROUTER_API_KEY"))

    print("=" * 78)
    print(f"LOCAL AGENT PREVIEW  -  {len(payloads)} payload(s)  -  "
          f"{'LLM-backed' if using_llm else 'offline fallback reply (no OpenRouter key)'}")
    print("=" * 78)

    with TestClient(main.app) as tc:
        for i, p in enumerate(payloads, 1):
            r = tc.post("/debug/preview-reply", json=p)
            print(f"\n[{i}] {p.get('note', p.get('name', 'payload'))}")
            print(f"    persona : {p.get('name','?')} ({p.get('tier','D_general')})"
                  + (f", VIP" if p.get('is_vip') else ""))
            print(f"    comment : {p.get('text','')!r}")
            if r.status_code != 200:
                print(f"    ERROR   : HTTP {r.status_code} - {r.text[:200]}")
                continue
            data = r.json()
            print(f"    type    : {data.get('engagement_type')}")
            print(f"    AGENT > : {data.get('reply')}")

    print("\n" + "=" * 78)
    if not using_llm:
        print("NOTE: offline fallback is intentionally generic and varies only by comment")
        print("      type (question / critical / praise). Set a real sk-or-v1-... OpenRouter")
        print("      key in .env to see full persona/tier-tuned drafts.")


if __name__ == "__main__":
    main_cli()
