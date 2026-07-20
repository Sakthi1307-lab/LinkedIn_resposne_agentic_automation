"""
Live system test for the engagement agent (member/w_member_social path).

Because LinkedIn blocks reading comments without Community Management API,
this test supplies the "engagement" manually: you give the post to comment on
and (optionally) the text of an incoming comment. The agent then runs its real
pipeline — persona -> on-brand reply draft -> voice lint -> post — and publishes
the reply as a comment on your post, as YOU.

Usage (PowerShell), after get_linkedin_token.py has populated .env:
  python scripts/system_test.py --post-url "https://www.linkedin.com/feed/update/urn:li:activity:7350000000000000000/" --comment "How do you keep AI on-brand at scale?"

  # dry run — draft the reply but DON'T post:
  python scripts/system_test.py --post-url "<url>" --comment "..." --dry-run

Accepts either a full post URL or a bare urn:li:activity:... / urn:li:share:...
"""

import argparse
import re
import sys

import requests
from dotenv import dotenv_values

# Force local-test config so importing the app doesn't demand org secrets.
import os
os.environ.setdefault("LOCAL_TEST_MODE", "true")
os.environ.setdefault("DEBUG", "true")

# Allow running from anywhere: put the project root (parent of scripts/) on the path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# If OPENROUTER_API_KEY is unset/placeholder, blank it so config validation
# passes and the agent uses its offline fallback reply (fine for a live test
# of the LinkedIn write path — swap in a real sk-or-v1-... key for LLM drafts).
from dotenv import dotenv_values as _dv  # noqa: E402

_ok = _dv(os.path.join(_ROOT, ".env")).get("OPENROUTER_API_KEY", "")
if not _ok or "your_" in _ok:
    os.environ["OPENROUTER_API_KEY"] = ""

from src.linkedin_client import linkedin_client  # noqa: E402
from src.persona_resolver import PersonaContext  # noqa: E402
from src.response_generator import response_generator  # noqa: E402

USERINFO_URL = "https://api.linkedin.com/v2/userinfo"


def extract_post_urn(value: str) -> str:
    """Pull a urn:li:activity/share/ugcPost URN out of a URL or raw string."""
    m = re.search(r"urn:li:(?:activity|share|ugcPost):[0-9]+", value)
    if m:
        return m.group(0)
    # A bare numeric id in a /feed/update/ URL without the urn prefix:
    m = re.search(r"(?:activity|update)[:/](\d{15,})", value)
    if m:
        return f"urn:li:activity:{m.group(1)}"
    return ""


def resolve_actor_urn(token: str) -> str:
    r = requests.get(USERINFO_URL, headers={"Authorization": f"Bearer {token}"}, timeout=15)
    r.raise_for_status()
    return f"urn:li:person:{r.json()['sub']}"


def main():
    parser = argparse.ArgumentParser(description="Live system test of the engagement agent")
    parser.add_argument("--post-url", required=True, help="URL or URN of the post to comment on")
    parser.add_argument("--comment", default="", help="Text of the incoming engagement to reply to")
    parser.add_argument("--name", default="there", help="Commenter's first name (for persona)")
    parser.add_argument("--tier", default="D_general", help="Persona tier")
    parser.add_argument("--dry-run", action="store_true", help="Draft only; do not post")
    args = parser.parse_args()

    env = dotenv_values(".env")
    token = env.get("LINKEDIN_ACCESS_TOKEN", "")
    if not token:
        print("ERROR: no LINKEDIN_ACCESS_TOKEN in .env — run scripts/get_linkedin_token.py first.")
        sys.exit(1)

    post_urn = extract_post_urn(args.post_url)
    if not post_urn:
        print(f"ERROR: couldn't find a post URN in: {args.post_url}")
        print("  Paste the full post URL, or a urn:li:activity:... / urn:li:share:... value.")
        sys.exit(1)

    try:
        actor_urn = resolve_actor_urn(token)
    except Exception as e:
        print(f"ERROR: could not resolve your member URN from the token: {e}")
        sys.exit(1)

    # The "engagement" the agent reacts to. If no comment text was given, treat
    # it as a light praise engagement so the pipeline still has something to work with.
    engagement_text = args.comment or "Really enjoyed this — great perspective."

    persona = PersonaContext(
        linkedin_urn="urn:li:person:external",
        name=args.name,
        tier=args.tier,
        relationship_to_us="unknown",
        company=None,
        role_guess=None,
        confidence_score=0.5,
        is_vip=False,
        is_own_ceo=False,
        voice_note=None,
    )

    print("=" * 72)
    print("LIVE SYSTEM TEST — engagement agent (member path)")
    print("=" * 72)
    print("Target post URN :", post_urn)
    print("Posting as      :", actor_urn)
    print("Incoming comment:", engagement_text)
    print("-" * 72)

    # THE AGENT: draft an on-brand reply through the real pipeline.
    reply = response_generator.generate(engagement_text, persona, "comment")
    print("Agent drafted   :", reply)
    print("-" * 72)

    if args.dry_run:
        print("DRY RUN — not posting. Re-run without --dry-run to publish it live.")
        return

    ok = linkedin_client.post_member_comment(post_urn, reply, actor_urn)
    if ok:
        print("RESULT: ✅ posted live. Open your post on LinkedIn to see the comment.")
    else:
        print("RESULT: ❌ post failed — see the logged LinkedIn API error above.")
        print("If it's 403 partnerApi, the comment WRITE also needs Community Management API.")


if __name__ == "__main__":
    main()
