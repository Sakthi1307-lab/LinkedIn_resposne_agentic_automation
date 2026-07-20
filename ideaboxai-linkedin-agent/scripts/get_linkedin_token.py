"""
Local LinkedIn OAuth helper — mints a real access token via the 3-legged flow.

It spins up a temporary listener on http://localhost:8000/callback, opens your
browser to LinkedIn's consent screen, catches the redirect, and exchanges the
authorization code for an access token — no manual code-copying.

Prerequisites (LinkedIn developer portal → your app):
  - Auth tab → add redirect URL exactly: http://localhost:8000/callback
  - Auth tab → copy Client ID and Client Secret

Usage (PowerShell):
  $env:LINKEDIN_CLIENT_ID     = "784r8dyvcllgq7"
  $env:LINKEDIN_CLIENT_SECRET = "<your app client secret from the Auth tab>"
  python scripts/get_linkedin_token.py

Scopes: defaults to what your app already has today (member posting). Once the
Community Management API product is approved, re-run with the org scopes:
  python scripts/get_linkedin_token.py --scope "r_organization_social w_organization_social rw_organization_admin"

The token is printed at the end. Add --write-env to also update .env in place.
"""

import argparse
import os
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests

REDIRECT_URI = "http://localhost:8000/callback"
AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
USERINFO_URL = "https://api.linkedin.com/v2/userinfo"

# What your app can grant TODAY (Share on LinkedIn + Sign In). This posts
# comments/replies as YOU (your personal profile). Override with --scope once
# Community Management API is approved to act as the Page instead.
DEFAULT_SCOPE = "openid profile email w_member_social"

# A fixed, non-secret value; LinkedIn echoes it back so we can sanity-check the
# redirect really came from the flow we started.
STATE = "ideaboxai-local-oauth"

_auth_code = {"code": None, "error": None}


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return

        params = urllib.parse.parse_qs(parsed.query)
        _auth_code["code"] = params.get("code", [None])[0]
        _auth_code["error"] = params.get("error_description", params.get("error", [None]))[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        if _auth_code["code"]:
            msg = "Authorization received. You can close this tab and return to the terminal."
        else:
            msg = f"Authorization failed: {_auth_code['error']}"
        self.wfile.write(f"<html><body><h3>{msg}</h3></body></html>".encode())

    def log_message(self, *args):
        pass  # silence the default request logging


def _wait_for_code(timeout=300):
    server = HTTPServer(("localhost", 8000), _CallbackHandler)
    server.timeout = timeout
    thread = threading.Thread(target=server.handle_request)  # one request, then done
    thread.start()
    thread.join(timeout)
    server.server_close()
    return _auth_code["code"], _auth_code["error"]


def _update_env(token: str):
    """Replace LINKEDIN_ACCESS_TOKEN in .env, preserving everything else."""
    path = ".env"
    if not os.path.exists(path):
        print("  (no .env found; skipping --write-env)")
        return
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith("LINKEDIN_ACCESS_TOKEN="):
            lines[i] = f"LINKEDIN_ACCESS_TOKEN={token}\n"
            replaced = True
            break
    if not replaced:
        lines.append(f"LINKEDIN_ACCESS_TOKEN={token}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("  .env updated (LINKEDIN_ACCESS_TOKEN)")


def main():
    parser = argparse.ArgumentParser(description="Mint a LinkedIn access token via local OAuth")
    parser.add_argument("--scope", default=DEFAULT_SCOPE, help="Space-separated OAuth scopes")
    parser.add_argument("--write-env", action="store_true", help="Write the token into .env")
    args = parser.parse_args()

    # Prefer real environment variables, but fall back to .env so you can just
    # fill the file once instead of exporting shell vars every run.
    from dotenv import dotenv_values

    env_file = dotenv_values(".env")
    client_id = os.environ.get("LINKEDIN_CLIENT_ID") or env_file.get("LINKEDIN_CLIENT_ID")
    client_secret = os.environ.get("LINKEDIN_CLIENT_SECRET") or env_file.get("LINKEDIN_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("ERROR: LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET not found.")
        print("  Add them to your .env file (or export them as env vars):")
        print("    LINKEDIN_CLIENT_ID=784r8dyvcllgq7")
        print("    LINKEDIN_CLIENT_SECRET=<from your app's Auth tab>")
        sys.exit(1)

    if "your_" in client_secret or client_secret.startswith("<"):
        print("ERROR: LINKEDIN_CLIENT_SECRET is still a placeholder. Paste the real")
        print("  Primary Client Secret from the app's Auth tab (click 'Show').")
        sys.exit(1)

    auth_params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": args.scope,
        "state": STATE,
    }
    auth_link = f"{AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

    print("Requesting scopes:", args.scope)
    print("Opening browser for LinkedIn consent...")
    print("If it doesn't open, paste this URL manually:\n ", auth_link, "\n")
    webbrowser.open(auth_link)

    print("Waiting for the redirect to localhost:8000/callback (up to 5 min)...")
    code, error = _wait_for_code()
    if not code:
        print("ERROR: no authorization code received.", f"({error})" if error else "")
        sys.exit(1)
    print("Got authorization code. Exchanging for a token...")

    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": REDIRECT_URI,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    if resp.status_code != 200:
        print(f"ERROR: token exchange failed (HTTP {resp.status_code}): {resp.text[:500]}")
        sys.exit(1)

    payload = resp.json()
    token = payload.get("access_token", "")
    print("\n" + "=" * 70)
    print("ACCESS TOKEN (valid ~60 days):")
    print(token)
    print("Scopes granted:", payload.get("scope", "(not reported)"))
    print("Expires in (s):", payload.get("expires_in", "?"))
    print("=" * 70)

    # Resolve the member URN so you can post AS yourself with w_member_social.
    try:
        who = requests.get(USERINFO_URL, headers={"Authorization": f"Bearer {token}"}, timeout=15)
        if who.status_code == 200:
            sub = who.json().get("sub", "")
            print("Your member URN:  urn:li:person:%s" % sub)
            print("Signed in as:    ", who.json().get("name", "?"))
    except Exception as e:
        print("(could not fetch userinfo:", e, ")")

    if args.write_env:
        _update_env(token)


if __name__ == "__main__":
    main()
