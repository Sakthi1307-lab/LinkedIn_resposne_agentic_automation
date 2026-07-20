import hashlib
import hmac
import logging
from datetime import datetime

import requests

from config import settings

logger = logging.getLogger(__name__)


class LinkedInClient:
    """
    Official LinkedIn Community Management API wrapper.
    Reference: https://learn.microsoft.com/en-us/linkedin/marketing/community-management-api

    Every public method here degrades gracefully (logs and returns an empty/
    falsy value) instead of raising. This client is called from an async
    background task in the engagement pipeline — an uncaught exception there
    would silently kill processing for that engagement with no trace beyond
    the task's own error log.
    """

    BASE_URL = "https://api.linkedin.com/v2"
    # The Community Management API lives under the versioned /rest tree and
    # requires a LinkedIn-Version (YYYYMM) header on every call.
    REST_BASE = "https://api.linkedin.com/rest"

    def __init__(self):
        self.token = settings.linkedin_access_token
        self.org_urn = settings.linkedin_organization_urn
        self.api_version = getattr(settings, "linkedin_api_version", "202606")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        # Versioned headers for every Community Management (/rest) call.
        self.rest_headers = {**self.headers, "LinkedIn-Version": self.api_version}
        logger.info(
            f"LinkedInClient initialized for org: {self.org_urn} (API version {self.api_version})"
        )

    @staticmethod
    def _encode_urn(urn: str) -> str:
        """URL-encode a URN for use as a /rest path segment."""
        import urllib.parse

        return urllib.parse.quote(urn, safe="")

    # ═════════════════════════════════════════════════════════════════════════
    # WEBHOOK VERIFICATION
    # ═════════════════════════════════════════════════════════════════════════

    def _webhook_secret(self) -> str:
        """
        The HMAC key for BOTH the challenge handshake and event signatures is
        the app's OAuth Client Secret (per LinkedIn's spec). Fall back to the
        legacy verification token only if the client secret isn't configured,
        so older setups don't silently break — but the client secret is correct.
        """
        return settings.linkedin_client_secret or settings.linkedin_webhook_verification_token or ""

    def compute_challenge_response(self, challenge_code: str) -> str:
        """
        Answer LinkedIn's webhook endpoint-validation challenge.

        LinkedIn GETs the webhook URL with ?challengeCode=<uuid>. We must return
        JSON {"challengeCode", "challengeResponse"} with 200 within 3 seconds,
        where:
            challengeResponse = lowercase_hex( HMACSHA256(key=clientSecret, msg=challengeCode) )

        Spec: https://learn.microsoft.com/en-us/linkedin/shared/api-guide/webhook-validation
        """
        return hmac.new(
            self._webhook_secret().encode(),
            challenge_code.encode(),
            hashlib.sha256,
        ).hexdigest()

    def verify_webhook_signature(self, body: bytes, signature: str) -> bool:
        """
        Verify an incoming webhook event POST came from LinkedIn.

        Per spec, the X-LI-Signature header is:
            lowercase_hex( HMACSHA256(key=clientSecret, msg="hmacsha256=" + raw_body) )
        The literal prefix "hmacsha256=" is part of the signed message only; the
        header itself carries just the hex digest. The body MUST be the raw bytes
        exactly as received — any re-serialization breaks the match.
        """
        try:
            if not signature:
                logger.warning("Webhook signature verification failed: no X-LI-Signature header present")
                return False

            secret = self._webhook_secret()
            if not secret:
                logger.error("Webhook signature check failed: no LINKEDIN_CLIENT_SECRET configured")
                return False

            string_to_sign = b"hmacsha256=" + body
            expected_signature = hmac.new(secret.encode(), string_to_sign, hashlib.sha256).hexdigest()

            is_valid = hmac.compare_digest(expected_signature, signature)
            if is_valid:
                logger.info("Webhook signature verified")
            else:
                logger.warning("Webhook signature verification failed: signature mismatch")
            return is_valid
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

    # ═════════════════════════════════════════════════════════════════════════
    # AUTH ERROR LOGGING (shared by every authenticated call)
    # ═════════════════════════════════════════════════════════════════════════

    def _log_auth_error(self, action: str, response: requests.Response) -> None:
        """
        401 and 403 need different fixes: 401 means the token itself is
        rejected (expired/revoked/malformed); 403 means the token is valid
        but the app/token doesn't have the scope or partner approval for
        this action. Logging them identically hides which one you're facing.
        """
        if response.status_code == 401:
            logger.error(
                f"LinkedIn API: 401 Unauthorized while {action}. "
                f"The access token is missing, expired, or invalid — refresh/reissue it. "
                f"Body: {response.text[:300]}"
            )
        elif response.status_code == 403:
            logger.error(
                f"LinkedIn API: 403 Forbidden while {action}. "
                f"Token was accepted but lacks the required scope, or the app isn't "
                f"approved for the Community Management API on this organization. "
                f"Body: {response.text[:300]}"
            )
        else:
            logger.error(
                f"LinkedIn API error while {action}: {response.status_code} - {response.text[:300]}"
            )

    # ═════════════════════════════════════════════════════════════════════════
    # FETCH ENGAGEMENT (COMMENTS, REACTIONS, MENTIONS)
    # ═════════════════════════════════════════════════════════════════════════

    def get_organization_post_urns(self, limit: int = 10) -> list:
        """
        List the URNs of the organization's most recent posts.

        Community Management API: GET /rest/posts?q=author&author={orgUrn}
        Requires the r_organization_social scope. Returns [] on any failure.

        NOTE: documented against the CM API but unverified against a live token
        until this app is granted Community Management API access.
        """
        try:
            response = requests.get(
                f"{self.REST_BASE}/posts",
                headers=self.rest_headers,
                params={"q": "author", "author": self.org_urn, "count": limit, "sortBy": "LAST_MODIFIED"},
                timeout=10,
            )
            if response.status_code == 200:
                elements = response.json().get("elements", [])
                # The posts API returns each post's URN under "id".
                return [e.get("id") for e in elements if e.get("id")]
            self._log_auth_error("listing organization posts", response)
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error listing organization posts: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing organization posts: {e}", exc_info=True)
            return []

    def get_new_comments(self, limit: int = 10, start_timestamp: int = None) -> list:
        """
        Fetch recent comments on the organization page, normalized to the event
        shape the pipeline consumes.

        Real Community Management API flow (there is no single "all comments on
        my org" endpoint): list recent posts, then read each post's comments via
        GET /rest/socialActions/{postUrn}/comments. Requires r_organization_social.

        Returns a list of dicts, each:
        - urn            (the comment's own URN — the reply target)
        - text           (comment text)
        - actor          ({"urn": commenter person URN})
        - type           ("comment")
        - created        (epoch millis)
        - parentActivity (the post URN the comment is on)

        Returns [] on any failure. Documented against the CM API but unverified
        against a live token until this app has Community Management API access.
        """
        try:
            post_urns = self.get_organization_post_urns(limit=limit)
            if not post_urns:
                logger.info("No organization posts returned (or no access yet)")
                return []

            comments: list = []
            for post_urn in post_urns:
                url = f"{self.REST_BASE}/socialActions/{self._encode_urn(post_urn)}/comments"
                params = {"count": limit}
                if start_timestamp:
                    params["createdTimeFrom"] = start_timestamp
                response = requests.get(url, headers=self.rest_headers, params=params, timeout=10)

                if response.status_code != 200:
                    self._log_auth_error(f"fetching comments on {post_urn}", response)
                    continue

                for c in response.json().get("elements", []):
                    comments.append(
                        {
                            "urn": c.get("$URN") or c.get("id") or c.get("commentUrn", ""),
                            "text": (c.get("message") or {}).get("text", ""),
                            "actor": {"urn": c.get("actor", "")},
                            "type": "comment",
                            "created": (c.get("created") or {}).get("time"),
                            "parentActivity": post_urn,
                        }
                    )

            logger.info(f"Fetched {len(comments)} comments across {len(post_urns)} posts")
            return comments[:limit]

        except requests.exceptions.Timeout:
            logger.error("LinkedIn API timeout (10s) while fetching comments")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching comments: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching comments: {e}", exc_info=True)
            return []

    # ═════════════════════════════════════════════════════════════════════════
    # PRE-FLIGHT CONNECTIVITY CHECK
    # ═════════════════════════════════════════════════════════════════════════

    def check_connectivity(self) -> dict:
        """
        Pre-flight check: can this token actually list posts on the
        configured organization page — the first real capability
        get_new_comments() depends on (it lists posts, then reads each
        post's comments; no call can read comments without first listing
        posts, so this is the earliest point a scope/access problem shows up).

        Deliberately exercises the real GET /rest/posts call (same one
        get_organization_post_urns() makes, count=1) rather than a proxy
        signal like /v2/me: a token can be "valid" and still lack Community
        Management API scope for this specific organization (unverified
        app, no partner approval, wrong org URN), and that's exactly the
        failure mode worth catching before going live. get_new_comments()
        swallows every failure into [], which is the right behavior for the
        polling loop but useless for diagnosing *why* — this returns the
        real status instead.
        """
        try:
            response = requests.get(
                f"{self.REST_BASE}/posts",
                headers=self.rest_headers,
                params={"q": "author", "author": self.org_urn, "count": 1, "sortBy": "LAST_MODIFIED"},
                timeout=10,
            )
            if response.status_code == 200:
                return {"ok": True, "detail": f"Token can list posts on {self.org_urn} via the Community Management API."}
            elif response.status_code == 401:
                return {"ok": False, "detail": "401 Unauthorized — token is missing, expired, or invalid."}
            elif response.status_code == 403:
                return {
                    "ok": False,
                    "detail": (
                        "403 Forbidden — token lacks Community Management API scope "
                        "(r_organization_social) for this organization, the app isn't "
                        "partner-approved for it, or the app isn't verified as associated "
                        "with this page. A Sign-In or Share-on-LinkedIn-only token will 403 here."
                    ),
                }
            else:
                return {"ok": False, "detail": f"Unexpected status {response.status_code}: {response.text[:200]}"}
        except requests.exceptions.Timeout:
            return {"ok": False, "detail": "Timed out reaching LinkedIn API (10s)."}
        except requests.exceptions.RequestException as e:
            return {"ok": False, "detail": f"Network error reaching LinkedIn API: {e}"}

    # ═════════════════════════════════════════════════════════════════════════
    # POST REPLY TO COMMENT
    # ═════════════════════════════════════════════════════════════════════════

    def post_reply(self, comment_urn: str, reply_text: str) -> bool:
        """
        Reply to an existing comment AS the organization (the Page).

        Community Management API: POST /rest/socialActions/{commentUrn}/comments
        with actor = the organization URN. Posting to the comment's own social
        action creates a nested reply to that comment. Requires the
        w_organization_social scope (part of Community Management API access).

        Returns True on success, False otherwise (never raises). Documented
        against the CM API but unverified against a live token until this app
        has Community Management API access.
        """
        try:
            if not comment_urn:
                logger.warning("post_reply called with empty comment_urn, skipping")
                return False

            if not reply_text or len(reply_text.strip()) == 0:
                logger.warning("Empty reply text, skipping")
                return False

            if not self.org_urn:
                logger.error("post_reply: no organization URN configured; cannot post as the Page")
                return False

            if len(reply_text) > 500:
                logger.warning(f"Reply exceeds 500 chars ({len(reply_text)}), truncating")
                reply_text = reply_text[:497] + "..."

            payload = {
                "actor": self.org_urn,
                "message": {"text": reply_text},
            }

            url = f"{self.REST_BASE}/socialActions/{self._encode_urn(comment_urn)}/comments"
            logger.info(f"Posting reply as {self.org_urn} to comment: {comment_urn[:50]}...")
            response = requests.post(url, headers=self.rest_headers, json=payload, timeout=10)

            if response.status_code in (200, 201):
                logger.info("Reply posted successfully")
                return True
            elif response.status_code in (401, 403):
                self._log_auth_error("posting reply", response)
                return False
            else:
                self._log_auth_error("posting reply", response)
                return False

        except requests.exceptions.Timeout:
            logger.error("LinkedIn API timeout while posting reply")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error posting reply: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error posting reply: {e}", exc_info=True)
            return False

    # ═════════════════════════════════════════════════════════════════════════
    # GET PERSON PROFILE
    # ═════════════════════════════════════════════════════════════════════════

    def get_profile(self, linkedin_urn: str) -> dict:
        """
        Fetch public profile info for a person by URN.

        LIMITATION: LinkedIn restricts third-party access to member profile
        data. This endpoint may return limited info depending on your app's
        scopes and approval level. Most apps can only get name and headline
        for the authenticated member (via /me), not arbitrary member URNs or
        company affiliation.

        Returns: dict with available fields, or {} on failure/no access:
        {
            "name": "John Doe",
            "headline": "VP Sales at ExampleCorp",
            "company": None,  # Usually not available to third-party apps
        }
        """
        try:
            if not linkedin_urn or not linkedin_urn.startswith("urn:li:person:"):
                logger.warning(f"Invalid person URN format: {linkedin_urn}")
                return {}

            logger.info(f"Fetching profile for: {linkedin_urn}")

            # The Community Management API only exposes /me (the authenticated
            # member), not arbitrary third-party profiles by URN. We still hit
            # /me here as the best-effort call our app is actually approved
            # for; a real per-member lookup would need Marketing Developer
            # Platform / Partner APIs the app may not have.
            response = requests.get(
                f"{self.BASE_URL}/me",
                headers=self.headers,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                profile = {
                    "name": f"{data.get('localizedFirstName', '')} {data.get('localizedLastName', '')}".strip(),
                    "headline": data.get("headline", ""),
                    "company": None,  # Not available via /me
                }
                logger.info(f"Profile fetched: {profile['name']} - {profile['headline']}")
                return profile
            elif response.status_code in (401, 403):
                self._log_auth_error("fetching profile", response)
                return {}
            else:
                self._log_auth_error("fetching profile", response)
                return {}

        except requests.exceptions.Timeout:
            logger.error("LinkedIn API timeout while fetching profile")
            return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching profile: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error fetching profile: {e}", exc_info=True)
            return {}

    # ═════════════════════════════════════════════════════════════════════════
    # POST COMMENT AS A MEMBER (w_member_social path)
    # ═════════════════════════════════════════════════════════════════════════

    def post_member_comment(
        self,
        post_urn: str,
        reply_text: str,
        actor_urn: str,
        api_version: str = "202606",
    ) -> bool:
        """
        Post a comment on `post_urn` AS a member (the personal profile in
        `actor_urn`), using the w_member_social scope.

        This is the only LinkedIn write path this app currently has access to.
        It does NOT act as the organization/Page — the comment is authored by
        the member. Reading the engagement to reply to still requires the
        Community Management API (partner-gated), so callers must supply the
        target `post_urn` themselves.

        Returns True on success, False on any failure (never raises).
        """
        try:
            if not post_urn or not actor_urn:
                logger.warning("post_member_comment: missing post_urn or actor_urn, skipping")
                return False
            if not reply_text or not reply_text.strip():
                logger.warning("post_member_comment: empty reply text, skipping")
                return False

            import urllib.parse

            encoded = urllib.parse.quote(post_urn, safe="")
            url = f"https://api.linkedin.com/rest/socialActions/{encoded}/comments"
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "X-Restli-Protocol-Version": "2.0.0",
                "LinkedIn-Version": api_version,
            }
            payload = {"actor": actor_urn, "message": {"text": reply_text}}

            logger.info(f"Posting member comment on {post_urn[:50]}... as {actor_urn}")
            response = requests.post(url, headers=headers, json=payload, timeout=10)

            if response.status_code in (200, 201):
                logger.info("Member comment posted successfully")
                return True
            elif response.status_code in (401, 403):
                self._log_auth_error("posting member comment", response)
                return False
            else:
                self._log_auth_error("posting member comment", response)
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error posting member comment: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error posting member comment: {e}", exc_info=True)
            return False

    # ═════════════════════════════════════════════════════════════════════════
    # SEND DM (NOT SUPPORTED)
    # ═════════════════════════════════════════════════════════════════════════

    def send_dm(self, recipient_urn: str, message_text: str) -> bool:
        """
        Documented stub — NOT implemented.

        The Community Management API this client is built against does not
        expose DM/messaging endpoints; sending a LinkedIn DM requires the
        separate Messaging API with its own partner approval and scopes,
        which this app does not have. Returning False here is the correct
        and only honest behavior — do not attempt to fake this by, e.g.,
        posting a public comment instead.
        """
        logger.warning(
            "send_dm() is unsupported: Community Management API has no DM/messaging "
            "endpoint. Requires the separate LinkedIn Messaging API (different partner "
            "scope), which is not integrated."
        )
        return False

    # ═════════════════════════════════════════════════════════════════════════
    # HELPER: PARSE COMMENT PAYLOAD
    # ═════════════════════════════════════════════════════════════════════════

    def parse_comment_payload(self, payload: dict) -> dict:
        """
        Parse a LinkedIn webhook payload into a normalized structure.

        Returns: {
            "linkedin_urn": person URN,
            "comment_urn": comment URN,
            "text": comment text,
            "type": engagement type,
            "timestamp": creation timestamp,
        }
        Returns {} if the payload can't be parsed at all.
        """
        try:
            return {
                "linkedin_urn": payload.get("actor", {}).get("urn", ""),
                "comment_urn": payload.get("urn", ""),
                "text": payload.get("text", ""),
                "type": "comment",
                "timestamp": payload.get("created", datetime.utcnow().isoformat()),
            }
        except Exception as e:
            logger.error(f"Failed to parse comment payload: {e}")
            return {}


# Initialize singleton
linkedin_client = LinkedInClient()
