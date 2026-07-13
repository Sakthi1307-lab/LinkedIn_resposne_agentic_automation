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

    def __init__(self):
        self.token = settings.linkedin_access_token
        self.org_urn = settings.linkedin_organization_urn
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        logger.info(f"LinkedInClient initialized for org: {self.org_urn}")

    # ═════════════════════════════════════════════════════════════════════════
    # WEBHOOK VERIFICATION
    # ═════════════════════════════════════════════════════════════════════════

    def verify_webhook_signature(self, body: bytes, signature: str) -> bool:
        """
        Verify that a webhook payload came from LinkedIn.
        Uses HMAC-SHA256 with the verification token as the secret.
        """
        try:
            if not signature:
                logger.warning("Webhook signature verification failed: no signature header present")
                return False

            expected_signature = hmac.new(
                settings.linkedin_webhook_verification_token.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()

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

    def get_new_comments(self, limit: int = 10, start_timestamp: int = None) -> list:
        """
        Fetch new comments on the organization page.

        Real API: GET /mmComments (search by organization)
        Query by organization URN and creation time filter.

        Returns: list of comment objects with:
        - urn (comment URN)
        - text (comment text)
        - actor (person who commented, includes URN)
        - created (timestamp)
        - parentActivity (the post that was commented on)

        Returns [] on any failure — callers should treat that as "nothing new
        this cycle" rather than distinguishing "no comments" from "API down".
        """
        try:
            query_params = {
                "q": "organization",
                "organizationId": self.org_urn,
                "limit": limit,
                "sortBy": "RECENT",
            }

            if start_timestamp:
                query_params["createdTimeFrom"] = start_timestamp

            logger.info(f"Fetching comments from LinkedIn (limit={limit})")
            response = requests.get(
                f"{self.BASE_URL}/mmComments",
                headers=self.headers,
                params=query_params,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                comments = data.get("elements", [])
                logger.info(f"Fetched {len(comments)} comments")
                return comments
            elif response.status_code in (401, 403):
                self._log_auth_error("fetching comments", response)
                return []
            else:
                self._log_auth_error("fetching comments", response)
                return []

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
    # POST REPLY TO COMMENT
    # ═════════════════════════════════════════════════════════════════════════

    def post_reply(self, comment_urn: str, reply_text: str) -> bool:
        """
        Post a reply to an existing comment.

        API: POST /mmComments
        Payload: parentComment (the comment URN we're replying to)
                 commentContent (the reply text)

        Returns: True if successful, False otherwise (never raises).
        """
        try:
            if not comment_urn:
                logger.warning("post_reply called with empty comment_urn, skipping")
                return False

            if not reply_text or len(reply_text.strip()) == 0:
                logger.warning("Empty reply text, skipping")
                return False

            if len(reply_text) > 500:
                logger.warning(f"Reply exceeds 500 chars ({len(reply_text)}), truncating")
                reply_text = reply_text[:497] + "..."

            payload = {
                "parentComment": comment_urn,
                "commentContent": {"text": reply_text},
            }

            logger.info(f"Posting reply to comment: {comment_urn[:50]}...")
            response = requests.post(
                f"{self.BASE_URL}/mmComments",
                headers=self.headers,
                json=payload,
                timeout=10,
            )

            if response.status_code in (200, 201):
                result = response.json()
                reply_urn = result.get("urn", "unknown")
                logger.info(f"Reply posted successfully: {reply_urn}")
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
