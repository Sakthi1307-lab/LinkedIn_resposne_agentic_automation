"""
Tests for the LinkedIn webhook validation layer.

Covers the two security-critical HMAC schemes LinkedIn specifies
(https://learn.microsoft.com/en-us/linkedin/shared/api-guide/webhook-validation):
  1. Endpoint-validation challenge: challengeResponse = hex(HMAC(clientSecret, challengeCode))
  2. Event signature: X-LI-Signature = hex(HMAC(clientSecret, "hmacsha256=" + raw_body))
"""

import hashlib
import hmac

import pytest

from src.linkedin_client import LinkedInClient

SECRET = "unit-test-client-secret"


@pytest.fixture
def client(monkeypatch):
    """A LinkedInClient whose webhook secret is a known value."""
    from config import settings

    monkeypatch.setattr(settings, "linkedin_client_secret", SECRET)
    return LinkedInClient()


def _expected_challenge(code: str) -> str:
    return hmac.new(SECRET.encode(), code.encode(), hashlib.sha256).hexdigest()


def _sign(body: bytes) -> str:
    return hmac.new(SECRET.encode(), b"hmacsha256=" + body, hashlib.sha256).hexdigest()


# ── Challenge handshake ──────────────────────────────────────────────────────

def test_challenge_response_matches_spec(client):
    code = "890e4665-4dfe-4ab1-b689-ed553bceeed0"
    assert client.compute_challenge_response(code) == _expected_challenge(code)


def test_challenge_response_is_deterministic(client):
    code = "abc-123"
    assert client.compute_challenge_response(code) == client.compute_challenge_response(code)


def test_challenge_response_is_lowercase_hex(client):
    resp = client.compute_challenge_response("some-code")
    assert resp == resp.lower()
    assert len(resp) == 64  # SHA-256 hex digest


# ── Event signature verification ─────────────────────────────────────────────

def test_valid_signature_accepted(client):
    body = b'{"id":"evt-1","type":"COMMENT"}'
    assert client.verify_webhook_signature(body, _sign(body)) is True


def test_tampered_body_rejected(client):
    body = b'{"id":"evt-1"}'
    sig = _sign(body)
    assert client.verify_webhook_signature(body + b" ", sig) is False


def test_wrong_signature_rejected(client):
    body = b'{"id":"evt-1"}'
    assert client.verify_webhook_signature(body, "deadbeef") is False


def test_missing_signature_rejected(client):
    assert client.verify_webhook_signature(b'{"id":"evt-1"}', "") is False


def test_signature_requires_the_hmacsha256_prefix(client):
    """A signature computed WITHOUT the literal 'hmacsha256=' prefix must fail."""
    body = b'{"id":"evt-1"}'
    wrong = hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()  # no prefix
    assert client.verify_webhook_signature(body, wrong) is False


def test_no_secret_configured_fails_closed(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "linkedin_client_secret", None)
    monkeypatch.setattr(settings, "linkedin_webhook_verification_token", None)
    c = LinkedInClient()
    body = b'{"id":"evt-1"}'
    assert c.verify_webhook_signature(body, "anything") is False
