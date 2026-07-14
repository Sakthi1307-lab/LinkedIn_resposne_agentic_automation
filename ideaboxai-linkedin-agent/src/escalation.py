import logging

import requests

from config import settings
from src.sentiment_analyzer import sentiment_analyzer

logger = logging.getLogger(__name__)


class EscalationHandler:
    """
    Decides if an engagement needs manual review before posting.
    """

    def __init__(self):
        self.auto_escalate_enabled = settings.auto_escalate_hostile_comments

    def should_escalate(self, engagement_text: str, persona_tier: str) -> bool:
        """
        Determine if this engagement should be escalated to manual review.

        Only escalate if:
        1. auto_escalate_enabled is True AND
        2. (sentiment is hostile OR VIP + business question)

        If auto_escalate_enabled is False, never escalate — full autonomy
        is the default per the product spec, and this check must fail
        closed (no escalation) rather than open when the feature is off.
        """

        if not self.auto_escalate_enabled:
            logger.info("Escalation disabled, posting automatically")
            return False

        # Check sentiment
        sentiment, risk_score = sentiment_analyzer.classify(engagement_text)

        # >= 0.6, not > 0.6: two hostile keywords (the common case, e.g.
        # "suing" + "fraud") score exactly 0.6 under sentiment_analyzer's
        # 0.3-per-keyword formula. A strict > would silently never escalate
        # the two-keyword case.
        if sentiment == "hostile" and risk_score >= 0.6:
            logger.warning(f"Escalating hostile comment (risk={risk_score:.1%})")
            self.alert_slack(
                f"🚨 HOSTILE COMMENT ESCALATION\n\nRisk: {risk_score:.1%}\nText: {engagement_text[:200]}",
                "hostile",
            )
            return True

        # Check if VIP + business question
        if persona_tier in ("A_internal_leadership", "B_external_vip"):
            business_keywords = ["pricing", "partnership", "collaboration", "integration", "job", "hiring", "board"]
            if any(kw in engagement_text.lower() for kw in business_keywords):
                logger.warning("Escalating VIP business inquiry")
                self.alert_slack(
                    f"📋 VIP BUSINESS INQUIRY\n\nTier: {persona_tier}\nText: {engagement_text[:200]}",
                    "business",
                )
                return True

        return False

    def alert_slack(self, message: str, category: str):
        """Send alert to Slack if webhook is configured."""
        if not settings.slack_webhook_url:
            logger.info(f"Slack not configured, logging escalation: {message}")
            return

        payload = {
            "channel": settings.slack_alert_channel,
            "text": f"[{category.upper()}] IdeaBoxAI Engage",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message,
                    },
                }
            ],
        }

        try:
            requests.post(settings.slack_webhook_url, json=payload, timeout=5)
            logger.info("Slack alert sent")
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")


escalation_handler = EscalationHandler()
