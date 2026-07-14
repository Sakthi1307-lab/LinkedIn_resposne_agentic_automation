# NOTE: this is a v1, keyword-matching sentiment classifier. It is
# deliberately crude — no model, no context awareness, no negation handling
# ("not broken" still matches "broken"). It exists to give escalation.py a
# cheap first-pass signal for hostile/critical content, not to be a
# production-grade sentiment engine. Swap in a real classifier (or route
# through the LLM client) before relying on this for anything beyond a
# coarse escalation trigger.

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Quick sentiment classification for escalation detection.
    """

    HOSTILE_KEYWORDS = [
        "lawsuit", "legal", "threat", "harass", "abuse", "hate",
        "racist", "sexist", "scam", "fraud", "fake", "lying",
        "death threat", "kill", "attack", "destroy", "suing",
    ]

    CRITICAL_KEYWORDS = ["doesn't work", "broken", "useless", "terrible", "worst", "awful"]

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify engagement sentiment.

        Returns: (sentiment, escalation_risk_score)
        sentiment: positive, neutral, critical, hostile, elevated
        escalation_risk_score: 0.0-1.0 (how likely this needs manual review)
        """
        text_lower = text.lower()

        # Check hostile first
        hostile_count = sum(1 for kw in self.HOSTILE_KEYWORDS if kw in text_lower)
        if hostile_count > 0:
            risk = min(hostile_count * 0.3, 1.0)
            return "hostile", risk

        # Check critical
        critical_count = sum(1 for kw in self.CRITICAL_KEYWORDS if kw in text_lower)
        if critical_count > 0:
            return "critical", 0.5

        # Check ALL CAPS (elevated tone)
        if text != text_lower and len(text) > 20:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.3:
                return "elevated", 0.4

        # Check positive
        positive_words = ["great", "love", "thanks", "helpful", "amazing", "brilliant", "excellent"]
        if any(w in text_lower for w in positive_words):
            return "positive", 0.0

        # Default to neutral
        return "neutral", 0.1


sentiment_analyzer = SentimentAnalyzer()
