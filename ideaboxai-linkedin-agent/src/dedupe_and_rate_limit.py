import logging
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from src.models import EngagementEvent, ReplyLog

logger = logging.getLogger(__name__)


class DedupeAndRateLimit:
    """
    Prevents double-replies and rate-limit spam.
    """

    def __init__(self, db_session: Session, max_per_person_per_hour: int = 3):
        self.db = db_session
        self.max_per_person_per_hour = max_per_person_per_hour

    def can_reply(self, linkedin_urn: str, comment_urn: str) -> bool:
        """
        Check if we can reply to this comment.

        Returns False if:
        - We've already posted a reply to this exact comment
        - This person has >= max_per_person_per_hour replies in the past hour
        """

        # Check dedup: have we already posted a reply to THIS comment?
        #
        # ReplyLog has no comment_urn column of its own — the comment URN
        # lives on EngagementEvent.linkedin_comment_urn, linked via
        # ReplyLog.engagement_id. Filtering only on ReplyLog.linkedin_urn
        # (as if that identified the comment) would match ANY reply ever
        # posted to this person, permanently blocking every future reply to
        # them after the first — which defeats rate limiting too, since the
        # dedup check would always fire first. Join through EngagementEvent
        # so dedup is scoped to the actual comment.
        existing = (
            self.db.query(ReplyLog)
            .join(EngagementEvent, ReplyLog.engagement_id == EngagementEvent.id)
            .filter(
                EngagementEvent.linkedin_comment_urn == comment_urn,
                ReplyLog.posted_at.isnot(None),  # Only count actually posted replies
            )
            .first()
        )

        if existing:
            logger.warning(f"Duplicate reply attempt for {comment_urn}, skipping")
            return False

        # Check rate limit: how many replies has this person received in the past hour?
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_replies = (
            self.db.query(ReplyLog)
            .filter(
                ReplyLog.linkedin_urn == linkedin_urn,
                ReplyLog.posted_at > one_hour_ago,
                ReplyLog.posted_at.isnot(None),
            )
            .count()
        )

        if recent_replies >= self.max_per_person_per_hour:
            logger.warning(f"Rate limit: {linkedin_urn} has {recent_replies} replies in past hour")
            return False

        return True
