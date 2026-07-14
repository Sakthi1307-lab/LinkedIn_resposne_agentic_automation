import argparse
import json
import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from config import settings
from src.dedupe_and_rate_limit import DedupeAndRateLimit
from src.escalation import escalation_handler
from src.linkedin_client import linkedin_client
from src.llm_client import llm_client
from src.models import EngagementEvent, ReplyLog, SessionLocal
from src.persona_resolver import PersonaResolver
from src.response_generator import response_generator

# Setup logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="IdeaBoxAI Engage",
    version="1.0.0",
    description="Autonomous LinkedIn engagement agent",
)

scheduler = AsyncIOScheduler()

# ═══════════════════════════════════════════════════════════════════════════════
# WEBHOOK ENDPOINT — LinkedIn pushes engagement events here
# ═══════════════════════════════════════════════════════════════════════════════


@app.post("/webhook/linkedin")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """
    LinkedIn Community Management API sends engagement events here.

    We verify the signature, parse the event, and queue it for async
    processing. This handler does only cheap work (signature check, JSON
    parse) and returns immediately — actual reply generation/posting happens
    in the background task, never inline here, so LinkedIn always gets a
    fast response regardless of how long the pipeline takes.
    """
    body = await request.body()
    signature = request.headers.get("X-LinkedIn-Signature", "")

    # Verify signature
    if not linkedin_client.verify_webhook_signature(body, signature):
        logger.warning("Invalid webhook signature, rejecting")
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse payload
    try:
        event_data = json.loads(body)
    except json.JSONDecodeError:
        logger.error("Failed to parse webhook payload as JSON")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    logger.info(f"Webhook received: {event_data.get('type', 'unknown')}")

    # Queue async processing
    background_tasks.add_task(process_engagement, event_data)

    # Return 200 immediately so LinkedIn knows we got it
    return {"status": "received"}


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC ENGAGEMENT PROCESSOR — The full pipeline
# ═══════════════════════════════════════════════════════════════════════════════


async def process_engagement(event_data: dict):
    """
    Full pipeline for a single engagement:

    1. Parse event and store in database
    2. Resolve WHO is engaging (persona resolver)
    3. Check dedup & rate limit
    4. Check if escalation needed
    5. Generate reply
    6. Post reply
    7. Log everything
    """

    db = SessionLocal()
    # Bound before the try block so the except handler can always reference
    # them for logging/cleanup, even if something fails before they'd
    # otherwise be assigned (e.g. a malformed event_data whose "actor" key
    # isn't a dict).
    linkedin_urn = ""
    comment_urn = ""
    engagement = None

    try:
        # Parse event
        linkedin_urn = event_data.get("actor", {}).get("urn", "")
        comment_urn = event_data.get("urn", "")
        engagement_text = event_data.get("text", "")
        engagement_type = event_data.get("type", "comment")

        if not linkedin_urn or not comment_urn or not engagement_text:
            logger.warning(f"Incomplete event data: {event_data}")
            return

        logger.info(f"Processing engagement from {linkedin_urn[:40]}...")

        # Step 1: Store the raw engagement event
        engagement = EngagementEvent(
            linkedin_urn=linkedin_urn,
            linkedin_comment_urn=comment_urn,
            engagement_type=engagement_type,
            engagement_text=engagement_text,
            engagement_timestamp=datetime.utcnow(),
        )
        db.add(engagement)
        db.commit()
        logger.info(f"Engagement stored (ID={engagement.id})")

        # Step 2: Resolve persona
        resolver = PersonaResolver(db)
        persona = resolver.resolve(linkedin_urn)
        logger.info(f"Persona: {persona.name} ({persona.tier}, confidence={persona.confidence_score:.1%})")

        # Step 3: Check dedup & rate limit
        dedup = DedupeAndRateLimit(db, max_per_person_per_hour=settings.max_replies_per_person_per_hour)
        if not dedup.can_reply(linkedin_urn, comment_urn):
            logger.info("Blocked by dedup/rate limit")
            engagement.processed = True
            db.commit()
            return

        # Step 4: Check escalation
        should_escalate = escalation_handler.should_escalate(engagement_text, persona.tier)
        if should_escalate:
            engagement.escalated = True
            engagement.processed = True
            db.commit()
            logger.warning("Engagement escalated to manual review")
            return

        # Step 5: Generate reply
        reply = response_generator.generate(engagement_text, persona, engagement_type)
        logger.info(f"Reply generated: {reply[:80]}...")

        # Step 6: Log the reply before posting (for audit trail)
        model_used = llm_client.select_model(persona.tier)
        reply_log = ReplyLog(
            engagement_id=engagement.id,
            linkedin_urn=linkedin_urn,
            persona_tier=persona.tier,
            generated_reply=reply,
            model_used=model_used,
        )
        db.add(reply_log)
        db.commit()
        logger.info(f"Reply logged (ID={reply_log.id})")

        # Step 7: Post the reply (full autonomy)
        success = linkedin_client.post_reply(comment_urn, reply)

        if success:
            reply_log.posted_reply = reply
            reply_log.posted_at = datetime.utcnow()
            reply_log.success = True
            engagement.processed = True
            db.commit()
            logger.info("Reply posted successfully")
        else:
            logger.error(f"Failed to post reply (engagement_id={engagement.id}, comment_urn={comment_urn})")
            reply_log.success = False
            db.commit()

    except Exception as e:
        # Log with enough context to debug without re-running: which person,
        # which comment, and the raw event that triggered this.
        logger.error(
            f"Error processing engagement (linkedin_urn={linkedin_urn!r}, comment_urn={comment_urn!r}): {e}",
            exc_info=True,
        )
        # A failed commit (e.g. an IntegrityError from a duplicate
        # comment_urn, which is unique on EngagementEvent) leaves the session
        # in a failed-transaction state. Roll back before trying to touch it
        # again, and guard against `engagement` never having been assigned
        # or never having been successfully flushed (no id yet).
        db.rollback()
        if engagement is not None and engagement.id is not None:
            try:
                engagement.processed = False
                db.commit()
            except Exception as cleanup_error:
                logger.error(
                    f"Failed to mark engagement {engagement.id} as unprocessed after error: {cleanup_error}",
                    exc_info=True,
                )
                db.rollback()

    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# POLLING FALLBACK — LinkedIn webhook missed? We catch it here
# ═══════════════════════════════════════════════════════════════════════════════


async def poll_linkedin():
    """
    Polls LinkedIn for new comments every 2 minutes.
    Catches anything the webhook missed.
    """
    try:
        logger.info("Polling LinkedIn for new comments...")

        comments = linkedin_client.get_new_comments(limit=10)

        if not comments:
            logger.info("No new comments found")
            return

        logger.info(f"Found {len(comments)} new comments to process")

        db = SessionLocal()

        try:
            for comment in comments:
                # Check if we've already processed this comment
                comment_urn = comment.get("urn", "")

                existing = db.query(EngagementEvent).filter_by(linkedin_comment_urn=comment_urn).first()

                if existing:
                    logger.info(f"Comment already processed: {comment_urn}")
                else:
                    logger.info("New comment detected, queueing for processing")
                    await process_engagement(comment)
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Polling error: {e}", exc_info=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP & SHUTDOWN
# ═══════════════════════════════════════════════════════════════════════════════


@app.on_event("startup")
async def startup():
    """Initialize scheduler and logging."""
    logger.info("=" * 80)
    logger.info("IdeaBoxAI Engage starting up...")
    logger.info(f"  Database: {settings.database_url}")
    logger.info(f"  Organization URN: {settings.linkedin_organization_urn}")
    logger.info(f"  Debug mode: {settings.debug}")
    logger.info(f"  Auto-escalate hostile: {settings.auto_escalate_hostile_comments}")
    logger.info(f"  Max replies per person/hour: {settings.max_replies_per_person_per_hour}")
    logger.info("=" * 80)

    # Start polling scheduler
    scheduler.add_job(poll_linkedin, "interval", minutes=2, id="poll_linkedin")
    scheduler.start()
    logger.info("Polling scheduler started (every 2 minutes)")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("IdeaBoxAI Engage shutting down...")
    scheduler.shutdown()
    logger.info("Scheduler stopped")


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK & DEBUG ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/health")
async def health():
    """Simple health check."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "database": "connected" if settings.database_url else "unconfigured",
        "openrouter": "configured" if settings.openrouter_api_key else "unconfigured",
    }


@app.get("/debug/last-replies")
async def debug_last_replies(limit: int = 5):
    """
    DEBUG ENDPOINT: Show the last N generated replies.
    Only works if DEBUG=true in .env
    """
    if not settings.debug:
        raise HTTPException(status_code=403, detail="Debug mode disabled")

    db = SessionLocal()
    try:
        replies = db.query(ReplyLog).order_by(ReplyLog.id.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "engagement_id": r.engagement_id,
                "persona_tier": r.persona_tier,
                "generated_reply": r.generated_reply,
                "posted": r.posted_at is not None,
                "posted_at": r.posted_at.isoformat() if r.posted_at else None,
                "success": r.success,
                "model": r.model_used,
            }
            for r in replies
        ]
    finally:
        db.close()


@app.get("/debug/stats")
async def debug_stats():
    """DEBUG ENDPOINT: Show agent statistics."""
    if not settings.debug:
        raise HTTPException(status_code=403, detail="Debug mode disabled")

    db = SessionLocal()
    try:
        total_engagements = db.query(EngagementEvent).count()
        processed_engagements = db.query(EngagementEvent).filter_by(processed=True).count()
        escalated_engagements = db.query(EngagementEvent).filter_by(escalated=True).count()
        successful_replies = db.query(ReplyLog).filter_by(success=True).count()

        return {
            "total_engagements": total_engagements,
            "processed": processed_engagements,
            "escalated": escalated_engagements,
            "successful_replies": successful_replies,
            "uptime": "check logs",
        }
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_args():
    parser = argparse.ArgumentParser(description="IdeaBoxAI Engage")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.version:
        print("IdeaBoxAI Engage 1.0.0")
    else:
        import uvicorn

        logger.info(f"Starting server on port {settings.port}...")
        uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level=settings.log_level.lower())
