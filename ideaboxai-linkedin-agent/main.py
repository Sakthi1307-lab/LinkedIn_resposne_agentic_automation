import argparse
import json
import logging
import sys
from datetime import datetime

if "--version" in sys.argv[1:]:
    print("IdeaBoxAI Engage 1.0.0")
    raise SystemExit(0)

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from config import settings
from src.dedupe_and_rate_limit import DedupeAndRateLimit
from src.linkedin_client import linkedin_client
from src.llm_client import llm_client
from src.models import EngagementEvent, ReplyLog, SessionLocal
from src.persona_resolver import PersonaContext, PersonaResolver
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

# linkedin_client.post_reply() only knows how to post a public comment reply
# via the Community Management API's /mmComments endpoint — there is no DM
# path (send_dm() is an explicit unsupported stub) and no separate "reaction"
# posting semantics. Auto-posting a dm- or reaction-sourced payload through
# post_reply() would either post private conversation content publicly or
# attach a reply to a URN that was never a real comment thread. Only these
# two engagement types are safe to carry all the way to a public post.
SUPPORTED_PUBLIC_REPLY_TYPES = {"comment", "mention"}

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
    local_test_header = request.headers.get("X-Local-Test", "false").lower() == "true"

    # Verify signature
    if settings.local_test_mode and local_test_header:
        logger.info("Local test mode enabled; skipping webhook signature verification")
    elif not linkedin_client.verify_webhook_signature(body, signature):
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
    4. Generate reply
    5. Log the reply (audit trail)
    6. Post reply

    FULL AUTONOMY: every reply that passes dedup/rate-limit posts
    immediately. There is no approval step and no escalation queue — the
    only judgment call this pipeline makes is WHAT to say and HOW (handled
    inside response_generator, based on persona), never WHETHER to reply.
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

        if engagement_type not in SUPPORTED_PUBLIC_REPLY_TYPES:
            logger.warning(
                f"engagement_type={engagement_type!r} is not eligible for automatic public "
                f"reply (comment_urn={comment_urn[:50]}...); this client can only post public "
                f"comment replies, and doing so with dm/reaction-sourced content would leak a "
                f"private exchange or post to an invalid thread. Skipping without replying."
            )
            engagement.processed = True
            db.commit()
            return

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

        # Step 4: Generate reply — no escalation gate. response_generator
        # already builds the reply's tone/content around persona.tier
        # (VIP, decision-maker, general, etc.); that persona-awareness is
        # the only per-person judgment this pipeline applies.
        reply = response_generator.generate(engagement_text, persona, engagement_type)
        logger.info(f"Reply generated: {reply[:80]}...")

        # Step 5: Log the reply before posting (for audit trail)
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

        # Step 6: Post the reply immediately — full autonomy, no approval step
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
    logger.info(f"  Local test mode: {settings.local_test_mode}")
    logger.info("  Mode: FULL AUTONOMY (no approval step, no escalation queue)")
    logger.info(f"  Max replies per person/hour: {settings.max_replies_per_person_per_hour}")
    logger.info("=" * 80)

    if settings.local_test_mode:
        logger.info("Local test mode enabled; polling scheduler disabled")
    else:
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
        "local_test_mode": settings.local_test_mode,
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


@app.post("/debug/test-engagement")
async def debug_test_engagement(payload: dict, background_tasks: BackgroundTasks):
    """
    DEBUG ENDPOINT: Inject a synthetic engagement payload directly into the
    pipeline.

    This is intended for local personal-account testing. It skips LinkedIn
    signature checks and only runs when both DEBUG and LOCAL_TEST_MODE are
    enabled.
    """
    if not settings.debug or not settings.local_test_mode:
        raise HTTPException(status_code=403, detail="Local test mode disabled")

    background_tasks.add_task(process_engagement, payload)
    return {
        "status": "queued",
        "mode": "local-test",
    }


@app.post("/debug/preview-reply")
async def debug_preview_reply(payload: dict):
    """
    DEBUG ENDPOINT: Generate and return a reply preview for a personal-account
    comment or message.

    Use this when you want to see how the agent would respond without needing
    LinkedIn org permissions, webhook signatures, or posting access.
    """
    if not settings.debug or not settings.local_test_mode:
        raise HTTPException(status_code=403, detail="Local test mode disabled")

    engagement_text = payload.get("text", "")
    if not engagement_text:
        raise HTTPException(status_code=400, detail="Missing text")

    engagement_type = payload.get("type", "comment")
    persona = PersonaContext(
        linkedin_urn=payload.get("actor", {}).get("urn", "urn:li:person:test"),
        name=payload.get("name", "Friend"),
        tier=payload.get("tier", "D_general"),
        relationship_to_us=payload.get("relationship_to_us", "unknown"),
        company=payload.get("company"),
        role_guess=payload.get("role_guess"),
        confidence_score=float(payload.get("confidence_score", 0.5)),
        is_vip=bool(payload.get("is_vip", False)),
        is_own_ceo=bool(payload.get("is_own_ceo", False)),
        voice_note=payload.get("voice_note"),
    )

    reply = response_generator.generate(engagement_text, persona, engagement_type)

    return {
        "mode": "preview",
        "persona": {
            "name": persona.name,
            "tier": persona.tier,
            "confidence_score": persona.confidence_score,
            "relationship_to_us": persona.relationship_to_us,
            "company": persona.company,
        },
        "engagement_type": engagement_type,
        "engagement_text": engagement_text,
        "reply": reply,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_args():
    parser = argparse.ArgumentParser(description="IdeaBoxAI Engage")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check LinkedIn + OpenRouter connectivity with the current .env and exit "
        "(no server started, nothing posted)",
    )
    return parser.parse_args()


def _run_preflight() -> bool:
    """
    Verify the current .env can actually reach both external dependencies
    before anything goes live. Returns True iff both checks pass (or are
    intentionally skipped) — the caller decides the process exit code.

    This exists because both failure modes it catches are silent otherwise:
    a missing/rejected OPENROUTER_API_KEY makes llm_client fall back to a
    static canned reply with no error at all (see LLMClient.check_connectivity's
    docstring), and a token that's valid but lacks org-page scope 403s only
    when poll_linkedin() happens to run, which get_new_comments() swallows
    into an empty list rather than surfacing.
    """
    print("Running pre-flight checks against the current .env...\n")

    llm_result = llm_client.check_connectivity()
    print(f"[{'OK' if llm_result['ok'] else 'FAIL'}] OpenRouter (LLM): {llm_result['detail']}")

    if settings.local_test_mode:
        print("[SKIP] LinkedIn organization connectivity — LOCAL_TEST_MODE is enabled.")
        linkedin_ok = True
    else:
        li_result = linkedin_client.check_connectivity()
        print(f"[{'OK' if li_result['ok'] else 'FAIL'}] LinkedIn (org comments): {li_result['detail']}")
        linkedin_ok = li_result["ok"]

    print()
    all_ok = llm_result["ok"] and linkedin_ok
    print("All checks passed — ready to run live." if all_ok else "Pre-flight failed — fix the above before running live.")
    return all_ok


if __name__ == "__main__":
    args = _parse_args()
    if args.version:
        print("IdeaBoxAI Engage 1.0.0")
    elif args.preflight:
        sys.exit(0 if _run_preflight() else 1)
    else:
        import uvicorn

        logger.info(f"Starting server on port {settings.port}...")
        uvicorn.run(app, host="0.0.0.0", port=settings.port, log_level=settings.log_level.lower())
