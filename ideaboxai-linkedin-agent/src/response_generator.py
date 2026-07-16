import logging
import re

import yaml

from src.llm_client import llm_client
from src.persona_resolver import PersonaContext

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates on-brand LinkedIn replies for different engagement types and personas.
    """

    # This agent posts with zero human review, so these patterns exist to
    # catch anything a reply shouldn't say regardless of how good it reads:
    # guarantees/promises of outcomes, pricing/contract commitments, unverified
    # certification claims, and regulated-advice framing (legal/medical/tax).
    # Checked in _legal_lint() on every generated reply before it can post.
    LEGAL_RISK_PATTERNS = [
        r"\bguarantee[ds]?\b",
        r"\bpromise[ds]?\b",
        r"\b100%\s*(accurate|guaranteed|effective|safe|secure)\b",
        r"\brisk[- ]free\b",
        r"\bno risk\b",
        r"\bnever fails?\b",
        r"\balways works?\b",
        r"\bmoney[- ]back\b",
        r"\brefund\b",
        r"\bfree forever\b",
        r"\bcertified\b",
        r"\bcompliant with\b",
        r"\blegal advice\b",
        r"\bmedical advice\b",
        r"\b(financial|tax) advice\b",
    ]

    def __init__(self):
        self.voice_rules = self._load_voice_rules()
        self.templates = self._load_templates()

    def _load_voice_rules(self) -> dict:
        """
        Load brand voice rules from YAML. Returns {} on any failure — every
        consumer of self.voice_rules below uses .get() with a safe default,
        so an empty dict degrades generation (no banned-word/pattern
        guidance in the prompt) rather than crashing the pipeline.
        """
        try:
            with open("brand/voice_rules.yaml", "r") as f:
                data = yaml.safe_load(f)
            return data or {}
        except Exception as e:
            logger.error(f"Failed to load voice rules: {e}, continuing with no voice rules")
            return {}

    def _load_templates(self) -> dict:
        """Load reply templates from YAML. Returns {} on any failure."""
        try:
            with open("brand/response_templates.yaml", "r") as f:
                data = yaml.safe_load(f)
            return data or {}
        except Exception as e:
            logger.error(f"Failed to load templates: {e}, continuing with no templates")
            return {}

    def generate(self, engagement_text: str, persona: PersonaContext, engagement_type: str = "comment") -> str:
        """
        Generate a reply to an engagement.

        Args:
            engagement_text: The comment/question/engagement
            persona: PersonaContext identifying who is engaging
            engagement_type: "comment", "reaction", "mention", "dm" — changes
                the CONTEXT framing in the prompt (e.g. a mention isn't a
                direct question, a dm-sourced reply must stay safe to post
                publicly). main.py's pipeline only posts for "comment" and
                "mention"; other types are blocked before they reach
                post_reply(), but this method still frames them correctly
                since it's also reachable from /debug/preview-reply.

        Returns:
            Generated reply text (ready to post). Always on-brand, legally
            safe, and matched to what was actually said — falls back to
            _fallback_reply() if generation fails or the output doesn't
            pass _voice_lint()/_legal_lint(). There is no path that returns
            an unlinted reply.
        """

        # Step 1: Classify what was actually said (content) — independent
        # of engagement_type, which classifies where it came from.
        content_type = self._detect_content_type(engagement_text)

        # Step 2: Build system prompt with brand voice + persona + engagement context
        system_prompt = self._build_system_prompt(persona, content_type, engagement_type)

        # Step 3: Build user message with context
        user_message = self._build_user_message(engagement_text, persona, content_type)

        # Step 4: Select model based on persona tier (routing lives in llm_client)
        model = llm_client.select_model(persona.tier)

        # Step 5: Call the LLM — always through llm_client, never the SDK directly
        try:
            reply, tokens = llm_client.generate_reply(
                system_prompt=system_prompt,
                user_message=user_message,
                model=model,
                max_tokens=150,
                temperature=0.7,
            )

            logger.info(f"Generated reply ({persona.tier}, {content_type}): {reply[:80]}...")

            # Step 6: Voice + legal lint — the only quality gates; no bypass path
            if not self._voice_lint(reply):
                logger.warning("Reply failed voice lint, falling back to safe reply")
                return self._fallback_reply(persona, engagement_text, content_type)

            if not self._legal_lint(reply):
                logger.warning("Reply failed legal/compliance lint, falling back to safe reply")
                return self._fallback_reply(persona, engagement_text, content_type)

            return reply

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return self._fallback_reply(persona, engagement_text, content_type)

    def _detect_content_type(self, engagement_text: str) -> str:
        """
        Classify what the person actually said, based on text patterns.
        Returns one of: critical_feedback, substantive_question,
        acknowledgment_praise, neutral_comment.

        Order matters: a critical remark that also contains a question mark
        ("this is broken, why?") should be handled as feedback first, and
        praise keywords are checked before falling through — anything with
        no signal in either direction is neutral_comment, not manufactured
        praise. Treating "cool" or an off-topic remark as praise would make
        the LLM invent enthusiasm the commenter never expressed.
        """
        text_lower = engagement_text.lower()

        # Detect critical feedback
        critical_keywords = ["doesn't work", "broken", "problem", "issue", "fail", "bad", "hate", "disappointed", "worst"]
        if any(kw in text_lower for kw in critical_keywords):
            return "critical_feedback"

        # Detect question
        if "?" in engagement_text:
            return "substantive_question"

        # Detect praise
        praise_keywords = ["great", "love", "amazing", "excellent", "brilliant", "thanks", "awesome", "impressive"]
        if any(kw in text_lower for kw in praise_keywords):
            return "acknowledgment_praise"

        # No critical/question/praise signal — a neutral or off-topic
        # remark. Don't force it into acknowledgment_praise.
        return "neutral_comment"

    def _build_system_prompt(self, persona: PersonaContext, content_type: str, engagement_type: str = "comment") -> str:
        """
        Build the system prompt with brand voice + persona-specific tone +
        engagement-source context + non-negotiable legal/compliance rules.
        """

        # Base brand voice
        base_prompt = f"""
You are replying on behalf of IdeaBoxAI LinkedIn Company Page.

BRAND IDENTITY:
Company: IdeaBoxAI
Tagline: "Empower Your Ideas With AI"
Core belief: "Generic AI gives everyone the same answer. That is the problem."

BRAND VOICE RULES:
- Confident but never arrogant
- Direct but never cold
- Challenging but never aggressive
- Short sentences over long ones
- Active voice always
- Real numbers over vague claims
- End on conviction, not a question (unless driving engagement)

BANNED WORDS (NEVER USE):
{', '.join(self.voice_rules.get('banned_words', []))}

SENTENCE PATTERNS THAT WORK:
{chr(10).join(['- ' + p for p in self.voice_rules.get('sentence_patterns', [])])}

REPLY CONSTRAINTS:
- Max 500 characters
- Target: 2-3 short sentences or one paragraph
- Use the commenter's actual first name: {persona.name}
- Never use placeholder text like "[NAME]"
- Never say "Thanks for the comment!" or "Great question!"
- One idea per sentence
- Max 1 emoji (usually 0 for comment replies)
- Be specific to what they said — never generic affirmation
"""

        # Persona-specific tone
        if persona.is_own_ceo:
            tone = f"\nTONE: You are replying AS the company to our own CEO/founder ({persona.name}). Collegial, warm, peer-to-peer."
        elif persona.tier == "B_external_vip":
            tone = f"\nTONE: {persona.name} is from {persona.company} (founder/external VIP). Peer-to-peer tone. Acknowledge their specific point substantively first. No pitch unless they asked directly."
            if persona.voice_note:
                tone += f"\nNOTE: {persona.voice_note}"
        elif persona.tier == "A_internal_leadership":
            tone = "\nTONE: This is someone from our team. Warm, collegial, collaborative."
        elif persona.tier == "C_decision_maker":
            tone = "\nTONE: This is a decision-maker (VP/Head/Director/CRO). On-brand, helpful. One soft CTA only if it lands naturally. No hard selling."
        else:
            tone = "\nTONE: This is a general audience member. Warm, on-brand, brief. No CTA unless they asked a direct question."

        # Content-type specific guidance — what they actually said
        if content_type == "critical_feedback":
            content_guidance = "\nREPLY TYPE: CRITICAL FEEDBACK\nNever defensive. Never dismiss. Acknowledge their specific critique. Respond with substance or an honest 'fair point' where warranted."
        elif content_type == "substantive_question":
            content_guidance = "\nREPLY TYPE: SUBSTANTIVE QUESTION\nAnswer their question directly. One follow-up insight if relevant. Optional soft CTA only if it lands naturally."
        elif content_type == "acknowledgment_praise":
            content_guidance = "\nREPLY TYPE: ACKNOWLEDGMENT/PRAISE\nSpecific mention of what they said (no 'thanks for the feedback'). One insight or affirming fact."
        else:  # neutral_comment
            content_guidance = "\nREPLY TYPE: NEUTRAL/GENERAL COMMENT\nThey didn't express praise, criticism, or a question — don't manufacture enthusiasm they didn't show. Acknowledge specifically what they said and add one genuine, concrete point."

        # Engagement-source context — where this came from changes what a
        # sensible reply looks like, independent of what was said.
        if engagement_type == "mention":
            source_context = "\nCONTEXT: They mentioned/tagged IdeaBoxAI in their own post or comment — this is not a direct question to us. Acknowledge the specific mention and add value; don't assume they're addressing us."
        elif engagement_type == "reaction":
            source_context = "\nCONTEXT: This is a reaction (like/celebrate/etc.), not a written comment. Keep any acknowledgment minimal and specific — don't invent a conversation that didn't happen."
        elif engagement_type == "dm":
            source_context = "\nCONTEXT: This originated as a private message. This text may end up posted publicly, so write nothing that assumes privacy — keep it appropriate for a public audience."
        else:  # comment (default)
            source_context = "\nCONTEXT: This is a direct comment on our own company page post — a normal public reply thread."

        legal_block = """
LEGAL & COMPLIANCE (never break these — this reply posts automatically with no human review):
- Never guarantee, promise, or imply a specific outcome, result, ROI, or performance number for the reader (no "will increase", "guaranteed", "always works", "100%", "risk-free").
- Never state pricing, discounts, refunds, or contract terms that aren't already public.
- Never claim a certification, compliance standard, or partnership (e.g. SOC2, HIPAA, GDPR) unless it's a verified public fact — if unsure, don't mention it.
- Never give legal, medical, tax, or financial advice, even if directly asked — acknowledge the question and redirect to a real conversation instead of answering the substance.
- Prefer "built to", "designed for", "can help with" over "will", "guarantees", "always".
"""

        return base_prompt + tone + source_context + content_guidance + legal_block

    def _build_user_message(self, engagement_text: str, persona: PersonaContext, content_type: str) -> str:
        """
        Build the user message that gives the LLM context.
        """
        return f"""
Engagement from {persona.name} ({persona.tier}, confidence={persona.confidence_score:.1%}):

"{engagement_text}"

Generate a natural-sounding, human reply that actually responds to what they
said — not a generic template. Remember:
- Use {persona.name}'s actual first name in the reply
- Be specific to what they said
- Stay under 500 characters
- No placeholder text
- No banned words
- No guarantees, promises, pricing, or compliance claims
- Max 1 emoji (probably 0)
"""

    def _voice_lint(self, reply: str) -> bool:
        """
        Quality gate: does this reply pass brand voice checks?
        Returns True if it passes, False if it should be regenerated.

        Paired with _legal_lint() as the only two quality gates in the
        pipeline — generate() has no path that returns a reply that failed
        either one.
        """

        reply_lower = reply.lower()

        # Check for banned words. Entries like "Leverage (as verb)" carry a
        # parenthetical annotation for humans reading the YAML — matching
        # that literal substring against LLM output would never fire, since
        # the model won't write "(as verb)". Strip the annotation so the
        # actual word is what gets checked.
        for banned in self.voice_rules.get("banned_words", []):
            banned_word = re.sub(r"\s*\(.*?\)\s*$", "", banned).strip().lower()
            if banned_word and banned_word in reply_lower:
                logger.warning(f"Reply contains banned word: '{banned}'")
                return False

        # Check for filler phrases
        filler_phrases = [
            "thanks for the comment",
            "great question",
            "thanks for sharing",
            "thanks for the feedback",
            "how can we help",
            "feel free to reach out",
        ]
        for filler in filler_phrases:
            if filler in reply_lower:
                logger.warning(f"Reply contains filler phrase: '{filler}'")
                return False

        # Check length
        if len(reply) > 500:
            logger.warning(f"Reply exceeds 500 chars: {len(reply)}")
            return False

        # Check emoji count (rough proxy)
        emoji_count = sum(1 for c in reply if ord(c) > 127)
        if emoji_count > 1:
            logger.warning(f"Reply has too many emoji: {emoji_count}")
            return False

        return True

    def _legal_lint(self, reply: str) -> bool:
        """
        Compliance gate: this agent posts with zero human review, so a reply
        must never ship a guarantee, refund/pricing commitment, unverified
        certification claim, or regulated-advice statement (legal/medical/
        financial/tax). Returns True if the reply is clear of those risk
        patterns, False if it should fall back to a safe canned reply.
        """
        reply_lower = reply.lower()
        for pattern in self.LEGAL_RISK_PATTERNS:
            if re.search(pattern, reply_lower):
                logger.warning(f"Reply failed legal lint: matched pattern '{pattern}'")
                return False
        return True

    def _fallback_reply(self, persona: PersonaContext, engagement_text: str, content_type: str = "neutral_comment") -> str:
        """
        Fallback if generation or lint fails.
        Still on-brand, still safe, and matched to content_type so a
        complaint doesn't get the same chipper one-liner as a compliment.
        """
        if content_type == "critical_feedback":
            return f"{persona.name}, hearing this — looking into it directly."
        if content_type == "substantive_question":
            return f"{persona.name}, good question — following up directly so you get a real answer."
        if len(engagement_text) < 50:
            return f"Thanks for the signal, {persona.name}. 🎯"
        else:
            return f"{persona.name}, we're on this. More soon."


# Initialize singleton
response_generator = ResponseGenerator()
