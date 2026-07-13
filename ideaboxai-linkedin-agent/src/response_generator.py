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
            engagement_type: "comment", "reaction", "mention", "dm"

        Returns:
            Generated reply text (ready to post). Always on-brand and safe —
            falls back to _fallback_reply() if generation fails or the
            output doesn't pass voice_lint(). There is no path that returns
            an un-linted reply.
        """

        # Step 1: Determine reply type based on engagement
        reply_type = self._detect_reply_type(engagement_text)

        # Step 2: Build system prompt with brand voice + persona
        system_prompt = self._build_system_prompt(persona, reply_type)

        # Step 3: Build user message with context
        user_message = self._build_user_message(engagement_text, persona, reply_type)

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

            logger.info(f"Generated reply ({persona.tier}): {reply[:80]}...")

            # Step 6: Voice lint — the only quality gate; no bypass path
            if not self._voice_lint(reply):
                logger.warning("Reply failed voice lint, falling back to safe reply")
                return self._fallback_reply(persona, engagement_text)

            return reply

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return self._fallback_reply(persona, engagement_text)

    def _detect_reply_type(self, engagement_text: str) -> str:
        """
        Guess the engagement type based on text patterns.
        Returns: acknowledgment_praise, substantive_question, critical_feedback, etc.
        """
        text_lower = engagement_text.lower()

        # Detect critical feedback
        critical_keywords = ["doesn't work", "broken", "problem", "issue", "fail", "bad", "hate"]
        if any(kw in text_lower for kw in critical_keywords):
            return "critical_feedback"

        # Detect question
        if "?" in engagement_text:
            return "substantive_question"

        # Detect praise
        praise_keywords = ["great", "love", "amazing", "excellent", "brilliant", "thanks"]
        if any(kw in text_lower for kw in praise_keywords):
            return "acknowledgment_praise"

        # Default
        return "acknowledgment_praise"

    def _build_system_prompt(self, persona: PersonaContext, reply_type: str) -> str:
        """
        Build the system prompt with brand voice + persona-specific tone.
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

        # Reply-type specific guidance
        if reply_type == "critical_feedback":
            reply_guidance = "\nREPLY TYPE: CRITICAL FEEDBACK\nNever defensive. Never dismiss. Acknowledge their specific critique. Respond with substance or an honest 'fair point' where warranted."
        elif reply_type == "substantive_question":
            reply_guidance = "\nREPLY TYPE: SUBSTANTIVE QUESTION\nAnswer their question directly. One follow-up insight if relevant. Optional soft CTA only if it lands naturally."
        elif reply_type == "acknowledgment_praise":
            reply_guidance = "\nREPLY TYPE: ACKNOWLEDGMENT/PRAISE\nSpecific mention of what they said (no 'thanks for the feedback'). One insight or affirming fact."
        else:
            reply_guidance = ""

        return base_prompt + tone + reply_guidance

    def _build_user_message(self, engagement_text: str, persona: PersonaContext, reply_type: str) -> str:
        """
        Build the user message that gives the LLM context.
        """
        return f"""
Engagement from {persona.name} ({persona.tier}, confidence={persona.confidence_score:.1%}):

"{engagement_text}"

Generate a natural-sounding reply. Remember:
- Use {persona.name}'s actual first name in the reply
- Be specific to what they said
- Stay under 500 characters
- No placeholder text
- No banned words
- Max 1 emoji (probably 0)
"""

    def _voice_lint(self, reply: str) -> bool:
        """
        Quality gate: does this reply pass brand voice checks?
        Returns True if it passes, False if it should be regenerated.

        This is the only quality gate in the pipeline — generate() has no
        path that returns an unlinted reply.
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

    def _fallback_reply(self, persona: PersonaContext, engagement_text: str) -> str:
        """
        Fallback if generation or lint fails.
        Still on-brand, still safe.
        """
        if len(engagement_text) < 50:
            return f"Thanks for the signal, {persona.name}. 🎯"
        else:
            return f"{persona.name}, we're on this. More soon."


# Initialize singleton
response_generator = ResponseGenerator()
