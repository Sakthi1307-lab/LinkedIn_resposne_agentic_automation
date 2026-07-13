import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import yaml
from sqlalchemy.orm import Session

from src.linkedin_client import linkedin_client
from src.models import PersonProfile

logger = logging.getLogger(__name__)

# Below this confidence, a tier guess is not reliable enough to be treated as
# VIP-equivalent (B_external_vip/C_decision_maker). This module's job is to
# report confidence honestly, not to guess a person into a higher tier than
# the evidence supports — downstream reply generation trusts `tier` as-is, so
# the downgrade has to happen here.
MIN_CONFIDENCE_FOR_ELEVATED_TIER = 0.5


@dataclass
class PersonaContext:
    """
    Everything response_generator.py needs to know about the person engaging.
    """

    linkedin_urn: str
    name: str
    tier: str  # A_internal_leadership / B_external_vip / C_decision_maker / D_general
    relationship_to_us: str  # self, team, investor, partner, customer, external, unknown
    company: Optional[str] = None
    role_guess: Optional[str] = None
    confidence_score: float = 0.0  # 0.0-1.0
    is_vip: bool = False
    is_own_ceo: bool = False
    voice_note: Optional[str] = None


class PersonaResolver:
    """
    Identifies who is engaging with the company page.
    Uses a three-tier confidence pyramid: VIP registry (authoritative) ->
    cached profile (30-day TTL) -> live lookup + heuristic guess.
    """

    def __init__(self, db_session: Session):
        self.db = db_session
        self.vip_registry = self._load_vip_registry()
        logger.info(f"PersonaResolver initialized with {len(self.vip_registry)} VIPs")

    def _load_vip_registry(self) -> dict:
        """
        Load VIP registry from YAML file.

        data/vip_registry.yaml stores `vips:` as a dict keyed by an internal
        nickname (e.g. `harish_ceo:`), with each entry's own `linkedin_urn`
        field being the actual lookup key we need. Re-key by linkedin_urn.

        Returns dict keyed by LinkedIn URN. Falls back to an empty registry
        (never raises) if the file is missing or fails to parse — the agent
        should degrade to Tier 2/3 resolution, not go down entirely.
        """
        try:
            with open("data/vip_registry.yaml", "r") as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning("VIP registry file is empty, falling back to empty registry")
                return {}

            raw_vips = data.get("vips", {})
            if not isinstance(raw_vips, dict):
                logger.warning(
                    f"VIP registry 'vips' key has unexpected type {type(raw_vips).__name__} "
                    "(expected a mapping), falling back to empty registry"
                )
                return {}

            vips = {}
            for nickname, vip in raw_vips.items():
                if not isinstance(vip, dict):
                    logger.warning(f"Skipping malformed VIP entry '{nickname}' (not a mapping)")
                    continue
                urn = vip.get("linkedin_urn")
                if urn:
                    vips[urn] = vip
                else:
                    logger.warning(f"Skipping VIP entry '{nickname}': missing linkedin_urn")

            logger.info(f"Loaded {len(vips)} VIPs from registry")
            return vips

        except FileNotFoundError:
            logger.warning("VIP registry file not found at data/vip_registry.yaml, falling back to empty registry")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse VIP registry YAML: {e}, falling back to empty registry")
            return {}
        except Exception as e:
            logger.error(f"Failed to load VIP registry: {e}, falling back to empty registry", exc_info=True)
            return {}

    def resolve(self, linkedin_urn: str) -> PersonaContext:
        """
        Three-tier persona resolution.
        Returns PersonaContext with all info about this person.
        """

        # ═════════════════════════════════════════════════════════════════════
        # TIER 1: VIP REGISTRY (instant, authoritative — always wins)
        # ═════════════════════════════════════════════════════════════════════

        if linkedin_urn in self.vip_registry:
            vip = self.vip_registry[linkedin_urn]
            is_own_ceo = vip.get("relationship_to_us") == "self"

            logger.info(f"VIP found: {vip.get('name')} ({vip.get('relationship_to_us')})")

            return PersonaContext(
                linkedin_urn=linkedin_urn,
                name=vip.get("name", "Friend"),
                tier=vip.get("reply_tier", "D_general"),
                relationship_to_us=vip.get("relationship_to_us", "unknown"),
                company=vip.get("company"),
                role_guess=vip.get("role"),
                confidence_score=1.0,
                is_vip=True,
                is_own_ceo=is_own_ceo,
                voice_note=vip.get("voice_note"),
            )

        # ═════════════════════════════════════════════════════════════════════
        # TIER 2: CACHED PROFILE (30-day TTL)
        # ═════════════════════════════════════════════════════════════════════

        cached = self.db.query(PersonProfile).filter_by(linkedin_urn=linkedin_urn).first()
        if cached:
            days_old = (datetime.utcnow() - cached.last_profile_fetch).days
            if days_old < 30:
                logger.info(f"Using cached profile: {cached.name} (cached {days_old}d ago)")

                tier = self._cap_tier_by_confidence(cached.vip_tier or "D_general", cached.confidence_score)

                return PersonaContext(
                    linkedin_urn=linkedin_urn,
                    name=cached.name,
                    tier=tier,
                    relationship_to_us="unknown",
                    company=cached.company,
                    role_guess=cached.title,
                    confidence_score=cached.confidence_score,
                    is_vip=cached.is_vip,
                    is_own_ceo=False,
                )
            else:
                logger.info(f"Cached profile expired ({days_old}d), refreshing")

        # ═════════════════════════════════════════════════════════════════════
        # TIER 3: LIVE LOOKUP + HEURISTIC FALLBACK
        # ═════════════════════════════════════════════════════════════════════

        profile_data = linkedin_client.get_profile(linkedin_urn)

        if profile_data and profile_data.get("name"):
            raw_tier = self._guess_tier_from_title(profile_data.get("headline", ""))
            confidence = self._score_confidence(profile_data, raw_tier)
            tier = self._cap_tier_by_confidence(raw_tier, confidence)

            # Cache this profile. is_vip is intentionally never set True here —
            # a heuristic guess from a title keyword is not the same claim as
            # a curated VIP registry entry, and Tier 1 must remain the only
            # source of truth for is_vip.
            new_profile = PersonProfile(
                linkedin_urn=linkedin_urn,
                name=profile_data.get("name", "Friend"),
                company=profile_data.get("company"),
                title=profile_data.get("headline"),
                vip_tier=tier,
                confidence_score=confidence,
                last_profile_fetch=datetime.utcnow(),
            )
            self.db.merge(new_profile)
            self.db.commit()

            logger.info(f"Profile fetched and cached: {profile_data.get('name')} ({tier})")

            return PersonaContext(
                linkedin_urn=linkedin_urn,
                name=profile_data.get("name", "Friend"),
                tier=tier,
                relationship_to_us="unknown",
                company=profile_data.get("company"),
                role_guess=profile_data.get("headline"),
                confidence_score=confidence,
                is_vip=False,
                is_own_ceo=False,
            )

        # ═════════════════════════════════════════════════════════════════════
        # FALLBACK: We got nothing, default to generic
        # ═════════════════════════════════════════════════════════════════════

        logger.warning(f"No profile data available for {linkedin_urn}, defaulting to generic")

        return PersonaContext(
            linkedin_urn=linkedin_urn,
            name="Friend",
            tier="D_general",
            relationship_to_us="unknown",
            confidence_score=0.1,
            is_vip=False,
            is_own_ceo=False,
        )

    def _guess_tier_from_title(self, title: str) -> str:
        """
        Heuristic: scan the title for keywords to guess tier.

        Decision-maker keywords: VP, Director, Head, CRO, CMO, CTO, Chief
        External VIP keywords: Founder, CEO
        """
        if not title:
            return "D_general"

        title_lower = title.lower()

        # Check for founder/CEO/leadership
        external_vip_keywords = ["founder", "ceo", "cto", "cfo", "coo", "chief"]
        if any(kw in title_lower for kw in external_vip_keywords):
            return "B_external_vip"

        # Check for decision maker
        decision_maker_keywords = ["vp ", "director", "head of", "officer", "manager", "chief"]
        if any(kw in title_lower for kw in decision_maker_keywords):
            return "C_decision_maker"

        # Default to general
        return "D_general"

    def _score_confidence(self, profile_data: dict, tier: str) -> float:
        """
        Confidence score 0.0-1.0 based on how much data we have.

        Full profile (name + title + company) = high confidence
        Partial profile = medium
        """
        score = 0.3  # baseline

        if profile_data.get("name"):
            score += 0.25
        if profile_data.get("headline"):
            score += 0.3
        if profile_data.get("company"):
            score += 0.15

        return min(score, 1.0)

    def _cap_tier_by_confidence(self, tier: str, confidence: float) -> str:
        """
        A heuristic-derived tier of B_external_vip or C_decision_maker is a
        claim that this person deserves elevated handling. If confidence is
        below MIN_CONFIDENCE_FOR_ELEVATED_TIER, that claim isn't earned —
        downgrade to D_general rather than let a low-confidence guess pass
        as VIP-equivalent further down the pipeline.
        """
        if tier in ("B_external_vip", "C_decision_maker") and confidence < MIN_CONFIDENCE_FOR_ELEVATED_TIER:
            logger.info(
                f"Downgrading tier {tier} to D_general: confidence {confidence:.2f} "
                f"below threshold {MIN_CONFIDENCE_FOR_ELEVATED_TIER}"
            )
            return "D_general"
        return tier


# Test helper
if __name__ == "__main__":
    from src.models import SessionLocal

    db = SessionLocal()
    resolver = PersonaResolver(db)

    # Test with a fake URN
    test_persona = resolver.resolve("urn:li:person:test123")
    print(f"Test persona: {test_persona}")
