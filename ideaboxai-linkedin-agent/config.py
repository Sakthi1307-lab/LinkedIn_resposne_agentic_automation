from typing import Optional

from pydantic import ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LinkedIn
    linkedin_access_token: Optional[str] = None
    linkedin_organization_urn: Optional[str] = None
    linkedin_webhook_verification_token: Optional[str] = None

    # OpenRouter
    openrouter_api_key: Optional[str] = None
    openrouter_model_high_stakes: str = "anthropic/claude-sonnet-4.6"
    openrouter_model_standard: str = "openai/gpt-4-turbo"
    openrouter_model_fast: str = "openai/gpt-3.5-turbo"

    # App
    debug: bool = False
    local_test_mode: bool = False
    port: int = 8000
    database_url: str = "sqlite:///./ideaboxai_engage.db"
    log_level: str = "INFO"

    # Safety
    # Not consumed by main.py's pipeline — the agent runs full-autonomy
    # (no approval step, no escalation queue). Only affects src/escalation.py
    # if you re-wire it back into the pipeline yourself.
    auto_escalate_hostile_comments: bool = False
    max_replies_per_person_per_hour: int = 3
    reply_dedup_window_hours: int = 24

    # Known facts the response generator can hand the LLM to answer with,
    # instead of it having to invent a plausible-sounding but fake detail
    # (e.g. "here's the link" with no real link to give). Optional — when
    # unset, replies that need one of these facts fall back to an honest
    # concrete next step instead of a link/number that doesn't exist.
    demo_booking_url: Optional[str] = None

    # Slack
    slack_webhook_url: Optional[str] = None
    slack_alert_channel: str = "#ideaboxai-engage-alerts"

    @field_validator("linkedin_access_token", "openrouter_api_key")
    @classmethod
    def not_placeholder(cls, v):
        if v and ("your_" in v or "PLACEHOLDER" in v or "xxxxxxx" in v):
            raise ValueError("You must set a real secret in .env")
        return v

    @model_validator(mode="after")
    def require_real_secrets_when_not_local(self):
        if self.local_test_mode:
            return self

        missing = []
        if not self.linkedin_access_token:
            missing.append("LINKEDIN_ACCESS_TOKEN")
        if not self.linkedin_organization_urn:
            missing.append("LINKEDIN_ORGANIZATION_URN")
        if not self.linkedin_webhook_verification_token:
            missing.append("LINKEDIN_WEBHOOK_VERIFICATION_TOKEN")
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")

        if missing:
            raise ValueError("Missing required configuration: " + ", ".join(missing))

        return self

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


try:
    settings = Settings()
except ValidationError as exc:
    missing_fields = []
    for error in exc.errors():
        if error.get("type") == "missing":
            location = error.get("loc", ())
            if location:
                missing_fields.append(str(location[0]).upper())
        elif error.get("type") == "value_error":
            message = error.get("msg", "")
            if "Missing required configuration:" in message:
                missing_fields.extend(message.split(":", 1)[1].strip().split(", "))

    if missing_fields:
        missing_list = ", ".join(missing_fields)
        raise RuntimeError(
            "Missing required configuration. Copy .env.example to .env and set: "
            f"{missing_list}."
        ) from None

    raise RuntimeError(
        "Invalid configuration in .env. Check the required LinkedIn and OpenRouter values."
    ) from None
