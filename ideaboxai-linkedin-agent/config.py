from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LinkedIn
    linkedin_access_token: str
    linkedin_organization_urn: str
    linkedin_webhook_verification_token: str

    # OpenRouter
    openrouter_api_key: str
    openrouter_model_high_stakes: str = "anthropic/claude-sonnet-4.6"
    openrouter_model_standard: str = "openai/gpt-4-turbo"
    openrouter_model_fast: str = "openai/gpt-3.5-turbo"

    # App
    debug: bool = False
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

    # Slack
    slack_webhook_url: Optional[str] = None
    slack_alert_channel: str = "#ideaboxai-engage-alerts"

    @field_validator("linkedin_access_token", "openrouter_api_key")
    @classmethod
    def not_placeholder(cls, v):
        if not v or "your_" in v or "PLACEHOLDER" in v or "xxxxxxx" in v:
            raise ValueError("You must set a real secret in .env")
        return v

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


settings = Settings()
