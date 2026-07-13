from config import settings


def test_config_loads():
    """Test that config loads from .env"""
    assert settings.linkedin_access_token
    assert settings.openrouter_api_key
    assert settings.port == 8000


def test_config_validates_real_secrets():
    """Test that placeholder secrets are rejected"""
    assert "PLACEHOLDER" not in settings.linkedin_access_token
    assert "your_" not in settings.openrouter_api_key.lower()
