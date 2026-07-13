# IdeaBoxAI Engage

Autonomous LinkedIn engagement agent. Monitors comments on the IdeaBoxAI
company page, identifies who is engaging, and generates on-brand replies
using an OpenRouter-backed LLM pipeline.

## Setup

1. Create a virtualenv and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy the environment template and fill in real secrets:

   ```bash
   cp .env.example .env
   ```

   Required values:
   - `LINKEDIN_ACCESS_TOKEN` — Community Management API bearer token
   - `LINKEDIN_ORGANIZATION_URN` — the company page URN to monitor
   - `LINKEDIN_WEBHOOK_VERIFICATION_TOKEN` — shared secret for webhook signature checks
   - `OPENROUTER_API_KEY` — OpenRouter API key for LLM access

3. Verify configuration loads and the database is created:

   ```bash
   python -c "from config import settings; print(settings.database_url)"
   ```

4. Run the app:

   ```bash
   python main.py
   ```

   Visit `http://localhost:8000/health`.

5. Print the version without starting the server:

   ```bash
   python main.py --version
   ```

## Project layout

- `config.py` — loads and validates `.env` settings
- `src/models.py` — SQLAlchemy ORM models (person profiles, engagement
  events, reply log, VIP registry)
- `src/linkedin_client.py` — LinkedIn API integration (stub)
- `src/llm_client.py` — OpenRouter LLM client (stub)
- `src/persona_resolver.py` — identifies who is engaging (stub)
- `src/response_generator.py` — generates on-brand replies (stub)
- `src/dedupe_and_rate_limit.py` — prevents duplicate/over-frequent replies (stub)
- `src/escalation.py` — flags hostile/high-risk engagement for review (stub)
- `src/sentiment_analyzer.py` — classifies engagement sentiment (stub)
- `brand/` — brand voice rules and reply templates
- `data/vip_registry.yaml` — seed data for VIP contacts
- `tests/` — pytest suite

## Tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t ideaboxai-engage .
docker run --env-file .env -p 8000:8000 ideaboxai-engage
```
