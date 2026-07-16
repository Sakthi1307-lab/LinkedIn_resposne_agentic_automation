# IdeaBoxAI Engage

Autonomous LinkedIn engagement agent. Monitors comments on the IdeaBoxAI
company page, identifies who is engaging, and generates on-brand replies
using an OpenRouter-backed LLM pipeline.

**Full autonomy**: every reply that passes dedup/rate-limiting posts
immediately. There is no approval step and no escalation queue. The only
judgment call the agent makes is *what* to say and *how*, based on who's
talking to us (persona tier) — never *whether* to reply.

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

   For local personal-account testing, set `LOCAL_TEST_MODE=true`. In that mode the
   webhook accepts requests with the `X-Local-Test: true` header and the polling job
   stays disabled so the app does not call org-only LinkedIn endpoints.

   You can also POST a payload directly to `POST /debug/test-engagement` when both
   `DEBUG=true` and `LOCAL_TEST_MODE=true` are set.

   If you only want to see the reply the agent would write for a personal-account
   comment, use `POST /debug/preview-reply`. That endpoint returns the generated
   reply immediately and does not try to post to LinkedIn.

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
- `src/escalation.py` — sentiment-based escalation logic; not wired into
  `main.py`'s pipeline (full autonomy mode has no approval step), kept
  available if you want to re-enable a manual-review gate later
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
