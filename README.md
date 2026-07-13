# LLM Gateway

Standalone microservice abstracting generation/embedding/reranking behind
a stable HTTP API. See `LLM_GATEWAY_PROJECT_PLAN.md` for the full design
and `LLM_GATEWAY_PROGRESS.md` for current status.

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit .env with your real Gemini key
```

Get a free Gemini API key at https://aistudio.google.com/apikey.

## Run

```bash
uvicorn app.main:app --reload
```

Docs at http://127.0.0.1:8000/docs (FastAPI auto-generates this from the
Pydantic schemas - open it, it's the fastest way to poke the API by hand).

## Try it

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-123" \
  -d '{"prompt": "Say hello in one short sentence."}'

curl http://127.0.0.1:8000/v1/health -H "X-API-Key: dev-secret-123"
```

## Adding a new generation backend (the whole point of the adapter pattern)

1. Create `app/backends/<provider>_generate.py`, implementing
   `GenerationBackend` (see `app/backends/base.py`) - a `generate()` and a
   `health_check()` method.
2. Add one line to `build_generation_registry()` in `app/registry.py`.
3. Add any new env vars (API key, model name) to `.env.example` and
   `app/config.py`.

Nothing else changes. If adding a backend ever requires touching
`main.py` or `router.py`, that's a signal the interface needs fixing, not
a one-off exception (Section 7.5 of the project plan).

## Known limitations (G0)

- No failover/retry logic yet - a rate-limited or failed call returns an
  error to the caller directly. Backoff + failover is Phase G2.
- Only the Gemini backend exists. Groq and local Qwen are Phase G1.
- SQLite log has no retention/rotation policy - fine at personal-project
  scale, explicitly deferred (see plan Section 5).
