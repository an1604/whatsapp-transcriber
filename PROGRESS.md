# Project Progress

> This file is updated after each step is merged to `dev`. `main` is never touched directly.

---

## Build Order & Status

| # | Step | Branch | PR | Status | Tests added |
|---|------|--------|----|--------|-------------|
| 1 | Skeleton + config + DB models | `feature/step-1-skeleton` | #1 | ✅ Merged to dev | 160 |
| 2 | URL extractor + yt-dlp downloader | `feature/step-2-url-extractor-downloader` | #2 | ✅ Merged to dev | 161 |
| 3 | faster-whisper transcriber | `feature/step-3-transcriber` | #3 | ✅ Merged to dev | 48 |
| 4 | LiteLLM summarizer + prompt loader | `feature/step-4-summarizer` | #4 | ✅ Merged to dev | 51 |
| 5 | Pipeline orchestrator + dedup + job queue | `feature/step-5-pipeline` | #5 | ✅ Merged to dev | 45 |
| 6 | FastAPI REST API | `feature/step-6-web-ui` | #6 | ✅ Merged to dev | 49 |
| 7 | HTMX + Jinja2 frontend | `feature/step-7-htmx-ui` | #7 | ✅ Merged to dev | 59 |
| 8 | Docker Compose (app + Ollama) | `feature/step-8-docker` | #8 | ✅ Merged to dev | 70 |
| 9 | WhatsApp Node.js sidecar | `feature/step-9-whatsapp-sidecar` | #9 | ✅ Merged to dev | 109 (Py) + 47 (JS) |
| 10 | Polish (search, bulk delete, settings UI) | `feature/step-10-polish` | #10 | ✅ Merged to dev | 50 |

**Total tests on dev: 732 Python + 47 Node.js = 779 total (all passing) 🎉**

---

## What each step delivered

### Step 1 — Skeleton
- `src/core/config.py` — Pydantic `AppConfig` loader; raises `ConfigurationError` on any issue
- `src/core/database.py` — Async SQLAlchemy `Database` class; `VideoORM`, `JobORM`; status machine enforced
- `src/core/models.py` — `VideoStatus`, `Platform`, `JobType` enums; `VideoCreate/Read/Update`, `JobCreate/Read`
- `src/core/exceptions.py` — Full exception hierarchy (no fallbacks anywhere)

### Step 2 — URL extractor + downloader
- `src/downloaders/platforms/{youtube,instagram,tiktok,facebook}.py` — URL patterns + canonicalization
- `src/downloaders/registry.py` — `detect_platform`, `canonicalize_url`
- `src/scrapers/url_extractor.py` — `extract_urls` (batch, dedup by hash), `extract_single_url` (strict)
- `src/downloaders/ytdlp_downloader.py` — `YtDlpDownloader`; async thread-pool wrapper; raises `DownloadError`

### Step 3 — Transcriber
- `src/transcribers/base.py` — `AbstractTranscriber`, `TranscriptionResult`
- `src/transcribers/faster_whisper.py` — CPU mode, lazy model load, auto language detection

### Step 4 — Summarizer
- `src/summarizers/prompt_loader.py` — `load_prompt` + `render_prompt`; raises `ConfigurationError`
- `src/summarizers/litellm_summarizer.py` — `LiteLLMSummarizer`; raises `SummarizationError`

### Step 5 — Pipeline
- `src/pipeline/deduplication.py` — `DeduplicationService`; hash-based dedup
- `src/pipeline/job_queue.py` — Async priority `JobQueue`; error capture without worker crash
- `src/pipeline/orchestrator.py` — `PipelineOrchestrator`; download→transcribe→summarize with DB tracking

### Step 6 — REST API
- `src/web/app.py` — `create_app` factory; lifespan DB init
- `src/web/routes/videos.py` — `GET/api/videos`, `GET /api/videos/{id}`, `DELETE /api/videos/{id}/audio`
- `src/web/routes/jobs.py` — `GET /api/jobs/video/{id}`, `GET /api/jobs/{id}`
- `src/web/routes/dashboard.py` — `GET /api/dashboard` (aggregate counts)
- `src/web/routes/submit.py` — `POST /api/submit` (manual URL, dedup-aware, 422 on bad URL)

### Step 10 — Polish
- `src/core/database.py` — `search_videos()` (LIKE across url/transcript/summary), `bulk_mark_audio_deleted()`, `get_disk_stats()`
- `src/web/routes/videos.py` — `GET /api/videos/search`, `POST /api/videos/bulk-delete-audio`
- `src/web/routes/dashboard.py` — extended with `videos_with_audio` + `videos_audio_deleted` fields
- `src/web/routes/htmx.py` — HTMX search route + HTMX bulk-delete-audio route
- `src/web/routes/pages.py` — `GET /settings`
- Templates — search box, checkboxes, bulk-delete form, settings page, disk stats on dashboard

### Step 9 — WhatsApp Node.js Sidecar
- `whatsapp-sidecar/src/url_extractor.js` — regex URL extraction for all 4 platforms
- `whatsapp-sidecar/src/pipeline_client.js` — fetch-based HTTP client to POST URLs to Python API
- `whatsapp-sidecar/src/message_handler.js` — community ID filtering, URL extraction, submission
- `whatsapp-sidecar/src/index.js` — WhatsApp Web.js client + Express health/status API
- `src/whatsapp/sidecar_client.py` — async Python client (SidecarError / SidecarNotReadyError)
- docker-compose.yml updated with `whatsapp-sidecar` service + `wa_session` volume

### Step 8 — Docker Compose
- `Dockerfile` — multi-stage build: builder (pip wheels) + runtime (ffmpeg, app code)
- `docker-compose.yml` — `app` + `ollama` services, named volumes, healthchecks, restart policies
- `scripts/entrypoint.sh` — waits for Ollama readiness before starting uvicorn
- `config/config.docker.yaml` — Docker-targeted config (Ollama service name, /data volume paths)
- `src/core/config.py` — `load_config_with_env_overrides()` for 12-factor deployments
  Supported: DATABASE_URL, AUDIO_CACHE_DIR, OLLAMA_BASE_URL, WEB_HOST, WEB_PORT, WHISPER_MODEL_SIZE

### Step 7 — HTMX + Jinja2 Frontend
- `src/web/routes/pages.py` — 5 full-page routes (`/`, `/videos`, `/videos/{id}`, `/submit`, `/jobs`)
- `src/web/routes/htmx.py` — 8 HTMX partial endpoints (stats, video table w/ filter+pagination, delete audio, jobs, submit form)
- `src/web/templates/` — `base.html` dark-theme layout + 5 pages + 5 partials + 404
- Pagination Prev/Next, status badges, platform filters, auto-poll every 5–10 s via HTMX

---
