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
| 7 | HTMX + Jinja2 frontend | `feature/step-7-htmx-ui` | — | 🔄 In progress | — |
| 8 | Docker Compose (app + Ollama) | `feature/step-8-docker` | — | ⬜ Pending | — |
| 9 | WhatsApp Node.js sidecar | `feature/step-9-whatsapp-sidecar` | — | ⬜ Pending | — |
| 10 | Polish (search, bulk delete, settings UI) | `feature/step-10-polish` | — | ⬜ Pending | — |

**Total tests on dev: 514 (all passing)**

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

---

## Upcoming: Step 7 — HTMX + Jinja2 Frontend

**Goal**: A fully usable browser UI without needing the WhatsApp sidecar.

Pages planned:
- `/` — Dashboard (queue depth, recent videos, disk usage)
- `/videos` — Paginated video list with platform/status filters
- `/videos/{id}` — Video detail (transcript, summary, audio delete button)
- `/submit` — Manual URL submission form
- `/jobs` — Job queue view

Tech: Jinja2 templates, HTMX for partial page updates (no full-page reloads), minimal vanilla CSS.
