"""HTMX partial-response routes — return HTML fragments, not full pages."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Database, JobORM, VideoORM
from src.core.exceptions import (
    DuplicateVideoError,
    PipelineNotConfiguredError,
    PlatformNotSupportedError,
    URLCanonicalizationError,
    VideoNotFoundError,
)
from src.core.models import (
    JobRead,
    Platform,
    VideoRead,
    VideoStatus,
    compute_url_hash,
)
from src.scrapers.url_extractor import extract_single_url
from src.web.dependencies import get_database, get_orchestrator, get_session

router = APIRouter(prefix="/htmx", tags=["htmx"])
templates = Jinja2Templates(directory="src/web/templates")


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

@router.get("/dashboard/stats", response_class=HTMLResponse)
async def htmx_dashboard_stats(
    request: Request,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    total = (await session.execute(
        select(func.count()).select_from(VideoORM)
    )).scalar_one()

    status_counts: dict[str, int] = {}
    for st in VideoStatus:
        cnt = (await session.execute(
            select(func.count()).select_from(VideoORM).where(VideoORM.status == st.value)
        )).scalar_one()
        status_counts[st.value] = cnt

    in_progress_statuses = {
        VideoStatus.DOWNLOADING, VideoStatus.DOWNLOADED,
        VideoStatus.TRANSCRIBING, VideoStatus.TRANSCRIBED,
        VideoStatus.SUMMARIZING,
    }
    in_progress = sum(status_counts.get(s.value, 0) for s in in_progress_statuses)

    disk = await db.get_disk_stats(session)
    stats = {
        "total_videos": total,
        "complete": status_counts.get(VideoStatus.COMPLETE.value, 0),
        "failed": status_counts.get(VideoStatus.FAILED.value, 0),
        "in_progress": in_progress,
        "discovered": status_counts.get(VideoStatus.DISCOVERED.value, 0),
        "videos_with_audio": disk["videos_with_audio"],
        "videos_audio_deleted": disk["videos_audio_deleted"],
    }
    return templates.TemplateResponse(
        request, "partials/dashboard_stats.html", {"stats": stats}
    )


# ---------------------------------------------------------------------------
# Recent videos (dashboard)
# ---------------------------------------------------------------------------

@router.get("/videos/recent", response_class=HTMLResponse)
async def htmx_recent_videos(
    request: Request,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    rows = await db.list_videos(session, limit=10)
    videos = [VideoRead.model_validate(r).model_dump(mode="json") for r in rows]
    return templates.TemplateResponse(
        request,
        "partials/video_table.html",
        {"videos": videos, "show_pagination": False, "limit": 10, "offset": 0},
    )


# ---------------------------------------------------------------------------
# Video table (videos page, filterable + paginated)
# ---------------------------------------------------------------------------

@router.get("/videos/table", response_class=HTMLResponse)
async def htmx_video_table(
    request: Request,
    platform: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    if limit < 1 or limit > 200:
        raise ValueError(f"limit must be between 1 and 200, got {limit}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")

    platform_enum = Platform(platform) if platform else None
    status_enum = VideoStatus(status) if status else None

    rows = await db.list_videos(
        session, platform=platform_enum, status=status_enum, limit=limit, offset=offset
    )
    videos = [VideoRead.model_validate(r).model_dump(mode="json") for r in rows]
    return templates.TemplateResponse(
        request,
        "partials/video_table.html",
        {
            "videos": videos,
            "show_pagination": True,
            "limit": limit,
            "offset": offset,
            "platform": platform or "",
            "status": status or "",
        },
    )


# ---------------------------------------------------------------------------
# Delete audio (video detail page)
# ---------------------------------------------------------------------------

@router.delete("/videos/{video_id}/audio", response_class=HTMLResponse)
async def htmx_delete_audio(
    request: Request,
    video_id: int,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    try:
        await db.mark_audio_deleted(session, video_id)
    except VideoNotFoundError as exc:
        return HTMLResponse(
            f'<span class="badge badge-failed">Error: {exc}</span>',
            status_code=404,
        )
    return HTMLResponse('<span class="text-muted">Deleted</span>')


# ---------------------------------------------------------------------------
# Jobs table (jobs page)
# ---------------------------------------------------------------------------

@router.get("/jobs/recent", response_class=HTMLResponse)
async def htmx_recent_jobs(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    stmt = (
        select(JobORM)
        .order_by(JobORM.created_at.desc())
        .limit(50)
    )
    rows = list((await session.execute(stmt)).scalars().all())
    jobs = [JobRead.model_validate(r).model_dump(mode="json") for r in rows]
    return templates.TemplateResponse(
        request, "partials/jobs_table.html", {"jobs": jobs}
    )


@router.get("/jobs/video/{video_id}", response_class=HTMLResponse)
async def htmx_jobs_for_video(
    request: Request,
    video_id: int,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    try:
        rows = await db.list_jobs_for_video(session, video_id)
    except VideoNotFoundError as exc:
        return HTMLResponse(f'<p class="alert alert-error">{exc}</p>', status_code=404)
    jobs = [JobRead.model_validate(r).model_dump(mode="json") for r in rows]
    return templates.TemplateResponse(
        request, "partials/jobs_table.html", {"jobs": jobs}
    )


# ---------------------------------------------------------------------------
# Search (videos page)
# ---------------------------------------------------------------------------

@router.get("/videos/search", response_class=HTMLResponse)
async def htmx_search_videos(
    request: Request,
    q: Optional[str] = None,
    limit: int = 25,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    if not q or not q.strip():
        return templates.TemplateResponse(
            request,
            "partials/video_table.html",
            {"videos": [], "show_pagination": False, "limit": limit, "offset": 0, "q": ""},
        )
    if limit < 1 or limit > 200:
        raise ValueError(f"limit must be between 1 and 200, got {limit}")
    rows = await db.search_videos(session, q, limit=limit, offset=offset)
    videos = [VideoRead.model_validate(r).model_dump(mode="json") for r in rows]
    return templates.TemplateResponse(
        request,
        "partials/video_table.html",
        {
            "videos": videos,
            "show_pagination": True,
            "limit": limit,
            "offset": offset,
            "q": q,
        },
    )


# ---------------------------------------------------------------------------
# Bulk delete audio (videos page)
# ---------------------------------------------------------------------------

@router.post("/videos/bulk-delete-audio", response_class=HTMLResponse)
async def htmx_bulk_delete_audio(
    request: Request,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    form = await request.form()
    raw_ids = form.getlist("video_ids")
    try:
        video_ids = [int(v) for v in raw_ids if v]
    except ValueError:
        return HTMLResponse('<span class="badge badge-failed">Error: invalid video ID</span>', status_code=400)

    if not video_ids:
        return HTMLResponse('<span class="text-muted">No videos selected.</span>')

    count = await db.bulk_mark_audio_deleted(session, video_ids)
    return HTMLResponse(
        f'<span class="badge badge-complete">{count} audio file(s) deleted.</span>'
    )


# ---------------------------------------------------------------------------
# Submit URL form (submit page)
# ---------------------------------------------------------------------------

@router.post("/submit", response_class=HTMLResponse)
async def htmx_submit_url(
    request: Request,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    form = await request.form()
    raw_url = str(form.get("url", "")).strip()
    source_group = str(form.get("source_group", "")).strip() or None

    if not raw_url:
        return templates.TemplateResponse(
            request, "partials/submit_result.html",
            {"error": "URL must not be empty"},
        )

    try:
        extracted = extract_single_url(raw_url)
    except (PlatformNotSupportedError, URLCanonicalizationError) as exc:
        return templates.TemplateResponse(
            request, "partials/submit_result.html", {"error": str(exc)}
        )

    from src.core.models import VideoCreate  # local import to avoid circular

    try:
        video = await db.create_video(
            session,
            VideoCreate(
                canonical_url=extracted.canonical_url,
                url_hash=extracted.url_hash,
                original_url=extracted.raw_url,
                platform=extracted.platform,
                source_group=source_group,
            ),
        )
        is_duplicate = False
    except DuplicateVideoError:
        stmt = select(VideoORM).where(VideoORM.url_hash == extracted.url_hash)
        row = (await session.execute(stmt)).scalar_one()
        video = row
        is_duplicate = True

    result = {
        "video_id": video.id,
        "canonical_url": extracted.canonical_url,
        "platform": extracted.platform.value,
        "is_duplicate": is_duplicate,
    }
    return templates.TemplateResponse(
        request, "partials/submit_result.html", {"result": result}
    )


# ---------------------------------------------------------------------------
# Inbox — discovered videos awaiting manual review
# ---------------------------------------------------------------------------

async def _run_pipeline_bg(video_id: int, db: Database, orchestrator) -> None:
    async with db._session_factory() as session:
        await orchestrator.process_video_by_id(session, video_id)


@router.get("/inbox", response_class=HTMLResponse)
async def htmx_inbox_table(
    request: Request,
    limit: int = 25,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    rows = await db.list_inbox_videos(session, limit=limit, offset=offset)
    total = await db.count_inbox_videos(session)
    videos = [VideoRead.model_validate(r).model_dump(mode="json") for r in rows]
    return templates.TemplateResponse(
        request,
        "partials/inbox_table.html",
        {"videos": videos, "total": total, "limit": limit, "offset": offset},
    )


# Static segments (bulk-process, bulk-dismiss) must come BEFORE /{video_id}/…

@router.post("/inbox/bulk-process", response_class=HTMLResponse)
async def htmx_bulk_process(
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    try:
        orchestrator = get_orchestrator()
    except PipelineNotConfiguredError:
        return HTMLResponse(
            '<span class="badge badge-failed">Pipeline not configured — '
            "start the app with config_path to enable processing.</span>",
            status_code=503,
        )

    form = await request.form()
    raw_ids = form.getlist("video_ids")
    try:
        video_ids = [int(v) for v in raw_ids if v]
    except ValueError:
        return HTMLResponse(
            '<span class="badge badge-failed">Invalid video ID in selection.</span>',
            status_code=400,
        )

    if not video_ids:
        return HTMLResponse('<span class="text-muted">No videos selected.</span>')

    queued = 0
    for vid_id in video_ids:
        try:
            video = await db.get_video_by_id(session, vid_id)
            if video.status == VideoStatus.DISCOVERED.value:
                background_tasks.add_task(_run_pipeline_bg, vid_id, db, orchestrator)
                queued += 1
        except VideoNotFoundError:
            continue

    return HTMLResponse(
        f'<span class="badge badge-complete">{queued} video(s) queued for processing.</span>',
        headers={"HX-Trigger": "inbox-updated"},
    )


@router.post("/inbox/bulk-dismiss", response_class=HTMLResponse)
async def htmx_bulk_dismiss(
    request: Request,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    form = await request.form()
    raw_ids = form.getlist("video_ids")
    try:
        video_ids = [int(v) for v in raw_ids if v]
    except ValueError:
        return HTMLResponse(
            '<span class="badge badge-failed">Invalid video ID in selection.</span>',
            status_code=400,
        )

    if not video_ids:
        return HTMLResponse('<span class="text-muted">No videos selected.</span>')

    count = await db.bulk_delete_videos(session, video_ids)
    return HTMLResponse(
        f'<span class="badge badge-complete">{count} video(s) dismissed.</span>',
        headers={"HX-Trigger": "inbox-updated"},
    )


@router.post("/inbox/{video_id}/process", response_class=HTMLResponse)
async def htmx_process_video(
    request: Request,
    video_id: int,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    try:
        orchestrator = get_orchestrator()
    except PipelineNotConfiguredError:
        return HTMLResponse(
            f'<tr id="inbox-row-{video_id}"><td colspan="6">'
            '<span class="badge badge-failed">Pipeline not configured.</span>'
            "</td></tr>",
            status_code=503,
        )

    try:
        video = await db.get_video_by_id(session, video_id)
    except VideoNotFoundError:
        return HTMLResponse("", status_code=404)

    if video.status != VideoStatus.DISCOVERED.value:
        return HTMLResponse(
            f'<tr id="inbox-row-{video_id}"><td colspan="6">'
            f'<span class="badge badge-progress">Already processing ({video.status})</span>'
            "</td></tr>",
            status_code=409,
        )

    background_tasks.add_task(_run_pipeline_bg, video_id, db, orchestrator)
    return templates.TemplateResponse(
        request,
        "partials/inbox_row.html",
        {
            "video": VideoRead.model_validate(video).model_dump(mode="json"),
            "queued": True,
        },
    )


@router.delete("/inbox/{video_id}", response_class=HTMLResponse)
async def htmx_dismiss_video(
    request: Request,
    video_id: int,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    try:
        await db.delete_video(session, video_id)
    except VideoNotFoundError:
        return HTMLResponse("", status_code=404)
    return HTMLResponse("")
