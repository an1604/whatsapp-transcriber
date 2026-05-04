"""Inbox REST routes — list, process, and dismiss discovered videos."""
from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Database
from src.core.exceptions import (
    PipelineNotConfiguredError,
    VideoNotFoundError,
)
from src.core.models import VideoRead, VideoStatus
from src.web.dependencies import get_database, get_orchestrator, get_session

router = APIRouter(prefix="/api/inbox", tags=["inbox"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class InboxListResponse(BaseModel):
    videos: list[VideoRead]
    total: int
    limit: int
    offset: int


class BulkProcessRequest(BaseModel):
    video_ids: list[int]


class BulkProcessResponse(BaseModel):
    queued_count: int
    video_ids: list[int]


class BulkDismissRequest(BaseModel):
    video_ids: list[int]


class BulkDismissResponse(BaseModel):
    dismissed_count: int
    video_ids: list[int]


# ---------------------------------------------------------------------------
# Background-task helper
# ---------------------------------------------------------------------------

async def _run_pipeline_bg(video_id: int, db: Database, orchestrator) -> None:
    async with db._session_factory() as session:
        await orchestrator.process_video_by_id(session, video_id)


# ---------------------------------------------------------------------------
# Routes — static paths BEFORE parameterised ones to avoid shadowing
# ---------------------------------------------------------------------------

@router.get("", response_model=InboxListResponse)
async def list_inbox(
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """List all videos in 'discovered' state awaiting manual review."""
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="limit must be between 1 and 500",
        )
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="offset must be >= 0",
        )
    rows = await db.list_inbox_videos(session, limit=limit, offset=offset)
    total = await db.count_inbox_videos(session)
    return InboxListResponse(
        videos=[VideoRead.model_validate(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/count")
async def inbox_count(
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """Return the count of videos currently awaiting review."""
    count = await db.count_inbox_videos(session)
    return {"count": count}


@router.post(
    "/bulk-process",
    response_model=BulkProcessResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def bulk_process(
    body: BulkProcessRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """Queue multiple discovered videos for processing."""
    if not body.video_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="video_ids must not be empty",
        )
    try:
        orchestrator = get_orchestrator()
    except PipelineNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    queued: list[int] = []
    for vid_id in body.video_ids:
        try:
            video = await db.get_video_by_id(session, vid_id)
        except VideoNotFoundError:
            continue
        if video.status != VideoStatus.DISCOVERED.value:
            continue
        background_tasks.add_task(_run_pipeline_bg, vid_id, db, orchestrator)
        queued.append(vid_id)

    return BulkProcessResponse(queued_count=len(queued), video_ids=queued)


@router.post(
    "/bulk-dismiss",
    response_model=BulkDismissResponse,
)
async def bulk_dismiss(
    body: BulkDismissRequest,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """Hard-delete multiple videos from the inbox."""
    if not body.video_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="video_ids must not be empty",
        )
    count = await db.bulk_delete_videos(session, body.video_ids)
    return BulkDismissResponse(dismissed_count=count, video_ids=body.video_ids)


@router.post(
    "/{video_id}/process",
    status_code=status.HTTP_202_ACCEPTED,
)
async def process_video(
    video_id: int,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """Trigger pipeline processing for a single discovered video.

    Returns 503 if no orchestrator is configured.
    Returns 404 if the video does not exist.
    Returns 409 if the video is not in 'discovered' status.
    """
    try:
        orchestrator = get_orchestrator()
    except PipelineNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    try:
        video = await db.get_video_by_id(session, video_id)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    if video.status != VideoStatus.DISCOVERED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Video id={video_id} has status={video.status!r}; "
                f"only 'discovered' videos can be queued for processing"
            ),
        )

    background_tasks.add_task(_run_pipeline_bg, video_id, db, orchestrator)
    return {"status": "queued", "video_id": video_id}


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def dismiss_video(
    video_id: int,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """Hard-delete a video (dismiss from inbox). Also removes all its jobs."""
    try:
        await db.delete_video(session, video_id)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
