from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Database
from src.core.exceptions import VideoNotFoundError
from src.core.models import Platform, VideoRead, VideoStatus
from src.web.dependencies import get_database, get_session

router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.get("", response_model=list[VideoRead])
async def list_videos(
    platform: Optional[Platform] = None,
    video_status: Optional[VideoStatus] = None,
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """List videos with optional filtering by platform and status."""
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
    rows = await db.list_videos(
        session,
        platform=platform,
        status=video_status,
        limit=limit,
        offset=offset,
    )
    return [VideoRead.model_validate(r) for r in rows]


@router.get("/{video_id}", response_model=VideoRead)
async def get_video(
    video_id: int,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """Return a single video by its database ID."""
    try:
        row = await db.get_video_by_id(session, video_id)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return VideoRead.model_validate(row)


@router.delete("/{video_id}/audio", response_model=VideoRead)
async def delete_audio(
    video_id: int,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """Mark the audio file as deleted (frees disk, keeps transcript + summary)."""
    try:
        row = await db.mark_audio_deleted(session, video_id)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return VideoRead.model_validate(row)
