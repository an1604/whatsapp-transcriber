from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Database
from src.core.exceptions import JobNotFoundError, VideoNotFoundError
from src.core.models import JobRead
from src.web.dependencies import get_database, get_session

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/video/{video_id}", response_model=list[JobRead])
async def list_jobs_for_video(
    video_id: int,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """Return all pipeline jobs for a given video."""
    try:
        rows = await db.list_jobs_for_video(session, video_id)
    except VideoNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return [JobRead.model_validate(r) for r in rows]


@router.get("/{job_id}", response_model=JobRead)
async def get_job(
    job_id: int,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """Return a single job by its database ID."""
    try:
        row = await db.get_job_by_id(session, job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return JobRead.model_validate(row)
