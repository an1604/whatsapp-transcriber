from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import VideoORM, Database
from src.core.models import VideoStatus
from src.web.dependencies import get_database, get_session

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


class DashboardStats(BaseModel):
    total_videos: int
    complete: int
    failed: int
    in_progress: int
    discovered: int


@router.get("", response_model=DashboardStats)
async def get_dashboard_stats(
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """Return aggregate counts for the dashboard."""
    total_result = await session.execute(select(func.count()).select_from(VideoORM))
    total = total_result.scalar_one()

    status_counts: dict[str, int] = {}
    for st in VideoStatus:
        result = await session.execute(
            select(func.count()).select_from(VideoORM).where(
                VideoORM.status == st.value
            )
        )
        status_counts[st.value] = result.scalar_one()

    in_progress_statuses = {
        VideoStatus.DOWNLOADING, VideoStatus.DOWNLOADED,
        VideoStatus.TRANSCRIBING, VideoStatus.TRANSCRIBED,
        VideoStatus.SUMMARIZING,
    }
    in_progress = sum(
        status_counts.get(s.value, 0) for s in in_progress_statuses
    )

    return DashboardStats(
        total_videos=total,
        complete=status_counts.get(VideoStatus.COMPLETE.value, 0),
        failed=status_counts.get(VideoStatus.FAILED.value, 0),
        in_progress=in_progress,
        discovered=status_counts.get(VideoStatus.DISCOVERED.value, 0),
    )
