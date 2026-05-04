from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Database
from src.core.exceptions import (
    DuplicateVideoError,
    PlatformNotSupportedError,
    URLCanonicalizationError,
)
from src.core.models import VideoRead
from src.scrapers.url_extractor import extract_single_url
from src.web.dependencies import get_database, get_session

router = APIRouter(prefix="/api/submit", tags=["submit"])


class SubmitURLRequest(BaseModel):
    url: str
    source_group: str | None = None


class SubmitURLResponse(BaseModel):
    video_id: int
    canonical_url: str
    platform: str
    is_duplicate: bool


@router.post("", response_model=SubmitURLResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_url(
    body: SubmitURLRequest,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    """Manually submit a video URL for processing.

    Returns 202 if the URL was accepted (new or duplicate).
    Returns 422 if the URL cannot be recognized or canonicalized.
    """
    url = body.url.strip()
    if not url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="url must not be empty",
        )

    try:
        extracted = extract_single_url(url)
    except (PlatformNotSupportedError, URLCanonicalizationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        )

    try:
        video = await db.create_video(
            session,
            __import__("src.core.models", fromlist=["VideoCreate"]).VideoCreate(
                canonical_url=extracted.canonical_url,
                url_hash=extracted.url_hash,
                original_url=extracted.raw_url,
                platform=extracted.platform,
                source_group=body.source_group,
            ),
        )
        is_duplicate = False
    except DuplicateVideoError:
        from src.core.database import VideoORM
        from sqlalchemy import select
        stmt = select(VideoORM).where(VideoORM.url_hash == extracted.url_hash)
        row = (await session.execute(stmt)).scalar_one()
        video = row
        is_duplicate = True

    return SubmitURLResponse(
        video_id=video.id,
        canonical_url=extracted.canonical_url,
        platform=extracted.platform.value,
        is_duplicate=is_duplicate,
    )
