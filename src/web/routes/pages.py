"""Full-page HTML routes — each renders a Jinja2 template."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import Database
from src.core.exceptions import VideoNotFoundError
from src.core.models import VideoRead
from src.web.dependencies import get_database, get_session

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory="src/web/templates")


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {})


@router.get("/videos", response_class=HTMLResponse)
async def videos_page(
    request: Request,
    platform: Optional[str] = None,
    status: Optional[str] = None,
):
    return templates.TemplateResponse(
        request, "videos.html", {"platform": platform, "status": status}
    )


@router.get("/videos/{video_id}", response_class=HTMLResponse)
async def video_detail_page(
    request: Request,
    video_id: int,
    session: AsyncSession = Depends(get_session),
    db: Database = Depends(get_database),
):
    try:
        row = await db.get_video_by_id(session, video_id)
    except VideoNotFoundError:
        return templates.TemplateResponse(
            request, "404.html",
            {"detail": f"Video #{video_id} not found"},
            status_code=404,
        )
    video = VideoRead.model_validate(row).model_dump(mode="json")
    return templates.TemplateResponse(request, "video_detail.html", {"video": video})


@router.get("/submit", response_class=HTMLResponse)
async def submit_page(request: Request):
    return templates.TemplateResponse(request, "submit.html", {})


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    return templates.TemplateResponse(request, "jobs.html", {})


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse(request, "settings.html", {})
