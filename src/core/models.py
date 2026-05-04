from __future__ import annotations

import enum
import hashlib
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


class VideoStatus(str, enum.Enum):
    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    TRANSCRIBING = "transcribing"
    TRANSCRIBED = "transcribed"
    SUMMARIZING = "summarizing"
    COMPLETE = "complete"
    FAILED = "failed"


# Explicit allowed transitions — no implicit fallbacks anywhere in the pipeline.
VALID_STATUS_TRANSITIONS: dict[VideoStatus, frozenset[VideoStatus]] = {
    VideoStatus.DISCOVERED:  frozenset({VideoStatus.DOWNLOADING, VideoStatus.FAILED}),
    VideoStatus.DOWNLOADING: frozenset({VideoStatus.DOWNLOADED,  VideoStatus.FAILED}),
    VideoStatus.DOWNLOADED:  frozenset({VideoStatus.TRANSCRIBING, VideoStatus.FAILED}),
    VideoStatus.TRANSCRIBING: frozenset({VideoStatus.TRANSCRIBED, VideoStatus.FAILED}),
    VideoStatus.TRANSCRIBED: frozenset({VideoStatus.SUMMARIZING, VideoStatus.FAILED}),
    VideoStatus.SUMMARIZING: frozenset({VideoStatus.COMPLETE,    VideoStatus.FAILED}),
    VideoStatus.COMPLETE:    frozenset(),           # terminal — no further transitions
    VideoStatus.FAILED:      frozenset({VideoStatus.DISCOVERED}),  # retry path
}


class Platform(str, enum.Enum):
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    FACEBOOK = "facebook"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, enum.Enum):
    DOWNLOAD = "download"
    TRANSCRIBE = "transcribe"
    SUMMARIZE = "summarize"


def compute_url_hash(canonical_url: str) -> str:
    """Return the SHA-256 hex digest of the canonical URL string."""
    return hashlib.sha256(canonical_url.encode()).hexdigest()


class VideoCreate(BaseModel):
    canonical_url: str
    url_hash: str
    original_url: str
    platform: Platform
    source_group: Optional[str] = None
    source_message_id: Optional[str] = None

    @field_validator("canonical_url")
    @classmethod
    def canonical_url_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("canonical_url must not be empty or blank")
        return v

    @field_validator("url_hash")
    @classmethod
    def url_hash_must_be_sha256_hex(cls, v: str) -> str:
        normalized = v.lower()
        if len(normalized) != 64 or not all(c in "0123456789abcdef" for c in normalized):
            raise ValueError(
                "url_hash must be a 64-character lowercase hex string (SHA-256 digest)"
            )
        return normalized


class VideoRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    canonical_url: str
    url_hash: str
    original_url: str
    platform: Platform
    title: Optional[str]
    transcript: Optional[str]
    summary: Optional[str]
    detected_language: Optional[str]
    status: VideoStatus
    audio_path: Optional[str]
    audio_deleted: bool
    source_group: Optional[str]
    source_message_id: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]


class VideoUpdate(BaseModel):
    title: Optional[str] = None
    transcript: Optional[str] = None
    summary: Optional[str] = None
    detected_language: Optional[str] = None
    audio_path: Optional[str] = None
    error_message: Optional[str] = None


class JobCreate(BaseModel):
    video_id: int
    job_type: JobType


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    video_id: int
    job_type: JobType
    status: JobStatus
    error: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
