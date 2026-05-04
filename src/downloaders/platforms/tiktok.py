from __future__ import annotations

import re
from urllib.parse import urlparse

PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"https?://(?:www\.)?tiktok\.com/@[\w.-]+/video/\d+", re.IGNORECASE),
    re.compile(r"https?://vm\.tiktok\.com/[\w]+/?", re.IGNORECASE),
    re.compile(r"https?://vt\.tiktok\.com/[\w]+/?", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?tiktok\.com/t/[\w]+/?", re.IGNORECASE),
    re.compile(r"https?://m\.tiktok\.com/v/\d+", re.IGNORECASE),
]

_SHORT_LINK_HOSTS = frozenset({"vm.tiktok.com", "vt.tiktok.com"})


def canonicalize(url: str) -> str:
    """Normalize a TikTok video URL.

    Short links (vm.tiktok.com, vt.tiktok.com) are kept as-is because resolving
    the redirect requires a network call — the downloader (yt-dlp) handles that.
    Long-form URLs are normalized by stripping query params.

    Raises ValueError if the URL is not a recognized TikTok video URL.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if host in _SHORT_LINK_HOSTS:
        slug = parsed.path.strip("/")
        if not slug:
            raise ValueError(f"TikTok short link has no slug: {url!r}")
        return f"https://{host}/{slug}"

    if "tiktok.com" in host:
        path = parsed.path.rstrip("/")

        # @username/video/ID
        m = re.match(r"^/@([\w.-]+)/video/(\d+)$", path)
        if m:
            username, video_id = m.group(1), m.group(2)
            return f"https://www.tiktok.com/@{username}/video/{video_id}"

        # /t/SLUG (share links)
        m2 = re.match(r"^/t/([\w]+)$", path)
        if m2:
            return f"https://www.tiktok.com/t/{m2.group(1)}"

        # m.tiktok.com/v/ID
        m3 = re.match(r"^/v/(\d+)", path)
        if m3:
            return f"https://www.tiktok.com/video/{m3.group(1)}"

    raise ValueError(f"URL does not look like a TikTok video URL: {url!r}")
