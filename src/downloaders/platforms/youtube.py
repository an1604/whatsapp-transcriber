from __future__ import annotations

import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Patterns that match YouTube video URLs (not channels/playlists/etc.)
PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"https?://(?:www\.)?youtube\.com/watch\?.*v=[\w-]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?youtube\.com/shorts/[\w-]+", re.IGNORECASE),
    re.compile(r"https?://youtu\.be/[\w-]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?youtube\.com/embed/[\w-]+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?youtube\.com/v/[\w-]+", re.IGNORECASE),
    re.compile(r"https?://(?:m\.)?youtube\.com/watch\?.*v=[\w-]+", re.IGNORECASE),
]

_ALLOWED_PARAMS = frozenset({"v", "list", "index"})


def canonicalize(url: str) -> str:
    """Normalize a YouTube URL to its canonical form, stripping tracking params.

    Converts youtu.be short links to youtube.com/watch?v=... form.
    Strips everything except v= (and playlist list= / index= if present).
    Raises ValueError if the URL contains no video ID.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower().lstrip("www.").lstrip("m.")

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0].split("?")[0]
        if not video_id:
            raise ValueError(f"Cannot extract video ID from youtu.be URL: {url!r}")
        return f"https://www.youtube.com/watch?v={video_id}"

    if "youtube.com" in host:
        path = parsed.path.rstrip("/")

        if "/shorts/" in path:
            video_id = path.split("/shorts/")[1].split("/")[0]
            if not video_id:
                raise ValueError(f"Cannot extract video ID from Shorts URL: {url!r}")
            return f"https://www.youtube.com/shorts/{video_id}"

        if path in ("/watch", "/watch/") or path == "":
            params = parse_qs(parsed.query)
            if "v" not in params:
                raise ValueError(f"YouTube watch URL missing v= parameter: {url!r}")
            video_id = params["v"][0]
            kept = {k: v[0] for k, v in params.items() if k in _ALLOWED_PARAMS}
            return f"https://www.youtube.com/watch?{urlencode(kept)}"

        if path.startswith("/embed/") or path.startswith("/v/"):
            parts = path.lstrip("/").split("/")
            video_id = parts[1] if len(parts) > 1 else ""
            if not video_id:
                raise ValueError(f"Cannot extract video ID from embed/v URL: {url!r}")
            return f"https://www.youtube.com/watch?v={video_id}"

    raise ValueError(f"URL does not look like a YouTube video URL: {url!r}")
