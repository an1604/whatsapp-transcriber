from __future__ import annotations

import re
from urllib.parse import urlparse

PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"https?://(?:www\.)?facebook\.com/reel/\d+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?facebook\.com/watch/?(?:\?v=\d+)?", re.IGNORECASE),
    re.compile(r"https?://fb\.watch/[\w]+/?", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?facebook\.com/[\w.]+/videos/\d+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?facebook\.com/video\.php\?v=\d+", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?facebook\.com/share/v/[\w]+/?", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?facebook\.com/share/r/[\w]+/?", re.IGNORECASE),
]

_SHORT_HOST = "fb.watch"


def canonicalize(url: str) -> str:
    """Normalize a Facebook video/reel URL.

    Short links (fb.watch) are kept as-is; yt-dlp resolves the redirect.
    Long-form URLs are normalized by stripping tracking params.

    Raises ValueError if the URL is not a recognized Facebook video URL.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if _SHORT_HOST in host:
        slug = parsed.path.strip("/")
        if not slug:
            raise ValueError(f"fb.watch URL has no slug: {url!r}")
        return f"https://fb.watch/{slug}"

    if "facebook.com" in host:
        path = parsed.path.rstrip("/")

        # /reel/ID
        m_reel = re.match(r"^/reel/(\d+)$", path)
        if m_reel:
            return f"https://www.facebook.com/reel/{m_reel.group(1)}"

        # /watch?v=ID or /watch/
        if path in ("/watch", "/watch/"):
            from urllib.parse import parse_qs

            params = parse_qs(parsed.query)
            if "v" in params:
                return f"https://www.facebook.com/watch?v={params['v'][0]}"
            return "https://www.facebook.com/watch/"

        # /username/videos/ID
        m_vid = re.match(r"^/([\w.]+)/videos/(\d+)$", path)
        if m_vid:
            return f"https://www.facebook.com/{m_vid.group(1)}/videos/{m_vid.group(2)}"

        # /video.php?v=ID
        if path == "/video.php":
            from urllib.parse import parse_qs

            params = parse_qs(parsed.query)
            if "v" in params:
                return f"https://www.facebook.com/video.php?v={params['v'][0]}"

        # /share/v/SLUG or /share/r/SLUG
        m_share = re.match(r"^/share/([vr])/([\w]+)$", path)
        if m_share:
            kind, slug = m_share.group(1), m_share.group(2)
            return f"https://www.facebook.com/share/{kind}/{slug}"

    raise ValueError(f"URL does not look like a Facebook video URL: {url!r}")
