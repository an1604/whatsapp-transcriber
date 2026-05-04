from __future__ import annotations

import re
from urllib.parse import urlparse

PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"https?://(?:www\.)?instagram\.com/reel/[\w-]+/?", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?instagram\.com/p/[\w-]+/?", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?instagram\.com/tv/[\w-]+/?", re.IGNORECASE),
    re.compile(r"https?://(?:www\.)?instagram\.com/reels/[\w-]+/?", re.IGNORECASE),
]

_TYPE_MAP = {
    "reel": "reel",
    "reels": "reel",
    "p": "p",
    "tv": "tv",
}


def canonicalize(url: str) -> str:
    """Strip tracking parameters and normalize an Instagram video URL.

    Raises ValueError if the path does not match a recognized video type.
    """
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]

    if len(parts) < 2:
        raise ValueError(f"Instagram URL has too few path components: {url!r}")

    content_type = parts[0].lower()
    if content_type not in _TYPE_MAP:
        raise ValueError(
            f"Instagram URL is not a recognized video type ({content_type!r}): {url!r}"
        )

    shortcode = parts[1]
    canonical_type = _TYPE_MAP[content_type]
    return f"https://www.instagram.com/{canonical_type}/{shortcode}/"
