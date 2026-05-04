from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.exceptions import URLExtractionError
from src.core.models import Platform, compute_url_hash
from src.downloaders.registry import canonicalize_url, detect_platform

# Broad URL regex used as a fast pre-filter before platform detection.
# Intentionally permissive — false positives are filtered by platform patterns.
_URL_PATTERN = re.compile(
    r"https?://"
    r"(?:[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%])+",
    re.IGNORECASE,
)

# Characters that frequently trail URLs in natural-language text
_TRAILING_JUNK = re.compile(r"[.,;:!?)\"']+$")


@dataclass(frozen=True)
class ExtractedURL:
    """A single URL extracted and resolved from a message."""

    raw_url: str
    canonical_url: str
    platform: Platform
    url_hash: str


def extract_urls(text: str) -> list[ExtractedURL]:
    """Extract all supported-platform video URLs from raw message text.

    - Finds candidate URLs via regex.
    - Strips trailing punctuation that gets appended in chat messages.
    - Detects platform and canonicalizes each URL.
    - Deduplicates by canonical URL hash (same video shared twice = one entry).
    - Ignores URLs from unsupported platforms silently.

    Raises URLExtractionError only for unexpected internal failures.
    The caller controls which platforms are enabled; this function always
    attempts all registered platforms.
    """
    if not isinstance(text, str):
        raise URLExtractionError(
            f"extract_urls expects a str, got {type(text).__name__!r}"
        )

    raw_candidates = _URL_PATTERN.findall(text)
    seen_hashes: set[str] = set()
    results: list[ExtractedURL] = []

    for raw in raw_candidates:
        url = _TRAILING_JUNK.sub("", raw)
        if not url:
            continue

        try:
            platform, canonical = canonicalize_url(url)
        except Exception:
            # URL belongs to an unsupported platform or cannot be canonicalized —
            # skip silently, this is the expected path for non-video URLs.
            continue

        url_hash = compute_url_hash(canonical)
        if url_hash in seen_hashes:
            continue

        seen_hashes.add(url_hash)
        results.append(
            ExtractedURL(
                raw_url=raw,
                canonical_url=canonical,
                platform=platform,
                url_hash=url_hash,
            )
        )

    return results


def extract_single_url(url: str) -> ExtractedURL:
    """Canonicalize a single URL that the caller knows is a video URL.

    Unlike extract_urls, this raises PlatformNotSupportedError or
    URLCanonicalizationError if the URL cannot be processed — it is intended
    for the manual URL submission flow where a bad URL is an error.
    """
    url = _TRAILING_JUNK.sub("", url.strip())
    platform, canonical = canonicalize_url(url)  # raises on failure
    return ExtractedURL(
        raw_url=url,
        canonical_url=canonical,
        platform=platform,
        url_hash=compute_url_hash(canonical),
    )
