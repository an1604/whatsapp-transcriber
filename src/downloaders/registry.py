from __future__ import annotations

import re
from typing import Callable

from src.core.exceptions import PlatformNotSupportedError
from src.core.models import Platform
from src.downloaders.platforms import facebook, instagram, tiktok, youtube

# Each entry: (Platform, list-of-patterns, canonicalize-fn)
_PLATFORM_REGISTRY: list[
    tuple[Platform, list[re.Pattern[str]], Callable[[str], str]]
] = [
    (Platform.YOUTUBE, youtube.PATTERNS, youtube.canonicalize),
    (Platform.INSTAGRAM, instagram.PATTERNS, instagram.canonicalize),
    (Platform.TIKTOK, tiktok.PATTERNS, tiktok.canonicalize),
    (Platform.FACEBOOK, facebook.PATTERNS, facebook.canonicalize),
]


def detect_platform(url: str) -> Platform:
    """Return the Platform for the given URL.

    Raises PlatformNotSupportedError if no registered platform matches.
    """
    for platform, patterns, _ in _PLATFORM_REGISTRY:
        for pattern in patterns:
            if pattern.search(url):
                return platform
    raise PlatformNotSupportedError(
        f"No supported platform matched URL: {url!r}. "
        f"Supported platforms: {[p.value for p, _, _ in _PLATFORM_REGISTRY]}"
    )


def canonicalize_url(url: str) -> tuple[Platform, str]:
    """Detect platform and return (Platform, canonical_url).

    Raises PlatformNotSupportedError if no platform matches.
    Raises URLCanonicalizationError if the platform-specific canonicalizer fails.
    """
    from src.core.exceptions import URLCanonicalizationError

    for platform, patterns, canonicalize_fn in _PLATFORM_REGISTRY:
        for pattern in patterns:
            if pattern.search(url):
                try:
                    canonical = canonicalize_fn(url)
                except ValueError as exc:
                    raise URLCanonicalizationError(
                        f"Failed to canonicalize {platform.value} URL {url!r}: {exc}"
                    ) from exc
                return platform, canonical

    raise PlatformNotSupportedError(
        f"No supported platform matched URL: {url!r}. "
        f"Supported platforms: {[p.value for p, _, _ in _PLATFORM_REGISTRY]}"
    )


def get_enabled_platforms() -> list[Platform]:
    """Return all platforms registered in the registry."""
    return [platform for platform, _, _ in _PLATFORM_REGISTRY]
