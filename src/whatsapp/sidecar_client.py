"""HTTP client for communicating with the Node.js WhatsApp sidecar."""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from src.core.exceptions import SidecarError, SidecarNotReadyError


@dataclass(frozen=True)
class SidecarStatus:
    whatsapp_ready: bool
    community_ids: list[str]
    pipeline_url: str


class SidecarClient:
    """Async HTTP client for the Node.js WhatsApp sidecar.

    Raises SidecarError on every failure — never silently returns None.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url must not be empty")
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, object]:
        """GET /health — returns raw JSON.

        Raises SidecarError if unreachable or non-2xx.
        """
        return await self._get("/health")

    async def status(self) -> SidecarStatus:
        """GET /status — returns structured status.

        Raises SidecarNotReadyError if WhatsApp is not yet authenticated.
        Raises SidecarError on network/HTTP failures.
        """
        data = await self._get("/status")
        ready = bool(data.get("whatsapp_ready", False))
        if not ready:
            raise SidecarNotReadyError(
                "WhatsApp sidecar is not ready — scan the QR code to authenticate"
            )
        return SidecarStatus(
            whatsapp_ready=True,
            community_ids=list(data.get("community_ids", [])),
            pipeline_url=str(data.get("pipeline_url", "")),
        )

    async def is_ready(self) -> bool:
        """Return True if the sidecar is up AND WhatsApp is authenticated.

        Never raises — catches all exceptions and returns False.
        """
        try:
            data = await self._get("/health")
            return bool(data.get("whatsapp_ready", False))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str) -> dict[str, object]:
        url = f"{self._base}{path}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url)
        except httpx.TimeoutException as exc:
            raise SidecarError(f"Sidecar request timed out: {url}") from exc
        except httpx.RequestError as exc:
            raise SidecarError(f"Sidecar unreachable at {url}: {exc}") from exc

        if not response.is_success:
            raise SidecarError(
                f"Sidecar returned HTTP {response.status_code} for {url}: "
                f"{response.text[:200]}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise SidecarError(
                f"Sidecar returned invalid JSON from {url}: {exc}"
            ) from exc
